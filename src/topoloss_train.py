import argparse
import yaml
from models import build_model
from datasets import get_dataloaders, get_patch_dataloaders
import torch
import numpy as np
from tqdm import tqdm
import torch.nn as nn
import os
from metrics import compute_dice, compute_iou, topographic_loss, combined_topo_tversky_dice, bce_dice_topographic_loss
from utils import save_metadata
from torch.optim.lr_scheduler import ReduceLROnPlateau
import logging
import matplotlib.pyplot as plt
import datetime
import random
from PIL import Image

def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

def parse_args():
    parser = argparse.ArgumentParser(description="Train a segmentation model with a specified encoder.")

    parser.add_argument('--config', '-c', type=str, required=True, 
                        help='Path to the configuration file')
    parser.add_argument('--experiment_name', '-e', type=str, default='unnamed_experiment_batched', 
                        help='Name of the experiment for logging purposes')

    return parser.parse_args()

# Focal Tversky Loss with recommended default params
def criterion(preds, targets, distance_maps):
    return bce_dice_topographic_loss(
        preds=preds,
        targets=targets,
        distance_maps=distance_maps,
        alpha=2.0,
        dice_weight=0.5,
        beta=1.0
    )

    return combined_topo_tversky_dice(
        preds, targets, distance_maps,
        topo_alpha=2.0,
        tv_alpha=0.7,
        tv_beta=0.3,
        tv_gamma=0.75,
        dice_weight=0.5
    )
    return topographic_loss(preds, targets, distance_maps,
                 base_loss='bce',
                 alpha=2.0,
                 beta=1.0)


def train_model(model, train_loader, test_loader, conf):
    model = model.cuda()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=conf['train']['learning_rate'],
    )

    scheduler = ReduceLROnPlateau(
        optimizer,
        mode='min',
        patience=10,
        factor=0.5,
        min_lr=1e-6
    )

    best_dice = 0.0
    best_val_loss = float('inf')
    best_epoch = 0
    best_model_path = None
    counter = 0
    accumulation_steps = conf['train'].get('accumulation_steps', 1)
    conf['train']['num_epochs'] = 1
    for epoch in tqdm(range(1, conf['train']['num_epochs'] + 1), desc="Training Epochs", leave=False, unit='epoch'):
        model.train()
        train_loss = 0.0
        optimizer.zero_grad()

        for batch_idx, (images, masks, distances) in enumerate(train_loader):
            images = images.cuda()
            masks = masks.cuda()
            distances = distances.cuda()
            preds = model(images)

            assert preds.shape == masks.shape, f"Mismatch: preds {preds.shape}, masks {masks.shape}"

            loss = criterion(preds, masks, distances) / accumulation_steps
            loss.backward()

            if (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1) == len(train_loader):
                optimizer.step()
                optimizer.zero_grad()

            train_loss += loss.item() * accumulation_steps

        avg_train_loss = train_loss / len(train_loader)

        # --- Validation Phase using test_model() ---
        model.eval()
        avg_val_loss, avg_dice, avg_iou = test_model(model, test_loader, conf, return_metrics_only=True)
        scheduler.step(avg_val_loss)

        # --- Update best metrics and save model ---
        if avg_dice > best_dice:
            best_dice = avg_dice
            best_epoch = epoch

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            counter = 0

            if best_model_path and os.path.exists(best_model_path):
                os.remove(best_model_path)

            if conf['train']['save_dir']:
                save_dir = os.path.join(conf['train']['save_dir'], conf['model']['name'], conf['train']['timestamp'])
                os.makedirs(save_dir, exist_ok=True)
                save_name = f"best_model.pth"
                best_model_path = os.path.join(save_dir, save_name)
                torch.save(model.state_dict(), os.path.join(save_dir, "best_model_state_dict.pth"))
                torch.save(model, os.path.join(save_dir, "best_model.pth"))
        else:
            counter += 1
            if counter >= conf['train']['patience']:
                logging.info(f"Early stopping at epoch {epoch} | Best Dice: {best_dice:.4f} at epoch {best_epoch}")
                return best_model_path, best_dice, best_val_loss, best_epoch

        # --- Logging ---
        if epoch % conf['train']['log_interval'] == 0 or epoch == conf['train']['num_epochs']:
            current_lr = optimizer.param_groups[0]['lr']
            logging.info(
                f"Epoch {epoch:03d} | "
                f"Train Loss: {avg_train_loss:.4f} | "
                f"Val Loss: {avg_val_loss:.4f} | "
                f"Dice: {avg_dice:.4f} | "
                f"IoU: {avg_iou:.4f} | "
                f"LR: {current_lr:.6f}"
            )

    logging.info("Training completed.")
    logging.info(f"Best Dice achieved: {best_dice}, Epoch: {best_epoch}")
    if best_model_path:
        logging.info(f"Best model path: {best_model_path}")

    return best_model_path, best_dice, best_val_loss, best_epoch

def reconstruct_from_patches(patches, coords, image_size, patch_size):
    H, W = map(int, image_size)
    full_prob = torch.zeros((H, W), device=patches.device)
    count = torch.zeros((H, W), device=patches.device)

    for patch, (top, left) in zip(patches, coords):
        full_prob[top:top+patch_size[0], left:left+patch_size[1]] += patch.squeeze(0)
        count[top:top+patch_size[0], left:left+patch_size[1]] += 1

    full_prob /= count.clamp(min=1e-7)
    return full_prob.clamp(0, 1)


def test_model(model, test_loader, conf, save_dir=None, return_metrics_only=False):
    model.eval()
    dice_scores = []
    iou_scores = []
    val_loss = 0.0
    batch_size = conf['test']['batch_size']
    patch_size = conf['dataset']['patch_size']

#    if not return_metrics_only:
#        viz_dir = os.path.join(
#            conf['test']['save_dir'],
#            conf['model']['name'],
#            f"{conf['train']['experiment_name']}_{conf['train']['timestamp']}"
#        )
#        os.makedirs(viz_dir, exist_ok=True)

    with torch.no_grad():
        for idx, batch in enumerate(test_loader):
            image_patches = batch['image_patches'].cuda()    # [N, 1, H, W]
            mask_patches = batch['mask_patches'].cuda()
            distance_patches = batch['dist_patches'].cuda()

            coords = batch['coords']
            image_size = batch['image_size']
            image_idx = batch['image_idx']
            H_orig, W_orig = batch['original_size']

            B, N, C, H, W = image_patches.shape
            image_patches = image_patches.view(B * N, C, H, W)
            mask_patches = mask_patches.view(B * N, 1, H, W)
            distance_patches = distance_patches.view(B * N, 1, H, W)

            preds = []
            for i in range(0, image_patches.size(0), batch_size):
                patch_batch = image_patches[i:i + batch_size]
                preds.append(model(patch_batch))
            preds = torch.cat(preds, dim=0)

            loss = criterion(preds, mask_patches, distance_patches)
            val_loss += loss.item()

            # Reconstruct full masks
            pred_full = reconstruct_from_patches(preds, coords, image_size, patch_size)
            mask_full = reconstruct_from_patches(mask_patches, coords, image_size, patch_size)

            pred_full = pred_full[:H_orig, :W_orig].unsqueeze(0).unsqueeze(0)
            mask_full = mask_full[:H_orig, :W_orig].unsqueeze(0).unsqueeze(0)

            dice = compute_dice(pred_full.unsqueeze(0), mask_full.unsqueeze(0))
            iou = compute_iou(pred_full.unsqueeze(0), mask_full.unsqueeze(0))

            dice_scores.append(dice)
            iou_scores.append(iou)

            # Visualize
            if not return_metrics_only:
                fig, axes = plt.subplots(1, 3, figsize=(15, 5))            
                img_path = os.path.join(test_loader.dataset.root_dir, "enhanced_images", batch['filename'])
                img = np.array(Image.open(img_path).convert("L"))

                axes[0].imshow(img, cmap='gray')
                axes[0].set_title('Original Image')
                axes[0].axis('off')

                axes[1].imshow(mask_full.squeeze().cpu().numpy(), cmap='gray')
                axes[1].set_title('Ground Truth')
                axes[1].axis('off')

                pred_binary = pred_full.squeeze().cpu().numpy().astype(np.float32)
                axes[2].imshow(pred_binary, cmap='gray')
                axes[2].set_title(f'Prediction\nDice: {dice:.4f}')
                axes[2].axis('off')

                plt.tight_layout()
                plt.savefig(os.path.join(save_dir, f'test_image_{idx}.png'))
                plt.close()

    avg_val_loss = val_loss / len(test_loader)
    avg_dice = np.mean(dice_scores)
    avg_iou = np.mean(iou_scores)

    return avg_val_loss, avg_dice, avg_iou


def main(conf):
    model = build_model(conf)
    train_loader, test_loader = get_patch_dataloaders(conf)
    best_model_path, best_dice, best_val_loss, best_epoch = train_model(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        conf=conf,
    )
    save_metadata(os.path.dirname(best_model_path), model, conf, best_dice, best_val_loss, best_epoch)
    model = torch.load(best_model_path, weights_only=False)
    save_dir = os.path.dirname(best_model_path)
    avg_val_loss, avg_dice, avg_iou = test_model(model, test_loader, conf, save_dir)
    logging.info(f"Test Loss: {avg_val_loss:.4f} | Dice: {avg_dice:.4f} | IoU: {avg_iou:.4f}")


if __name__ == '__main__':
    args = parse_args()
    args_dict = vars(args)

    with open(args.config, 'r') as f:
        conf = yaml.safe_load(f)
    conf['train']['experiment_name'] = args_dict.get('experiment_name', 'unnamed_experiment')
    conf['dataset']['dist_transform'] = True

    # Create logs directory
    log_dir = os.path.join('logs', f"{conf['model']['name']}")
    os.makedirs(log_dir, exist_ok=True)

    # Add timestamp to log filename
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    log_filename = os.path.join(log_dir, f"{conf['train']['experiment_name']}_{timestamp}.log")
    conf['train']['timestamp'] = timestamp
    # Set up logging to file and console
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[
            logging.FileHandler(log_filename),
            logging.StreamHandler()
        ]
    )
    logging.info(f"Starting training with configuration: {conf}")
    main(conf)