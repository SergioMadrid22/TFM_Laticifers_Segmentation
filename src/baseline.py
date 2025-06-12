import argparse
import yaml
from models import build_model
from dataset import get_dataloaders
import torch
import numpy as np
from tqdm import tqdm
import torch.nn as nn
import os
from metrics import compute_dice, compute_iou, focal_tversky_loss
from torch.optim.lr_scheduler import ReduceLROnPlateau
import logging
import matplotlib.pyplot as plt
import datetime
import random

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
    parser.add_argument('--experiment_name', '-e', type=str, default='unnamed_experiment', 
                        help='Name of the experiment for logging purposes')

    return parser.parse_args()

# Focal Tversky Loss with recommended default params
def criterion(pred, target):
    return focal_tversky_loss(pred, target, alpha=0.7, beta=0.3, gamma=0.75)


def train_model(model, train_loader, test_loader, conf):
    model = model.cuda()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=conf['train']['learning_rate'],
        #weight_decay=1e-5
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

    for epoch in tqdm(range(1, conf['train']['num_epochs'] + 1), desc="Training Epochs", leave=False, unit='epoch'):
        model.train()
        train_loss = 0.0
        optimizer.zero_grad()

        for batch_idx, (images, masks) in enumerate(train_loader):
            images, masks = images.cuda(), masks.cuda()
            preds = model(images)
            loss = criterion(preds, masks) / accumulation_steps
            loss.backward()

            if (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1) == len(train_loader):
                optimizer.step()
                optimizer.zero_grad()

            train_loss += loss.item() * accumulation_steps

        avg_train_loss = train_loss / len(train_loader)

        model.eval()
        val_loss = 0.0
        dice_scores = []
        iou_scores = []

        with torch.no_grad():
            for images, masks in test_loader:
                images, masks = images.cuda(), masks.cuda()
                preds = model(images)
                loss = criterion(preds, masks)
                val_loss += loss.item()
                dice_scores.append(compute_dice(preds, masks))
                iou_scores.append(compute_iou(preds, masks))

        avg_val_loss = val_loss / len(test_loader)
        avg_dice = np.mean(dice_scores)
        scheduler.step(avg_val_loss)

        if avg_dice > best_dice:
            best_dice = avg_dice
            best_epoch = epoch

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            counter = 0

            if best_model_path and os.path.exists(best_model_path):
                os.remove(best_model_path)

            if conf['train']['save_dir']:
                save_dir = os.path.join(conf['train']['save_dir'], conf['model']['name'])
                os.makedirs(save_dir, exist_ok=True)
                save_name = f"{conf['train']['experiment_name']}_{epoch}_{best_dice:.4f}_{conf['train']['timestamp']}.pth"
                best_model_path = os.path.join(save_dir, save_name)
                torch.save(model.state_dict(), best_model_path)
        else:
            counter += 1
            if counter >= conf['train']['patience']:
                logging.info(f"Early stopping at epoch {epoch} | Best Dice: {best_dice:.4f} at epoch {best_epoch}")
                return best_model_path, best_dice, best_val_loss

        if epoch % conf['train']['log_interval'] == 0 or epoch == conf['train']['num_epochs']:
            current_lr = optimizer.param_groups[0]['lr']
            logging.info(
                f"Epoch {epoch:03d} | "
                f"Train Loss: {avg_train_loss:.4f} | "
                f"Val Loss: {avg_val_loss:.4f} | "
                f"Dice: {avg_dice:.4f} | "
                f"IoU: {np.mean(iou_scores):.4f} | "
                f"LR: {current_lr:.6f}"
            )

    logging.info("Training completed.")
    logging.info(f"Best Dice achieved: {best_dice}, Epoch: {best_epoch}")
    if best_model_path:
        logging.info(f"Best model path: {best_model_path}")

    return best_model_path, best_dice, best_val_loss


def test_model(model, test_loader, conf):
    model.eval()
    dice_scores = []
    iou_scores = []
    val_loss = 0.0

    # Output directory for visualizations
    viz_dir = os.path.join(
        conf['test']['save_dir'],
        conf['model']['name'],
        f"{conf['train']['experiment_name']}_{conf['train']['timestamp']}"
    )
    os.makedirs(viz_dir, exist_ok=True)

    with torch.no_grad():
        for idx, (images, masks) in enumerate(test_loader):
            images, masks = images.cuda(), masks.cuda()
            preds = model(images)
            loss = criterion(preds, masks)
            val_loss += loss.item()

            dice = compute_dice(preds, masks)
            iou = compute_iou(preds, masks)

            dice_scores.append(dice)
            iou_scores.append(iou)

            # Visualize each sample in the batch
            for b in range(images.size(0)):
                fig, axes = plt.subplots(1, 3, figsize=(15, 5))

                img = images[b].cpu().squeeze().numpy()
                gt_mask = masks[b, 0].cpu().numpy()
                pred_mask = preds[b, 0].cpu().numpy()

                axes[0].imshow(img, cmap='gray')
                axes[0].set_title('Original Image')
                axes[0].axis('off')

                axes[1].imshow(gt_mask, cmap='gray')
                axes[1].set_title('Ground Truth')
                axes[1].axis('off')

                axes[2].imshow(pred_mask, cmap='gray')
                axes[2].set_title(f'Prediction\nDice: {dice:.4f}')
                axes[2].axis('off')

                plt.tight_layout()
                plt.savefig(os.path.join(viz_dir, f'test_sample_{idx}_{b}.png'))
                plt.close()

    avg_val_loss = val_loss / len(test_loader)
    avg_dice = np.mean(dice_scores)
    avg_iou = np.mean(iou_scores)

    logging.info(f"Test Loss: {avg_val_loss:.4f} | Dice: {avg_dice:.4f} | IoU: {avg_iou:.4f}")
    return avg_val_loss, avg_dice, avg_iou

def main(conf):
    model = build_model(conf)
    train_loader, test_loader = get_dataloaders(
        batch_size=conf['train']['batch_size'],
        dataset_root=conf['dataset']['root'],
        image_size=conf['dataset']['image_size'],
        dataset_csv=conf['dataset']['dataset_csv'],
        num_workers=4
    )
    best_model_path, best_dice, best_val_loss = train_model(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        conf=conf,
    )
    model.load_state_dict(torch.load(best_model_path))
    avg_val_loss, avg_dice, avg_iou = test_model(model, test_loader, conf)

import datetime

if __name__ == '__main__':
    args = parse_args()
    args_dict = vars(args)

    with open(args.config, 'r') as f:
        conf = yaml.safe_load(f)
    conf['train']['experiment_name'] = args_dict.get('experiment_name', 'unnamed_experiment')
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