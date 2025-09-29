import argparse, yaml, os, torch, numpy as np, random, logging, datetime
import matplotlib.pyplot as plt
from PIL import Image
from models import build_model
from datasets import get_patch_dataloaders
from metrics import compute_metrics
from losses import get_loss_function
from utils import save_metadata, reconstruct_from_patches
from torch.optim.lr_scheduler import ReduceLROnPlateau
from clDice.cldice_metric.cldice import clDice as compute_cldice
from sklearn.model_selection import train_test_split
from torch.amp import autocast, GradScaler
import warnings

def set_seed(seed=42):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('-c','--config', required=True, help='Path to config YAML')
    p.add_argument('-e','--experiment_name', default='exp', help='Experiment name')
    return p.parse_args()


def test_model(model, test_loader, conf, save_dir=None, return_metrics_only=False, epoch=None):
    model.eval()
    val_loss = 0.0
    patch_size = conf['dataset']['patch_size']
    batch_size = conf['test']['batch_size']
    use_topo = conf['loss'].get('use_topographic', False)
    loss_fn = get_loss_function(conf)
    all_metrics = {}

    with torch.no_grad():
        for idx, batch in enumerate(test_loader):
            image_patches = batch['image_patches'].cuda()  # (N, C, H, W)
            mask_patches = batch['mask_patches'].cuda()    # (N, 1, H, W)
            coords = batch['coords']
            image_size = batch['image_size']
            H_orig, W_orig = batch['original_size']
            B, N, C, H, W = image_patches.shape
            image_patches = image_patches.view(B * N, C, H, W)
            mask_patches = mask_patches.view(B * N, 1, H, W)

            preds = []
            for i in range(0, image_patches.size(0), batch_size):
                patch_batch = image_patches[i:i + batch_size]
                with autocast(device_type='cuda'):
                    batch_preds = model(patch_batch)
                    if isinstance(batch_preds, tuple):
                        batch_preds = batch_preds[0] 
                preds.append(batch_preds)
            preds = torch.cat(preds, dim=0)
            if use_topo:
                distance_patches = batch['dist_patches'].cuda()
                distance_patches = distance_patches.view(B * N, 1, H, W)
                loss = loss_fn(preds, mask_patches, distance_patches)
            else:
                loss = loss_fn(preds, mask_patches)

            val_loss += loss.item()

            pred_full_prob = reconstruct_from_patches(preds, coords, image_size, patch_size)
            mask_full = reconstruct_from_patches(mask_patches, coords, image_size, patch_size)

            pred_full_prob = pred_full_prob[:H_orig, :W_orig]
            mask_full = mask_full[:H_orig, :W_orig]

            # Binarize saving the  prediction
            pred_binary_np = (pred_full_prob.cpu().numpy() > 0.5).astype(np.uint8)

            # --- METRICS COMPUTATION (on binarized mask) ---
            pred_tensor = pred_full_prob.unsqueeze(0).unsqueeze(0)
            mask_tensor = mask_full.unsqueeze(0).unsqueeze(0)

            metrics = compute_metrics(pred_tensor, mask_tensor)
            for key in metrics:
                if key not in all_metrics:
                    all_metrics[key] = []
                all_metrics[key].append(metrics[key])

            # Modify idx condition to custom the nurmber of saved images
            if not return_metrics_only and epoch is not None and idx < 5:
                if isinstance(epoch, str) and epoch == "best":
                    epoch_save_dir = os.path.join(save_dir, "val_outputs", "best_model")
                    os.makedirs(epoch_save_dir, exist_ok=True)
                elif isinstance(epoch, int):
                    epoch_save_dir = os.path.join(save_dir, "val_outputs", f"epoch_{int(epoch):03d}")
                    os.makedirs(epoch_save_dir, exist_ok=True)
                else:
                    continue
                
                base_filename = os.path.splitext(batch['filename'][0])[0]

                # Save the binarized prediction mask
                pred_save_path = os.path.join(epoch_save_dir, f"{base_filename}_pred.png")
                pred_img = Image.fromarray(pred_binary_np * 255)
                pred_img.save(pred_save_path)

                # --- PLOT THE PROBABILITY MAP IN THE FIGURE ---
                fig, axes = plt.subplots(1, 3, figsize=(15, 5))
                img_path = os.path.join(test_loader.dataset.feature_dirs['image'], batch['filename'][0])
                img = np.array(Image.open(img_path).convert("L"))

                axes[0].imshow(img, cmap='gray')
                axes[0].set_title('Original Image')
                axes[1].imshow(mask_full.squeeze().cpu().numpy(), cmap='gray')
                axes[1].set_title('Ground Truth')
                axes[2].imshow(pred_full_prob.cpu().numpy(), cmap='gray')
                axes[2].set_title("Prediction\n" + "\n".join([f"{k}: {metrics[k]:.3f}" for k in sorted(metrics)]))
                
                for ax in axes: 
                    ax.axis('off')
                
                # Metrics displayed are still from the binarized version, which is correct
                metrics_text = "\n".join([f"{k}: {metrics[k]:.3f}" for k in sorted(metrics)])
                fig.text(0.92, 0.5, metrics_text, va='center', ha='left', fontsize=9,
                        bbox=dict(facecolor='white', alpha=0.7, edgecolor='black'))
                
                plt.tight_layout(rect=[0, 0, 0.9, 1])

                save_path = os.path.join(epoch_save_dir, f'test_image_{idx}.png')
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                plt.savefig(save_path)
                plt.close()

            torch.cuda.empty_cache()

    avg_metrics = {k: np.nanmean(v) for k, v in all_metrics.items()}
    avg_metrics['val_loss'] = val_loss / len(test_loader)
    return avg_metrics


def train_model(model, train_loader, test_loader, conf):
    model = model.cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=conf['train']['learning_rate'])
    scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5, min_lr=1e-6)
    scaler = GradScaler() 

    best_dice, best_cldice, best_val_loss = 0.0, 0.0, float('inf')
    best_soft_dice, best_soft_clDice = 0.0, float('inf')
    best_epoch, counter = 0, 0
    best_model_path = None
    accumulation_steps = conf['train'].get('accumulation_steps', 1)
    use_topo = conf['loss'].get('use_topographic', False)
    loss_fn = get_loss_function(conf)
    save_dir = conf['train']['save_dir']

    curriculum_schedule = conf['train'].get('curriculum_schedule', [])
    current_level = -1

    for epoch in range(1, conf['train']['num_epochs'] + 1):
        # Apply Curriculum Learning Level if specified
        new_level = 0
        for ep_threshold, level in curriculum_schedule:
            if epoch >= ep_threshold:
                new_level = level

        if new_level != current_level:
            current_level = new_level
            logging.info(f"--- Epoch {epoch}: Setting curriculum level to {current_level} ---")
            train_loader.dataset.curriculum_level = current_level

        model.train()
        train_loss = 0.0
        optimizer.zero_grad()

        for batch_idx, batch in enumerate(train_loader):
            images = batch['inputs'].cuda()
            masks = batch['masks'].cuda()

            with autocast(device_type='cuda'):
                preds = model(images)
                if isinstance(preds, tuple):
                    preds = preds[0]
                if use_topo:
                    distances = batch['dist_maps'].cuda()
                    loss = loss_fn(preds, masks, distances) / accumulation_steps
                else:
                    loss = loss_fn(preds, masks) / accumulation_steps

            scaler.scale(loss).backward()

            # Gradient accumulation step
            if (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1) == len(train_loader):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            train_loss += loss.item() * accumulation_steps

        avg_train_loss = train_loss / len(train_loader)
        avg_metrics = test_model(
            model,
            test_loader,
            conf,
            save_dir=save_dir,
            return_metrics_only=False,
            epoch=epoch
        )
        avg_val_loss = avg_metrics.get('val_loss', float('inf'))
        scheduler.step(avg_val_loss)

        # Update best metrics and save best models
        if avg_metrics['Dice'] > best_dice:
            best_dice = avg_metrics['Dice']
        if avg_metrics['clDice'] > best_cldice:
            best_cldice = avg_metrics['clDice']
        if avg_metrics['softclDice'] < best_soft_clDice:
            best_soft_clDice = avg_metrics['softclDice']
            torch.save(model, os.path.join(save_dir, "best_model_soft_clDice.pth"))
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model, os.path.join(save_dir, "best_model_loss.pth"))
        if avg_metrics['softDice'] > best_soft_dice:
            best_soft_dice = avg_metrics['softDice']
            best_epoch = epoch
            counter = 0
            best_model_path = os.path.join(save_dir, "best_model_soft_dice.pth")
            torch.save(model, best_model_path)
        else:
            counter += 1
            # Early stopping if no improvement
            if counter >= conf['train']['patience']:
                torch.save(model, os.path.join(save_dir, "final_model.pth"))
                logging.info(f"Early stopping at epoch {epoch} | Best Dice: {best_dice:.4f} at epoch {best_epoch}")
                return best_model_path, best_dice, best_cldice, best_val_loss, best_epoch

        # Log progress
        if epoch % conf['train']['log_interval'] == 0 or epoch == conf['train']['num_epochs']:
            logging.info(
                f"Epoch {epoch:03d} | Train Loss: {avg_train_loss:.4f} | "
                f"Val Loss: {avg_val_loss:.4f} | " +
                " | ".join([f"{k}: {v:.4f}" for k, v in avg_metrics.items()]) +
                f" | LR: {optimizer.param_groups[0]['lr']:.6f}"
            )

    logging.info("Training completed.")
    torch.save(model, os.path.join(save_dir, "final_model.pth"))
    return best_model_path, best_dice, best_cldice, best_val_loss, best_epoch


def main(conf):
    model = build_model(conf)
    
    # Get available filenames from mask folder
    feature_dirs = conf['dataset']['feature_dirs']
    mask_dir = feature_dirs['mask']
    all_filenames = [f for f in os.listdir(mask_dir) if f.endswith(".tif")]
    all_filenames.sort()

    train_filenames, val_filenames = train_test_split(
        all_filenames, test_size=0.1, random_state=42
    )

    train_loader, val_loader = get_patch_dataloaders(conf, train_filenames, val_filenames)
    best_model_path, best_dice, best_cldice, best_val_loss, best_epoch = train_model(
        model, train_loader, val_loader, conf
    )
    
    model = torch.load(best_model_path, weights_only=False)
    
    save_dir = os.path.dirname(best_model_path)
    # Save metadata for this fold
    save_metadata(
        save_dir,
        model,
        conf,
        best_dice,
        best_cldice,
        best_val_loss,
        best_epoch
    )
    # Final test on this fold (same val set) with saving best model predictions
    test_metrics = test_model(
        model,
        val_loader,
        conf,
        save_dir=save_dir,
        return_metrics_only=False,
        epoch="best"
    )
    avg_val_loss = test_metrics.pop('val_loss', float('inf'))
    logging.info(f"Test Loss: {avg_val_loss:.4f} | " + 
        " | ".join([f"{k}: {v:.4f}" for k, v in test_metrics.items()]))

if __name__ == '__main__':
    args = parse_args()
    conf = yaml.safe_load(open(args.config))
    seed = conf['train'].get('seed', 42)
    set_seed(seed) # Set random seed for reproducibility

    conf['train']['experiment_name'] = args.experiment_name
    conf['train']['timestamp'] = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    save_dir = os.path.join(
        conf['train']['save_dir'], 
        conf['model']['name'],
        f"{conf['train']['timestamp']}_{conf['train']['experiment_name']}"
    )
    conf['train']['save_dir'] = save_dir
    os.makedirs(save_dir, exist_ok=True)
    log_filename = os.path.join(save_dir, f"train_log.log")
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[logging.FileHandler(log_filename), logging.StreamHandler()]
    )
    logging.captureWarnings(True)
    logging.info(f"Starting training with configuration: {conf}")
    logging.info(f"Experiment name: {args.experiment_name}")
    main(conf)
