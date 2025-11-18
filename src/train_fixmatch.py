import argparse
import yaml
import os
import torch
import torch.nn.functional as F
import numpy as np
import random
import logging
import datetime
import pandas as pd
from sklearn.model_selection import KFold
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.amp import autocast, GradScaler

# --- Import your existing modules ---
# Crucially, we now import the new SSL dataloader
from models import build_model
from datasets import get_ssl_dataloaders 
from losses import get_loss_function
from utils import save_metadata
from train import test_model # We can reuse your excellent test_model function for validation!

def set_seed(seed=42):
    np.random.seed(seed); random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed); torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def parse_args():
    p = argparse.ArgumentParser(description="Train a segmentation model using the FixMatch SSL framework.")
    p.add_argument('-c', '--config', required=True, help='Path to the SSL training config YAML.')
    p.add_argument('-e', '--experiment_name', default='ssl_exp', help='Experiment name.')
    return p.parse_args()

# =========================================================================
#  ------ MODIFIED TRAINING LOOP FOR FIXMATCH ------
# =========================================================================

def train_ssl_model(model, train_loader, test_loader, save_dir, conf):
    """
    A modified training loop for semi-supervised learning with FixMatch.
    This version includes a supervised warm-up phase and a gradual lambda ramp-up.
    """
    model = model.cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=conf['train']['learning_rate'])
    scheduler = ReduceLROnPlateau(optimizer, mode='max', patience=8, factor=0.5, min_lr=1e-7)
    scaler = GradScaler() 

    best_dice = 0.0
    best_epoch, counter = 0, 0
    
    supervised_loss_fn = get_loss_function(conf)
    # Use reduction='none' to apply confidence mask later
    consistency_loss_fn = torch.nn.BCEWithLogitsLoss(reduction='none') 

    lambda_final = conf.get('train', {}).get('lambda_consistency', 0.1)
    confidence_thresh = conf.get('train', {}).get('confidence_thresh', 0.5)
    warmup_epochs = conf.get('train', {}).get('warmup_epochs', 5)

    os.makedirs(save_dir, exist_ok=True)

    for epoch in range(1, conf['train']['num_epochs'] + 1):
        model.train()
        train_loss_sup, train_loss_unsup = 0.0, 0.0
        
        # --- Lambda Ramp-up Schedule ---
        if epoch <= warmup_epochs:
            current_lambda = 0.0
        else:
            # Gradually increase lambda after warm-up
            ramp_up_progress = (epoch - warmup_epochs) / (conf['train']['num_epochs'] - warmup_epochs)
            current_lambda = lambda_final * np.exp(-5. * (1. - ramp_up_progress)**2)
        
        if epoch == warmup_epochs:
            logging.info(f"--- Epoch {epoch}: Warm-up complete. Starting unsupervised training. ---")

        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            
            labeled_mask = batch['is_labeled']
            unlabeled_mask = ~labeled_mask

            # --- 1. Supervised Loss ---
            loss_sup = torch.tensor(0.0, device='cuda')
            if labeled_mask.any():
                labeled_images = batch['labeled_image'][labeled_mask].cuda()
                gt_masks = batch['mask'][labeled_mask].cuda()
                with autocast(device_type='cuda'):
                    preds_sup = model(labeled_images)
                    if isinstance(preds_sup, tuple): preds_sup = preds_sup[0]
                    loss_sup = supervised_loss_fn(preds_sup, gt_masks)
            
            # --- 2. Unsupervised Loss (ONLY if not in warm-up) ---
            loss_unsup = torch.tensor(0.0, device='cuda')
            # only compute unsupervised loss if lambda > 0
            if unlabeled_mask.any() and current_lambda > 0:
                unlabeled_weak = batch['unlabeled_weak'][unlabeled_mask].cuda()
                unlabeled_strong = batch['unlabeled_strong'][unlabeled_mask].cuda()

                with torch.no_grad():
                    pseudo_logits = model(unlabeled_weak)
                    if isinstance(pseudo_logits, tuple): pseudo_logits = pseudo_logits[0]
                    pseudo_probs = torch.sigmoid(pseudo_logits)
                    pseudo_label = (pseudo_probs > confidence_thresh).float()
                    confidence_mask = (pseudo_probs > confidence_thresh) | (pseudo_probs < 1 - confidence_thresh)

                with autocast(device_type='cuda'):
                    preds_strong = model(unlabeled_strong)
                    if isinstance(preds_strong, tuple): preds_strong = preds_strong[0]
                    loss_unsup_map = consistency_loss_fn(preds_strong, pseudo_label)
                    
                    # Only calculate mean over confident pixels
                    if confidence_mask.sum() > 0:
                        loss_unsup = (loss_unsup_map * confidence_mask).sum() / confidence_mask.sum()
                    else:
                        loss_unsup = torch.tensor(0.0, device='cuda') # No confident pixels found

            # --- 3. Combine and Backpropagate ---
            total_loss = loss_sup + current_lambda * loss_unsup

            # Skip backpropagation if the total loss for this batch is zero
            if total_loss.item() > 0:
                scaler.scale(total_loss).backward()
                scaler.step(optimizer)
                scaler.update()

            train_loss_sup += loss_sup.item()
            train_loss_unsup += loss_unsup.item()
        
        
        # --- End of Epoch: Validation ---
        avg_train_loss_sup = train_loss_sup / len(train_loader)
        avg_train_loss_unsup = train_loss_unsup / len(train_loader)
        
        avg_metrics = test_model(model, test_loader, conf, save_dir=save_dir, epoch=epoch)
        current_dice = avg_metrics.get('softDice', 0.0) # Using softDice for scheduler
        scheduler.step(current_dice)

        logging.info(
            f"Epoch {epoch:03d} | Sup Loss: {avg_train_loss_sup:.4f} | Unsup Loss: {avg_train_loss_unsup:.4f} | "
            f"Val Dice: {current_dice:.4f} | Val clDice: {avg_metrics.get('clDice', 0.0):.4f}"
        )

        if current_dice > best_dice:
            best_dice = current_dice; best_epoch = epoch; counter = 0
            torch.save(model.state_dict(), os.path.join(save_dir, "best_model.pth"))
            logging.info(f"*** New best model saved with Dice: {best_dice:.4f} ***")
        else:
            counter += 1
        
        if counter >= conf['train']['patience']:
            logging.info(f"Early stopping at epoch {epoch}. Best Dice was {best_dice:.4f} at epoch {best_epoch}.")
            break
            
    return os.path.join(save_dir, "best_model.pth"), best_dice

def main_cv(conf):
    # --- Setup file lists ---
    image_dir = conf['dataset']['feature_dirs']['image']
    mask_dir = conf['dataset']['feature_dirs']['mask']
    
    labeled_files = sorted([f for f in os.listdir(mask_dir) if f.endswith(".tif")])
    all_image_files = sorted([f for f in os.listdir(image_dir) if f.endswith(".tif")])
    unlabeled_files = sorted(list(set(all_image_files) - set(labeled_files)))
    
    logging.info(f"Found {len(labeled_files)} labeled and {len(unlabeled_files)} unlabeled images.")

    seed = conf['train'].get('seed', 42)
    kf = KFold(n_splits=5, shuffle=True, random_state=seed)
    fold_results = []
    save_dir = conf['train']['save_dir']

    for fold, (train_idx, val_idx) in enumerate(kf.split(labeled_files)):
        logging.info(f"===== SSL Training Fold {fold+1} / 5 =====")
        
        # Split ONLY the labeled files for this fold's train/val sets
        train_labeled_files = [labeled_files[i] for i in train_idx]
        val_files = [labeled_files[i] for i in val_idx]
        
        # The unlabeled files are always used for training
        train_unlabeled_files = unlabeled_files
        
        fold_save_dir = os.path.join(save_dir, f"fold_{fold+1}")
        
        train_loader, val_loader = get_ssl_dataloaders(
            conf, train_labeled_files, train_unlabeled_files, val_files
        )
        
        model = build_model(conf)
        
        best_model_path, _ = train_ssl_model(model, train_loader, val_loader, fold_save_dir, conf)
        
        # --- Final Evaluation of the Fold ---
        model.load_state_dict(torch.load(best_model_path))
        test_metrics = test_model(model, val_loader, conf, save_dir=fold_save_dir, epoch="best")
        fold_results.append(test_metrics)
        logging.info(f"Fold {fold+1} results: " + " | ".join([f"{k}: {v:.4f}" for k,v in test_metrics.items()]))
    
    # Convert fold results to a DataFrame
    results_df = pd.DataFrame(fold_results)
    
    # Calculate mean and standard deviation for all metrics
    mean_metrics = results_df.mean()
    std_metrics = results_df.std()

    # Append mean and std as new rows to the DataFrame
    results_df.loc['mean'] = mean_metrics
    results_df.loc['std'] = std_metrics

    # Log the final aggregated results
    logging.info("===== Cross-validation results =====")
    log_msg_mean = "Mean: " + " | ".join([f"{k}: {v:.4f}" for k, v in mean_metrics.items()])
    log_msg_std = "Std Dev: " + " | ".join([f"{k}: {v:.4f}" for k, v in std_metrics.items()])
    logging.info(log_msg_mean)
    logging.info(log_msg_std)

    # Save the complete DataFrame (including folds, mean, and std) to a CSV file
    results_df.to_csv(os.path.join(save_dir, "cv_results.csv"))

if __name__ == '__main__':
        # Parse arguments and load configuration
    args = parse_args()
    conf = yaml.safe_load(open(args.config)) # Load configuration from YAML file
    seed = conf['train'].get('seed', 42)
    set_seed(seed) # Set random seed for reproducibility

    # Set experiment name and timestamp for saving
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
    # Set up logging to file and console
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[logging.FileHandler(log_filename), logging.StreamHandler()]
    )
    logging.captureWarnings(True)
    logging.info(f"Starting training with configuration: {conf}")
    logging.info(f"Experiment name: {args.experiment_name}")
    main_cv(conf)