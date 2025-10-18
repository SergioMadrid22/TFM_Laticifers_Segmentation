import argparse
import yaml
import os
import torch
import numpy as np
import random
import logging
import datetime
import pandas as pd
from PIL import Image
from glob import glob
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import KFold
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.amp import autocast, GradScaler
import matplotlib.pyplot as plt

# --- Import your existing modules ---
from models import build_model
from losses import get_loss_function # This will now fetch our 'masked_mse' loss

def set_seed(seed=42):
    np.random.seed(seed); random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed); torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def parse_args():
    parser = argparse.ArgumentParser(description="Pre-train the Unet_MIM autoencoder for Deep Closing.")
    parser.add_argument('-c', '--config', required=True, help='Path to the autoencoder training config YAML.')
    parser.add_argument('-e', '--experiment_name', default='deepclosing_ae', help='Experiment name for this pre-training run.')
    return parser.parse_args()

# =========================================================================
#  ------ NEW, SIMPLIFIED DATASET FOR THIS TASK ------
# =========================================================================
# This dataset loads ONLY the ground truth masks, as they serve as both
# the input and the target for the self-supervised autoencoder.

class AutoencoderMaskDataset(Dataset):
    def __init__(self, mask_dir, filenames, patch_size, patches_per_image):
        self.mask_dir = mask_dir
        self.filenames = filenames
        self.patch_size = patch_size
        self.patches_per_image = patches_per_image
        self.samples = []
        for i in range(len(self.filenames)):
            self.samples.extend([i] * self.patches_per_image)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        file_idx = self.samples[idx]
        fname = self.filenames[file_idx]
        
        mask_path = os.path.join(self.mask_dir, fname)
        mask_img = np.array(Image.open(mask_path).convert("L"))
        mask_bin = (mask_img > 127).astype(np.float32) # Normalize to 0.0 or 1.0
        
        # Randomly crop a patch
        H, W = mask_bin.shape
        ph, pw = self.patch_size
        top = np.random.randint(0, H - ph + 1)
        left = np.random.randint(0, W - pw + 1)
        
        mask_patch = mask_bin[top:top+ph, left:left+pw]
        
        # Return as a tensor with a channel dimension
        return torch.from_numpy(mask_patch).unsqueeze(0)

# =========================================================================
#  ------ NEW, SIMPLIFIED VALIDATION AND TRAINING LOOPS ------
# =========================================================================

def validate_autoencoder(model, val_loader, loss_fn, save_dir, epoch):
    """A simplified validation loop for the autoencoder."""
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for batch in val_loader:
            masks = batch.cuda()
            with autocast(device_type='cuda'):
                # The model handles its own masking and loss calculation
                output_dict = model(masks)
                loss = output_dict['loss']
            val_loss += loss.item()
    
    # --- Save a visual example to monitor training progress ---
    if epoch % 5 == 0: # Save visuals every 5 epochs
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axes[0].imshow(output_dict['original_image'][0, 0].cpu().numpy(), cmap='gray')
        axes[0].set_title('Original Mask')
        axes[1].imshow(output_dict['masked_image'][0, 0].cpu().numpy(), cmap='gray')
        axes[1].set_title('Masked Input')
        axes[2].imshow(output_dict['prediction'][0, 0].cpu().numpy(), cmap='gray')
        axes[2].set_title('Reconstruction')
        for ax in axes: ax.axis('off')
        plt.tight_layout()
        
        vis_dir = os.path.join(save_dir, "ae_val_visuals")
        os.makedirs(vis_dir, exist_ok=True)
        plt.savefig(os.path.join(vis_dir, f"epoch_{epoch:03d}.png"))
        plt.close()

    return val_loss / len(val_loader)

def train_autoencoder(model, train_loader, val_loader, save_dir, conf):
    """A simplified training loop for the Unet_MIM autoencoder."""
    model = model.cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=conf['train']['learning_rate'])
    scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=10, factor=0.2, min_lr=1e-6)
    scaler = GradScaler() 

    best_val_loss = float('inf')
    best_epoch, counter = 0, 0
    
    os.makedirs(save_dir, exist_ok=True)

    for epoch in range(1, conf['train']['num_epochs'] + 1):
        model.train()
        train_loss = 0.0
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch}/{conf['train']['num_epochs']}"):
            masks = batch.cuda()
            
            optimizer.zero_grad()
            with autocast(device_type='cuda'):
                output_dict = model(masks)
                loss = output_dict['loss']

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = validate_autoencoder(model, val_loader, None, save_dir, epoch)
        scheduler.step(avg_val_loss)

        logging.info(f"Epoch {epoch:03d} | Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_epoch = epoch
            counter = 0
            # Save the entire model (not just state_dict) as it will be loaded by DeepClosingRefiner
            torch.save(model, os.path.join(save_dir, "best_autoencoder.pth"))
            logging.info(f"*** New best autoencoder saved with val loss: {best_val_loss:.6f} ***")
        else:
            counter += 1
        
        if counter >= conf['train']['patience']:
            logging.info(f"Early stopping at epoch {epoch}. Best loss was {best_val_loss:.6f} at epoch {best_epoch}.")
            break
            
    return os.path.join(save_dir, "best_autoencoder.pth")

def main_cv(conf):
    """Main cross-validation loop for pre-training autoencoders."""
    mask_dir = conf['dataset']['feature_dirs']['mask']
    all_filenames = sorted([f for f in os.listdir(mask_dir) if f.endswith(".tif")])
    seed = conf['train'].get('seed', 42)
    kf = KFold(n_splits=5, shuffle=True, random_state=seed)
    save_dir = conf['train']['save_dir']

    for fold, (train_idx, val_idx) in enumerate(kf.split(all_filenames)):
        logging.info(f"===== Pre-training Autoencoder for Fold {fold+1} / 5 =====")
        train_files = [all_filenames[i] for i in train_idx]
        val_files   = [all_filenames[i] for i in val_idx]
        fold_save_dir = os.path.join(save_dir, f"fold_{fold+1}")

        # --- Create new, simple datasets for this task ---
        train_dataset = AutoencoderMaskDataset(mask_dir, train_files, tuple(conf['dataset']['patch_size']), conf['dataset']['num_patches'])
        val_dataset = AutoencoderMaskDataset(mask_dir, val_files, tuple(conf['dataset']['patch_size']), conf['dataset']['num_patches'])
        
        train_loader = DataLoader(train_dataset, batch_size=conf['train']['batch_size'], shuffle=True, num_workers=conf['dataset']['num_workers'])
        val_loader = DataLoader(val_dataset, batch_size=conf['train']['batch_size'], shuffle=False, num_workers=conf['dataset']['num_workers'])
        
        # --- Build and Train the Autoencoder ---
        model = build_model(conf)
        train_autoencoder(model, train_loader, val_loader, fold_save_dir, conf)

    logging.info("===== Autoencoder pre-training complete for all folds. =====")

if __name__ == '__main__':
    args = parse_args()
    conf = yaml.safe_load(open(args.config))
    set_seed(conf['train'].get('seed', 42))

    conf['train']['experiment_name'] = args.experiment_name
    conf['train']['timestamp'] = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    save_dir = os.path.join(
        conf['train']['save_dir'], 
        'deepclosing_ae', # Hardcode model name for clarity
        f"{conf['train']['timestamp']}_{conf['train']['experiment_name']}"
    )
    conf['train']['save_dir'] = save_dir
    os.makedirs(save_dir, exist_ok=True)
    log_filename = os.path.join(save_dir, "pretrain_log.log")
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s %(message)s',
        handlers=[logging.FileHandler(log_filename), logging.StreamHandler()]
    )
    logging.info(f"Starting Deep Closing Autoencoder pre-training with configuration: {conf}")
    main_cv(conf)