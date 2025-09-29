import argparse
import yaml
import os
import torch
import numpy as np
import random
import logging
import datetime
import pandas as pd
from sklearn.model_selection import KFold
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.amp import autocast, GradScaler

# --- Import your existing, excellent modules ---
from models import build_model
from datasets import get_patch_dataloaders
from losses import get_loss_function
from utils import save_metadata
from train_cv import test_model # We can reuse your test_model function for evaluation!

def set_seed(seed=42):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def parse_args():
    p = argparse.ArgumentParser(description="Train a segmentation model using an adversarial (GAN) framework.")
    p.add_argument('-c', '--config', required=True, help='Path to GAN training config YAML')
    p.add_argument('-e', '--experiment_name', default='gan_exp', help='Experiment name')
    return p.parse_args()

# ----------------------------------------------------------------------------------
#  ------ DISCRIMINATOR MODEL ------
# ----------------------------------------------------------------------------------
'''
class PatchGANDiscriminator(torch.nn.Module):
    """
    A simple PatchGAN discriminator.
    Takes a concatenated (image, mask) tensor and outputs a probability map
    where each value is the "realness" of a patch.
    """
    def __init__(self, in_channels=2, ndf=64):
        super().__init__()
        self.main = torch.nn.Sequential(
            # Input: (B, in_channels, 512, 512)
            torch.nn.Conv2d(in_channels, ndf, kernel_size=4, stride=2, padding=1, bias=False),
            torch.nn.LeakyReLU(0.2, inplace=True),
            # State: (B, ndf, 256, 256)
            torch.nn.Conv2d(ndf, ndf * 2, kernel_size=4, stride=2, padding=1, bias=False),
            torch.nn.BatchNorm2d(ndf * 2),
            torch.nn.LeakyReLU(0.2, inplace=True),
            # State: (B, ndf*2, 128, 128)
            torch.nn.Conv2d(ndf * 2, ndf * 4, kernel_size=4, stride=2, padding=1, bias=False),
            torch.nn.BatchNorm2d(ndf * 4),
            torch.nn.LeakyReLU(0.2, inplace=True),
            # State: (B, ndf*4, 64, 64)
            torch.nn.Conv2d(ndf * 4, ndf * 8, kernel_size=4, stride=2, padding=1, bias=False),
            torch.nn.BatchNorm2d(ndf * 8),
            torch.nn.LeakyReLU(0.2, inplace=True),
            # State: (B, ndf*8, 32, 32)
            torch.nn.Conv2d(ndf * 8, 1, kernel_size=4, stride=1, padding=1),
            # Output: (B, 1, 31, 31) -> A map of real/fake predictions
        )

    def forward(self, image, mask):
        # Concatenate image and mask along the channel dimension
        x = torch.cat([image, mask], dim=1)
        return self.main(x)
'''
class PatchGANDiscriminator(torch.nn.Module):
    def __init__(self, in_channels=2, ndf=64):
        super().__init__()
        self.main = torch.nn.Sequential(
            # Wrap each Conv2d layer with spectral_norm
            torch.nn.utils.spectral_norm(
                torch.nn.Conv2d(in_channels, ndf, kernel_size=4, stride=2, padding=1, bias=False)
            ),
            torch.nn.LeakyReLU(0.2, inplace=True),

            torch.nn.utils.spectral_norm(
                torch.nn.Conv2d(ndf, ndf * 2, kernel_size=4, stride=2, padding=1, bias=False)
            ),
            torch.nn.BatchNorm2d(ndf * 2),
            torch.nn.LeakyReLU(0.2, inplace=True),

            torch.nn.utils.spectral_norm(
                torch.nn.Conv2d(ndf * 2, ndf * 4, kernel_size=4, stride=2, padding=1, bias=False)
            ),
            torch.nn.BatchNorm2d(ndf * 4),
            torch.nn.LeakyReLU(0.2, inplace=True),
            
            torch.nn.utils.spectral_norm(
                torch.nn.Conv2d(ndf * 4, ndf * 8, kernel_size=4, stride=2, padding=1, bias=False)
            ),
            torch.nn.BatchNorm2d(ndf * 8),
            torch.nn.LeakyReLU(0.2, inplace=True),
            
            # The final layer is typically not normalized
            torch.nn.Conv2d(ndf * 8, 1, kernel_size=4, stride=1, padding=1),
        )

    def forward(self, image, mask):
        x = torch.cat([image, mask], dim=1)
        return self.main(x)
        
# ----------------------------------------------------------------------------------
#  ------ THE NEW GAN TRAINING LOOP ------
# ----------------------------------------------------------------------------------

def train_gan_model(generator, discriminator, train_loader, test_loader, save_dir, conf):
    """
    Main GAN training loop. Alternates between training the discriminator
    and the generator.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    generator.to(device)
    discriminator.to(device)

    # --- Setup Optimizers and Schedulers for BOTH networks ---
    lr_g = conf['train']['learning_rate_g']
    lr_d = conf['train']['learning_rate_d']
    optimizer_g = torch.optim.AdamW(generator.parameters(), lr=lr_g, betas=(0.5, 0.999))
    optimizer_d = torch.optim.AdamW(discriminator.parameters(), lr=lr_d, betas=(0.5, 0.999))
    # Using 'max' mode because we are monitoring Dice score, which should be maximized.
    scheduler_g = ReduceLROnPlateau(optimizer_g, mode='max', patience=5, factor=0.5, min_lr=1e-7)
    
    # --- Setup Loss Functions ---
    adversarial_loss = torch.nn.MSELoss()
    segmentation_loss_fn = get_loss_function(conf) # Your champion loss
    lambda_seg = conf['train']['lambda_seg'] # Weight for the segmentation loss

    # --- Setup Mixed Precision ---
    scaler_g = GradScaler()
    scaler_d = GradScaler()

    best_dice, best_epoch = 0.0, 0
    patience_counter = 0

    os.makedirs(save_dir, exist_ok=True)

    for epoch in range(1, conf['train']['num_epochs'] + 1):
        generator.train()
        discriminator.train()
        
        for batch in train_loader:
            real_images = batch['inputs'].to(device)
            real_masks = batch['masks'].to(device)
            
            # ------------------------------------
            #  (1) Train the Discriminator
            # ------------------------------------
            optimizer_d.zero_grad(set_to_none=True)

            with autocast(device_type='cuda'):
                # --- Train with REAL masks ---
                pred_real = discriminator(real_images, real_masks)
                real_labels = torch.ones_like(pred_real, device=device)
                loss_d_real = adversarial_loss(pred_real, real_labels)

                # --- Train with FAKE masks ---
                fake_masks_output = generator(real_images)
                if isinstance(fake_masks_output, tuple):
                    fake_masks_logits = fake_masks_output[0]
                else:
                    fake_masks_logits = fake_masks_output
                
                fake_masks = fake_masks_logits.sigmoid().detach() # detach to avoid backprop to generator
                pred_fake = discriminator(real_images, fake_masks)
                fake_labels = torch.zeros_like(pred_fake, device=device)
                loss_d_fake = adversarial_loss(pred_fake, fake_labels)
                
                loss_d = (loss_d_real + loss_d_fake) * 0.5
            
            scaler_d.scale(loss_d).backward()
            scaler_d.step(optimizer_d)
            scaler_d.update()

            # ------------------------------------
            #  (2) Train the Generator (U-Net)
            # ------------------------------------
            optimizer_g.zero_grad(set_to_none=True)

            with autocast(device_type='cuda'):
                # --- Generate fake masks ---
                fake_masks_output = generator(real_images)
                if isinstance(fake_masks_output, tuple):
                    fake_masks_logits = fake_masks_output[0]
                else:
                    fake_masks_logits = fake_masks_output
                
                fake_masks_prob = fake_masks_logits.sigmoid()
                
                # --- Adversarial Loss (Generator tries to fool Discriminator) ---
                pred_fake_for_g = discriminator(real_images, fake_masks_prob)
                loss_g_adv = adversarial_loss(pred_fake_for_g, real_labels) # Use real_labels to fool
                
                # --- Standard Segmentation Loss ---
                loss_g_seg = segmentation_loss_fn(fake_masks_logits, real_masks)

                # --- Combined Generator Loss ---
                loss_g = loss_g_adv + lambda_seg * loss_g_seg
            
            scaler_g.scale(loss_g).backward()
            scaler_g.step(optimizer_g)
            scaler_g.update()

        # --- End of Epoch: Validation ---
        avg_metrics = test_model(generator, test_loader, conf, save_dir=save_dir, epoch=epoch)
        current_dice = avg_metrics.get('softDice', 0.0)
        scheduler_g.step(current_dice)

        logging.info(f"Epoch {epoch}: Val Dice: {current_dice:.4f}, Val clDice: {avg_metrics.get('clDice', 0.0):.4f}")

        # --- Save best model ---
        if current_dice > best_dice:
            best_dice = current_dice
            best_epoch = epoch
            patience_counter = 0
            torch.save(generator, os.path.join(save_dir, "best_generator.pth"))
            torch.save(discriminator, os.path.join(save_dir, "best_discriminator.pth"))
            logging.info(f"*** New best model saved with Dice: {best_dice:.4f} at epoch {epoch} ***")
        else:
            patience_counter += 1
        
        # --- Early Stopping ---
        if patience_counter >= conf['train']['patience']:
            logging.info(f"Early stopping at epoch {epoch}. Best Dice was {best_dice:.4f} at epoch {best_epoch}.")
            break

    logging.info("GAN training completed.")
    return os.path.join(save_dir, "best_generator.pth"), best_dice


def main_cv(conf):
    """ Main cross-validation loop for GAN training. """
    mask_dir = conf['dataset']['feature_dirs']['mask']
    all_filenames = sorted([f for f in os.listdir(mask_dir) if f.endswith(".tif")])
    seed = conf['train'].get('seed', 42)
    kf = KFold(n_splits=5, shuffle=True, random_state=seed)
    fold_results = []
    save_dir = conf['train']['save_dir']

    for fold, (train_idx, val_idx) in enumerate(kf.split(all_filenames)):
        logging.info(f"===== GAN Training Fold {fold+1} / 5 =====")
        train_files = [all_filenames[i] for i in train_idx]
        val_files = [all_filenames[i] for i in val_idx]
        fold_save_dir = os.path.join(save_dir, f"fold_{fold+1}")

        train_loader, val_loader = get_patch_dataloaders(conf, train_files, val_files)

        # --- Build GAN models ---
        generator = build_model(conf) # U-Net is the generator
        feature_dirs = conf['dataset']['feature_dirs']
        in_channels = len(feature_dirs)
        if 'mask' in feature_dirs:
            in_channels -= 1
        if 'distance' in feature_dirs:
            in_channels -= 1
        discriminator = PatchGANDiscriminator(in_channels=in_channels + 1)
        
        # --- Train ---
        best_model_path, _ = train_gan_model(
            generator, discriminator, train_loader, val_loader, fold_save_dir, conf
        )
        
        # --- Final Evaluation of the Fold ---
        generator = torch.load(best_model_path, weights_only=False) # Load best generator weights
        test_metrics = test_model(generator, val_loader, conf, save_dir=fold_save_dir, epoch="best")
        fold_results.append(test_metrics)
        logging.info(f"Fold {fold+1} results: " + " | ".join([f"{k}: {v:.4f}" for k,v in test_metrics.items()]))
    
    # --- Aggregate and Save Final CV Results ---
    results_df = pd.DataFrame(fold_results)
    mean_metrics = results_df.mean()
    std_metrics = results_df.std()
    results_df.loc['mean'] = mean_metrics
    results_df.loc['std'] = std_metrics
    logging.info("===== GAN Cross-validation results =====")
    logging.info("Mean: " + " | ".join([f"{k}: {v:.4f}" for k, v in mean_metrics.items()]))
    logging.info("Std Dev: " + " | ".join([f"{k}: {v:.4f}" for k, v in std_metrics.items()]))
    results_df.to_csv(os.path.join(save_dir, "cv_results_gan.csv"))


if __name__ == '__main__':
    args = parse_args()
    conf = yaml.safe_load(open(args.config))
    set_seed(conf['train'].get('seed', 42))

    conf['train']['experiment_name'] = args.experiment_name
    conf['train']['timestamp'] = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    save_dir = os.path.join(
        conf['train']['save_dir'], 
        conf['model']['name'],
        f"{conf['train']['timestamp']}_{conf['train']['experiment_name']}_GAN"
    )
    conf['train']['save_dir'] = save_dir
    os.makedirs(save_dir, exist_ok=True)
    log_filename = os.path.join(save_dir, "train_gan_log.log")
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s %(message)s',
        handlers=[logging.FileHandler(log_filename), logging.StreamHandler()]
    )
    logging.info(f"Starting GAN training with configuration: {conf}")
    logging.info(f"Experiment name: {conf['train']['experiment_name']}")
    main_cv(conf)