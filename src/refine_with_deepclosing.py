import argparse
import yaml
import os
import torch
import numpy as np
import random
import logging
import pandas as pd
from PIL import Image
from glob import glob
from tqdm import tqdm
from sklearn.model_selection import KFold

# --- Import your existing and new modules ---
from datasets import get_patch_dataloaders
from metrics import compute_metrics
from clDice.cldice_metric.cldice import clDice as compute_cldice
from utils import reconstruct_from_patches
from models import DeepClosingRefiner # Import our new refiner model

def set_seed(seed=42):
    np.random.seed(seed); random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed); torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def parse_args():
    p = argparse.ArgumentParser(description="Refine segmentation masks using the Deep Closing pipeline.")
    p.add_argument('-c', '--config', required=True, help='Path to the original training config YAML.')
    p.add_argument('-d', '--directory', required=True, help='Path to the root experiment directory of the CHAMPION SEGMENTATION model.')
    p.add_argument('--ae_dir', required=True, help='Path to the root experiment directory of the PRE-TRAINED AUTOENCODERS.')
    p.add_argument('-o', '--output', default='deepclosing_results', help='Root directory to save outputs.')
    return p.parse_args()

# ... (You can copy the create_qualitative_overlay function here if you want visuals) ...

def main():
    args = parse_args()
    conf = yaml.safe_load(open(args.config)); set_seed(conf['train'].get('seed', 42))
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    all_metrics_refined = []
    all_metrics_original = []

    # --- Setup Fold Logic ---
    mask_dir = conf['dataset']['feature_dirs']['mask']
    all_filenames = sorted([f for f in os.listdir(mask_dir) if f.endswith(".tif")])
    kf = KFold(n_splits=5, shuffle=True, random_state=conf['train'].get('seed', 42))

    # --- Cross-Validation Loop ---
    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(all_filenames)):
        fold_num = fold_idx + 1
        logging.info(f"===== Refining and Evaluating Fold {fold_num}/5 =====")
        
        # --- Load the Correct Models for this Fold ---
        # 1. Load the standard, champion SEGMENTATION model
        seg_model_path = os.path.join(args.directory, f"fold_{fold_num}", "best_model_soft_dice.pth")
        if not os.path.exists(seg_model_path): continue
        seg_model = torch.load(seg_model_path, map_location=device).eval()

        # 2. Load the pre-trained AUTOENCODER for this fold
        ae_model_path = os.path.join(args.ae_dir, f"fold_{fold_num}", "best_autoencoder.pth")
        if not os.path.exists(ae_model_path): continue
        autoencoder = torch.load(ae_model_path, map_location=device)

        # 3. Build the DeepClosingRefiner
        refiner = DeepClosingRefiner(autoencoder, device=device)

        # --- Create Dataloader for this Fold's Validation Set ---
        val_files = [all_filenames[i] for i in val_idx]
        _, val_loader = get_patch_dataloaders(conf, val_filenames=val_files)

        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Refining Fold {fold_num}"):
                # --- Get the INITIAL prediction from the standard segmentation model ---
                image_patches = batch['image_patches'].to(device)
                B, N, C, H, W = image_patches.shape
                image_patches = image_patches.view(B*N, C, H, W)
                
                pred_logits = seg_model(image_patches)
                if isinstance(pred_logits, tuple): pred_logits = pred_logits[0]
                pred_probs = torch.sigmoid(pred_logits)

                initial_pred_full = reconstruct_from_patches(pred_probs, batch['coords'], batch['image_size'], conf['dataset']['patch_size'])
                H_orig, W_orig = batch['original_size']
                initial_pred_full = initial_pred_full[:H_orig, :W_orig]
                initial_pred_binary = (initial_pred_full > 0.5).float().unsqueeze(0).unsqueeze(0)

                # --- REFINE the initial prediction using the Deep Closing model ---
                refinement_dict = refiner(initial_pred_binary)
                final_closed_mask = refinement_dict['final_closed_mask']

                # --- Compare Metrics ---
                gt_mask_path = os.path.join(mask_dir, batch['filename'][0])
                gt_mask = (np.array(Image.open(gt_mask_path).convert("L")) > 0).astype(np.uint8)
                gt_tensor = torch.from_numpy(gt_mask).unsqueeze(0).unsqueeze(0).to(device)

                # Metrics for ORIGINAL prediction
                metrics_orig = compute_metrics(initial_pred_binary.float(), gt_tensor.float())
                metrics_orig['clDice'] = compute_cldice(initial_pred_binary.squeeze().cpu(), gt_tensor.squeeze().cpu()).item()
                all_metrics_original.append(metrics_orig)

                # Metrics for REFINED prediction
                metrics_refined = compute_metrics(final_closed_mask.float(), gt_tensor.float())
                metrics_refined['clDice'] = compute_cldice(final_closed_mask.squeeze().cpu(), gt_tensor.squeeze().cpu()).item()
                all_metrics_refined.append(metrics_refined)
                
    # --- Final Aggregation and Reporting ---
    df_orig = pd.DataFrame(all_metrics_original).mean()
    df_refined = pd.DataFrame(all_metrics_refined).mean()
    
    summary_df = pd.DataFrame([df_orig, df_refined], index=["Standard Model", "Deep Closing Refined"])
    
    logging.info("\n\n===== DEEP CLOSING EXPERIMENT SUMMARY =====")
    print(summary_df.to_string())
    
    os.makedirs(args.output, exist_ok=True)
    summary_df.to_csv(os.path.join(args.output, "deep_closing_summary.csv"))

if __name__ == '__main__':
    main()