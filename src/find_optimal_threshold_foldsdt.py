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
import cv2

from datasets import get_patch_dataloaders 
from metrics import compute_metrics
from clDice.cldice_metric.cldice import clDice as compute_cldice
from utils import reconstruct_from_patches

def set_seed(seed=42):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def parse_args():
    parser = argparse.ArgumentParser(description="Find the optimal binarization threshold for a trained model using cross-validation.")
    parser.add_argument('-c', '--config', required=True, help='Path to the original training config YAML file.')
    parser.add_argument('-d', '--directory', required=True, help='Path to the root experiment directory containing the fold checkpoints.')
    parser.add_argument('-o', '--output', default='threshold_experiment', help='Root directory to save the results and visuals.')
    return parser.parse_args()

def create_qualitative_overlay(image, gt_mask, pred_mask):
    image_color = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    gt = (gt_mask > 0).astype(np.uint8); pred = (pred_mask > 0).astype(np.uint8)
    tp = (gt & pred); fp = ((1 - gt) & pred); fn = (gt & (1 - pred))
    overlay = np.zeros_like(image_color); overlay[tp == 1] = [0,255,0]; overlay[fp == 1] = [0,0,255]; overlay[fn == 1] = [0,255,255]
    return cv2.cvtColor(cv2.addWeighted(image_color, 0.6, overlay, 0.4, 0), cv2.COLOR_BGR2RGB)

# ------------------ MAIN FUNCTION -------------------
def main():
    args = parse_args()
    conf = yaml.safe_load(open(args.config))
    set_seed(conf['train'].get('seed', 42))

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    thresholds_to_test = np.arange(0.1, 0.76, 0.05).tolist()
    logging.info(f"Will test the following thresholds: {[f'{t:.2f}' for t in thresholds_to_test]}")

    # --- MODIFIED: Store all individual results with fold information ---
    all_results = []
    
    os.makedirs(args.output, exist_ok=True)
    visuals_dir = os.path.join(args.output, "visuals")
    os.makedirs(visuals_dir, exist_ok=True)

    mask_dir = conf['dataset']['feature_dirs']['mask']
    all_filenames = sorted([f for f in os.listdir(mask_dir) if f.endswith(".tif")])
    kf = KFold(n_splits=5, shuffle=True, random_state=conf['train'].get('seed', 42))

    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(all_filenames)):
        fold_num = fold_idx + 1
        logging.info(f"--- Processing Fold {fold_num}/5 ---")
        
        model_path = os.path.join(args.directory, f"fold_{fold_num}", "best_model_soft_dice.pth")
        if not os.path.exists(model_path):
            logging.warning(f"Model not found for fold {fold_num} at {model_path}. Skipping.")
            continue
        model = torch.load(model_path, map_location=device, weights_only=False).eval()

        val_files = [all_filenames[i] for i in val_idx]
        _, val_loader = get_patch_dataloaders(conf, [], val_files)

        for batch in tqdm(val_loader, desc=f"Fold {fold_num} Inference"):
            with torch.no_grad():
                # ... (Inference and reconstruction logic is the same as before) ...
                
                # --- MODIFIED: Store fold_num with each result ---
                for threshold in thresholds_to_test:
                    # ... (Binarization logic is the same) ...
                    
                    # ... (Metric calculation is the same) ...
                    metrics = compute_metrics(...) # your metric call
                    
                    metrics['threshold'] = threshold
                    metrics['image_file'] = batch['filename'][0]
                    metrics['fold'] = fold_num # <-- Store the fold number
                    all_results.append(metrics)
                    
                    # ... (Visual saving logic is the same) ...

    # --- Final Aggregation and Reporting (CORRECTED) ---
    if not all_results:
        logging.error("No results were generated. Exiting.")
        return
    
    # --- STEP 1: Create a DataFrame with all individual image results ---
    full_results_df = pd.DataFrame(all_results)
    
    # --- STEP 2: Calculate the AVERAGE metric for each fold and each threshold ---
    # This creates a table where each row is a fold and columns are metrics for a given threshold
    fold_avg_df = full_results_df.groupby(['threshold', 'fold']).mean(numeric_only=True).reset_index()

    # --- STEP 3: Calculate the FINAL mean and std DEV across the 5 fold averages ---
    final_summary = fold_avg_df.groupby('threshold').agg(['mean', 'std'])

    # --- Clean up the final DataFrame for better presentation ---
    final_summary.columns = [f'{col[0]}_{col[1]}' for col in final_summary.columns] # Flatten MultiIndex
    
    logging.info("\n\n===== OPTIMAL THRESHOLD EXPERIMENT SUMMARY (Std Dev Across Folds) =====")
    print(final_summary.to_string())
    
    csv_path = os.path.join(args.output, "threshold_summary_by_fold.csv")
    final_summary.to_csv(csv_path)
    logging.info(f"\nSummary table saved to {csv_path}")
    logging.info(f"Visuals saved in: {visuals_dir}")

if __name__ == '__main__':
    main()