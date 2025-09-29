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
                image_patches = batch['image_patches'].to(device)
                B, N, C, H, W = image_patches.shape
                image_patches = image_patches.view(B*N, C, H, W)

                pred_logits = model(image_patches)
                if isinstance(pred_logits, tuple): pred_logits = pred_logits[0]
                pred_probs = torch.sigmoid(pred_logits)
                
                pred_full_prob = reconstruct_from_patches(pred_probs, batch['coords'], batch['image_size'], conf['dataset']['patch_size'])
                
                H_orig, W_orig = batch['original_size']
                pred_full_prob_cpu = pred_full_prob[:H_orig, :W_orig].cpu().numpy()
                
                image_name = os.path.splitext(batch['filename'][0])[0]
                image_path = os.path.join(conf['dataset']['feature_dirs']['image'], batch['filename'][0])
                original_image_np = np.array(Image.open(image_path).convert("L"))
                gt_mask_np = np.array(Image.open(os.path.join(mask_dir, batch['filename'][0])).convert("L"))

                image_visual_dir = os.path.join(visuals_dir, image_name)
                os.makedirs(image_visual_dir, exist_ok=True)
                
                mask_full = reconstruct_from_patches(batch['mask_patches'].squeeze(0), batch['coords'], batch['image_size'], conf['dataset']['patch_size'])
                mask_full = mask_full[:H_orig, :W_orig]
                gt_tensor = mask_full.unsqueeze(0).unsqueeze(0).to(device)

                for threshold in thresholds_to_test:
                    pred_binary_np = (pred_full_prob_cpu > threshold).astype(np.uint8)
                    
                    overlay_img = create_qualitative_overlay(original_image_np, gt_mask_np, pred_binary_np)
                    save_path = os.path.join(image_visual_dir, f"overlay_thresh_{threshold:.2f}.png")
                    Image.fromarray(overlay_img).save(save_path)
                    Image.fromarray(pred_binary_np * 255).save(os.path.join(image_visual_dir, f"binary_{threshold:.2f}.png"))
                    
                    pred_tensor = torch.from_numpy(pred_binary_np).unsqueeze(0).unsqueeze(0).to(device)
                    metrics = compute_metrics(pred_tensor.float(), gt_tensor.float())
                    
                    # Re-enabled clDice calculation
                    metrics['clDice'] = compute_cldice(pred_tensor.squeeze().cpu(), gt_tensor.squeeze().cpu()).item()
                    
                    metrics['threshold'] = threshold
                    metrics['image_file'] = batch['filename'][0]
                    all_results.append(metrics)

    # --- Final Aggregation and Reporting (CORRECTED) ---
    if not all_results:
        logging.error("No results were generated. Exiting.")
        return
    
    full_results_df = pd.DataFrame(all_results)
    
    # Select only the numeric columns for aggregation
    numeric_cols = full_results_df.select_dtypes(include=np.number).columns.tolist()
    
    grouped = full_results_df.groupby('threshold')
    
    # Perform aggregation only on the numeric columns
    mean_stats = grouped[numeric_cols].mean()
    std_stats = grouped[numeric_cols].std()
    
    summary_df = pd.DataFrame()
    for col in mean_stats.columns:
        summary_df[f"{col}_mean"] = mean_stats[col]
        summary_df[f"{col}_std"] = std_stats[col]
            
    logging.info("\n\n===== OPTIMAL THRESHOLD EXPERIMENT SUMMARY =====")
    print(summary_df.to_string())
    
    csv_path = os.path.join(args.output, "threshold_summary.csv")
    summary_df.to_csv(csv_path)
    logging.info(f"\nSummary table saved to {csv_path}")
    logging.info(f"Visuals saved in: {visuals_dir}")

if __name__ == '__main__':
    main()