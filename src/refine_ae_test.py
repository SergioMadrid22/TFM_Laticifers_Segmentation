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
import cv2 # Import OpenCV for visualization

# --- Import your existing and new modules ---
from metrics import compute_metrics
from clDice.cldice_metric.cldice import clDice as compute_cldice
from models import DeepClosingRefiner

def set_seed(seed=42):
    np.random.seed(seed); random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed); torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def parse_args():
    p = argparse.ArgumentParser(description="Refine pre-computed segmentation masks using the Deep Closing pipeline.")
    p.add_argument('-c', '--config', required=True, help='Path to the original training config YAML (for dataset paths).')
    p.add_argument('-p', '--predictions_dir', required=True, help='Path to the folder containing the initial binary prediction masks.')
    p.add_argument('--ae_dir', required=True, help='Path to the root experiment directory of the PRE-TRAINED AUTOENCODERS.')
    p.add_argument('-o', '--output', default='deepclosing_results', help='Root directory to save outputs.')
    return p.parse_args()

# --- NEW VISUALIZATION HELPER ---
def create_qualitative_overlay(image, gt_mask, pred_mask):
    """Creates a color-coded TP/FP/FN overlay image."""
    image_color = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    gt = (gt_mask > 0).astype(np.uint8)
    pred = (pred_mask > 0).astype(np.uint8)
    
    tp = (gt & pred); fp = ((1 - gt) & pred); fn = (gt & (1 - pred))
    
    overlay = np.zeros_like(image_color)
    overlay[tp == 1] = [0, 255, 0]    # Green
    overlay[fp == 1] = [0, 0, 255]    # Red
    overlay[fn == 1] = [0, 255, 255]  # Yellow
    
    blended = cv2.addWeighted(image_color, 0.6, overlay, 0.4, 0)
    return cv2.cvtColor(blended, cv2.COLOR_BGR2RGB)


def main():
    args = parse_args()
    conf = yaml.safe_load(open(args.config)); set_seed(conf['train'].get('seed', 42))
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    all_metrics_refined = []
    all_metrics_original = []

    # --- Setup File Paths ---
    mask_dir = conf['dataset']['feature_dirs']['mask']
    image_dir = conf['dataset']['feature_dirs']['image'] # Need this for overlays
    prediction_files = sorted(glob(os.path.join(args.predictions_dir, '*_binary.png')))
    if not prediction_files:
        raise FileNotFoundError(f"No prediction files ending with '_binary.png' found in {args.predictions_dir}")

    # --- (File to fold mapping and autoencoder loading is the same as before) ---
    all_filenames = sorted([f for f in os.listdir(mask_dir) if f.endswith(".tif")])
    kf = KFold(n_splits=5, shuffle=True, random_state=conf['train'].get('seed', 42))
    file_to_fold_map = {}
    for fold_idx, (_, val_idx) in enumerate(kf.split(all_filenames)):
        for i in val_idx:
            file_to_fold_map[all_filenames[i]] = fold_idx + 1
            
    autoencoders = {}
    for fold_num in range(1, 6):
        ae_model_path = os.path.join(args.ae_dir, f"fold_{fold_num}", "best_autoencoder.pth")
        if os.path.exists(ae_model_path):
            autoencoders[fold_num] = torch.load(ae_model_path, map_location='cpu', weights_only=False)
        else:
            logging.warning(f"Autoencoder for fold {fold_num} not found.")

    # --- Create output directory for visuals ---
    visuals_dir = os.path.join(args.output, "visuals")
    os.makedirs(visuals_dir, exist_ok=True)

    # --- Loop through each prediction file ---
    with torch.no_grad():
        for pred_path in tqdm(prediction_files, desc="Refining Predictions"):
            filename_key = os.path.basename(pred_path).replace('_binary.png', '.tif') 
            fold_num = file_to_fold_map.get(filename_key)
            
            if not fold_num or fold_num not in autoencoders:
                logging.warning(f"Could not find a valid fold/autoencoder for {filename_key}. Skipping.")
                continue
            
            refiner = DeepClosingRefiner(autoencoders[fold_num], device=device)

            # --- Load all necessary images ---
            initial_pred_np = (np.array(Image.open(pred_path).convert("L")) > 0).astype(np.uint8)
            gt_mask_np = (np.array(Image.open(os.path.join(mask_dir, filename_key)).convert("L")) > 0).astype(np.uint8)
            original_image_np = np.array(Image.open(os.path.join(image_dir, filename_key)).convert("L"))

            # --- Refine the prediction ---
            initial_pred_tensor = torch.from_numpy(initial_pred_np).unsqueeze(0).unsqueeze(0).float().to(device)
            refinement_dict = refiner(initial_pred_tensor)
            final_closed_mask_tensor = refinement_dict['final_closed_mask']
            
            # --- Convert intermediate steps to numpy for saving ---
            dilated_np = refinement_dict['deep_dilation_output'].squeeze().cpu().numpy().astype(np.uint8)
            final_closed_np = final_closed_mask_tensor.squeeze().cpu().numpy().astype(np.uint8)

            # --- Save Visualizations ---
            base_name = os.path.splitext(filename_key)[0]
            Image.fromarray(initial_pred_np * 255).save(os.path.join(visuals_dir, f"{base_name}_01_initial_pred.png"))
            Image.fromarray(dilated_np * 255).save(os.path.join(visuals_dir, f"{base_name}_02_deep_dilation.png"))
            Image.fromarray(final_closed_np * 255).save(os.path.join(visuals_dir, f"{base_name}_03_deep_closing.png"))
            
            # Create and save overlay of the final result
            overlay = create_qualitative_overlay(original_image_np, gt_mask_np, final_closed_np)
            Image.fromarray(overlay).save(os.path.join(visuals_dir, f"{base_name}_04_final_overlay.png"))

            # --- Compare Metrics ---
            gt_tensor = torch.from_numpy(gt_mask_np).unsqueeze(0).unsqueeze(0).to(device)
            metrics_orig = compute_metrics(initial_pred_tensor.float(), gt_tensor.float())
            metrics_orig['clDice'] = compute_cldice(initial_pred_tensor.squeeze().cpu(), gt_tensor.squeeze().cpu()).item()
            all_metrics_original.append(metrics_orig)

            metrics_refined = compute_metrics(final_closed_mask_tensor.float(), gt_tensor.float())
            metrics_refined['clDice'] = compute_cldice(final_closed_mask_tensor.squeeze().cpu(), gt_tensor.squeeze().cpu()).item()
            all_metrics_refined.append(metrics_refined)
                
    # --- (Final Aggregation and Reporting is the same as before) ---
    if not all_metrics_original:
        logging.error("Processing failed for all files. No metrics were computed.")
        return

    df_orig = pd.DataFrame(all_metrics_original).mean()
    df_refined = pd.DataFrame(all_metrics_refined).mean()
    summary_df = pd.DataFrame([df_orig, df_refined], index=["Standard Model", "Deep Closing Refined"])
    
    logging.info("\n\n===== DEEP CLOSING EXPERIMENT SUMMARY =====")
    print(summary_df.to_string())
    
    os.makedirs(args.output, exist_ok=True)
    summary_df.to_csv(os.path.join(args.output, "deep_closing_summary.csv"))

if __name__ == '__main__':
    main()