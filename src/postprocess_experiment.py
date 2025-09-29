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
from skimage.morphology import (binary_closing, binary_opening, 
                                remove_small_objects, skeletonize, 
                                binary_dilation, reconstruction, disk)
from skimage.measure import label
from skan import Skeleton, summarize

# --- Import your existing modules ---
# Ensure these files are in the same directory or your Python path
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
    parser = argparse.ArgumentParser(description="Run cross-validated inference experiments (TTA, Post-processing).")
    parser.add_argument('-c', '--config', required=True, help='Path to the original training config YAML file.')
    parser.add_argument('-d', '--directory', required=True, help='Path to the root experiment directory containing fold checkpoints (e.g., checkpoints_cv3/unet/experiment_name).')
    parser.add_argument('-o', '--output', default='cv_inference_experiments', help='Root directory to save outputs.')
    return parser.parse_args()

# ------------------ TTA Function (for batches of patches) -------------------
def apply_tta_on_patches(model, patch_batch):
    flips = [
        lambda x: x,
        lambda x: torch.flip(x, dims=[-1]),  # horizontal
        lambda x: torch.flip(x, dims=[-2]),  # vertical
        lambda x: torch.flip(x, dims=[-1, -2])  # both
    ]
    all_preds = []
    for flip_fn in flips:
        flipped_input = flip_fn(patch_batch)
        with torch.no_grad():
            pred_logits = model(flipped_input)
            if isinstance(pred_logits, tuple): pred_logits = pred_logits[0]
            # Crucially, un-flip the output before applying sigmoid
            pred_logits = flip_fn(pred_logits)
        all_preds.append(torch.sigmoid(pred_logits))
    return torch.stack(all_preds, dim=0).mean(dim=0)

# ------------------ Postprocessing & Overlay Functions -------------------
def apply_hysteresis_thresholding(prob_map, low_thresh, high_thresh):
    seeds = prob_map > high_thresh
    candidates = prob_map > low_thresh
    reconstructed = reconstruction(seeds, candidates, method='dilation')
    return reconstructed.astype(np.uint8)

def apply_morphological_postprocessing(mask_np, method,
                                     min_size=400,
                                     opening_radius=8,
                                     closing_radius=8,
                                     min_branch_len=5,
                                     dilate_radius=4):
    if method == "none":
        return mask_np
    
    mask_bool = mask_np.astype(bool)
    
    if method == "opening_then_closing":
        opened = binary_opening(mask_bool, disk(opening_radius))
        closed = binary_closing(opened, disk(closing_radius))
        return remove_small_objects(closed, min_size=min_size).astype(np.uint8)
        
    elif method == "skeleton_prune_reconnect":
        if not np.any(mask_bool): return mask_np
        skeleton = Skeleton(mask_bool)
        branch_data = summarize(skeleton)
        pruned_skeleton_img = np.zeros_like(mask_bool, dtype=np.uint8)
        for index in range(skeleton.n_paths):
            if branch_data.loc[index, 'branch-distance'] >= min_branch_len:
                path_coords = skeleton.path_coordinates(index)
                pruned_skeleton_img[path_coords[:, 0], path_coords[:, 1]] = 1
        
        if not np.any(pruned_skeleton_img): return np.zeros_like(mask_np, dtype=np.uint8)
        reconnected_mask = binary_dilation(pruned_skeleton_img, disk(dilate_radius))
        final_mask = remove_small_objects(reconnected_mask, min_size=min_size)
        return final_mask.astype(np.uint8)
        
    else:
        raise ValueError(f"Unknown morphological postprocessing method: {method}")

def create_qualitative_overlay(image, gt_mask, pred_mask):
    image_color = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    gt = (gt_mask > 0).astype(np.uint8); pred = (pred_mask > 0).astype(np.uint8)
    tp = (gt & pred); fp = ((1-gt) & pred); fn = (gt & (1-pred))
    overlay = np.zeros_like(image_color); overlay[tp == 1] = [0,255,0]; overlay[fp == 1] = [0,0,255]; overlay[fn == 1] = [0,255,255]
    return cv2.cvtColor(cv2.addWeighted(image_color, 0.6, overlay, 0.4, 0), cv2.COLOR_BGR2RGB)

def calculate_and_save(experiment_name, pred_mask_np, gt_mask_np, original_image_np, filename, output_root, device):
    pred_tensor = torch.from_numpy(pred_mask_np).unsqueeze(0).unsqueeze(0).to(device)
    gt_tensor = torch.from_numpy(gt_mask_np).unsqueeze(0).unsqueeze(0).to(device)
    metrics = compute_metrics(pred_tensor.float(), gt_tensor.float())
    
    metrics['experiment'] = experiment_name
    metrics['image_file'] = filename

    visual_output_dir = os.path.join(output_root, "visuals", experiment_name)
    os.makedirs(visual_output_dir, exist_ok=True)
    image_name = os.path.splitext(filename)[0]
    
    Image.fromarray(pred_mask_np * 255).save(os.path.join(visual_output_dir, f"{image_name}_binary.png"))
    overlay_img = create_qualitative_overlay(original_image_np, gt_mask_np, pred_mask_np)
    Image.fromarray(overlay_img).save(os.path.join(visual_output_dir, f"{image_name}_overlay.png"))
    
    return metrics

# ------------------ MAIN FUNCTION -------------------
def main():
    args = parse_args()
    conf = yaml.safe_load(open(args.config)); set_seed(conf['train'].get('seed', 42))
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- DEFINE THE FINAL EXPERIMENTAL MATRIX ---
    experiments = {
        "A_Hysteresis_Baseline": {"use_tta": False, "postprocess": "none"},
        "B_Hysteresis_TTA": {"use_tta": True, "postprocess": "none"},
        "C_Hysteresis_OpeningClosing": {"use_tta": False, "postprocess": "opening_then_closing"},
        "D_Hysteresis_SkeletonPrune": {"use_tta": False, "postprocess": "skeleton_prune_reconnect"},
        "E_Hysteresis_TTA_SkeletonPrune": {"use_tta": True, "postprocess": "skeleton_prune_reconnect"},
        "F_Hysteresis_TTA_OpeningClosing": {"use_tta": True, "postprocess": "opening_then_closing"},
    }
    HYST_LOW, HYST_HIGH = 0.3, 0.7

    prob_maps_no_tta, prob_maps_with_tta = {}, {}
    ground_truths, original_images = {}, {}

    mask_dir = conf['dataset']['feature_dirs']['mask']
    image_dir = conf['dataset']['feature_dirs']['image']
    all_filenames = sorted([f for f in os.listdir(mask_dir) if f.endswith(".tif")])
    kf = KFold(n_splits=5, shuffle=True, random_state=conf['train'].get('seed', 42))

    # =========================================================================
    # STAGE 1: Generate all raw probability map predictions (once per image)
    # =========================================================================
    logging.info(f"Saving all outputs to: {args.output}")
    logging.info("Stage 1: Generating raw probability map predictions (with and without TTA)...")
    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(all_filenames)):
        fold_num = fold_idx + 1
        model_path = os.path.join(args.directory, f"fold_{fold_num}", "best_model_soft_dice.pth")
        if not os.path.exists(model_path): continue
        model = torch.load(model_path, map_location=device, weights_only=False).eval()
        val_files = [all_filenames[i] for i in val_idx]
        _, val_loader = get_patch_dataloaders(conf, val_filenames=val_files)

        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Inferring Fold {fold_num}"):
                filename = batch['filename'][0]
                if filename in prob_maps_no_tta: continue

                image_patches = batch['image_patches'].to(device)
                B, N, C, H, W = image_patches.shape
                image_patches = image_patches.view(B*N, C, H, W)

                # Inference WITHOUT TTA
                pred_logits = model(image_patches)
                if isinstance(pred_logits, tuple): pred_logits = pred_logits[0]
                pred_probs_no_tta = torch.sigmoid(pred_logits)
                
                # Inference WITH TTA
                pred_probs_with_tta = apply_tta_on_patches(model, image_patches)

                def reconstruct(probs):
                    full_prob = reconstruct_from_patches(probs, batch['coords'], batch['image_size'], conf['dataset']['patch_size'])
                    H_orig, W_orig = batch['original_size']
                    return full_prob[:H_orig, :W_orig].cpu().numpy()

                prob_maps_no_tta[filename] = reconstruct(pred_probs_no_tta)
                prob_maps_with_tta[filename] = reconstruct(pred_probs_with_tta)
                
                ground_truths[filename] = (np.array(Image.open(os.path.join(mask_dir, filename)).convert("L")) > 0).astype(np.uint8)
                original_images[filename] = np.array(Image.open(os.path.join(image_dir, filename)).convert("L"))

    # =========================================================================
    # STAGE 2: Apply experiments, compute metrics, and save visuals
    # =========================================================================
    logging.info("Stage 2: Applying post-processing and computing metrics...")
    all_individual_results = []
    
    for filename in tqdm(ground_truths.keys(), desc="Applying Experiments"):
        for exp_name, exp_config in experiments.items():
            prob_map_np = prob_maps_with_tta[filename] if exp_config["use_tta"] else prob_maps_no_tta[filename]
            pred_binary_np = apply_hysteresis_thresholding(prob_map_np, HYST_LOW, HYST_HIGH)
            #pred_binary_np = (prob_map_np > 0.5).astype(np.uint8)  # Using simple thresholding instead of hysteresis for clarity
            pred_processed_np = apply_morphological_postprocessing(pred_binary_np, exp_config["postprocess"])
            
            metrics = calculate_and_save(
                exp_name, pred_processed_np, ground_truths[filename], 
                original_images[filename], filename, args.output, device
            )
            all_individual_results.append(metrics)

    # =========================================================================
    # STAGE 3: Final Aggregation and Reporting
    # =========================================================================
    if not all_individual_results:
        logging.error("No results were generated. Exiting."); return

    full_results_df = pd.DataFrame(all_individual_results)
    grouped = full_results_df.groupby('experiment')
    
    mean_stats = grouped.mean(numeric_only=True)
    std_stats = grouped.std(numeric_only=True)

    summary_df = pd.DataFrame()
    for col in mean_stats.columns:
        if col not in ['fold']:
            summary_df[f"{col}_mean"] = mean_stats[col]
            summary_df[f"{col}_std"] = std_stats[col]

    logging.info("\n\n===== FINAL CROSS-VALIDATED INFERENCE EXPERIMENT SUMMARY =====")
    
    column_order = []
    # Ensure a logical order in the final table
    metric_keys = ['Dice', 'clDice', 'IoU', 'HD95'] 
    for metric in metric_keys:
        if f"{metric}_mean" in summary_df.columns:
            column_order.append(f"{metric}_mean")
            column_order.append(f"{metric}_std")
    
    summary_df = summary_df[column_order]
    
    print(summary_df.to_string())
    
    os.makedirs(args.output, exist_ok=True)
    detailed_csv_path = os.path.join(args.output, "cv_inference_detailed_results.csv")
    summary_csv_path = os.path.join(args.output, "cv_inference_summary.csv")
    
    full_results_df.to_csv(detailed_csv_path, index=False)
    summary_df.to_csv(summary_csv_path)
    
    logging.info(f"\nDetailed results saved to {detailed_csv_path}")
    logging.info(f"Summary saved to {summary_csv_path}")

if __name__ == '__main__':
    main()