import argparse
import yaml
import os
import torch
import torch.nn.functional as F
import numpy as np
import random
import logging
from PIL import Image
from glob import glob
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2 

def set_seed(seed=42):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def parse_args():
    parser = argparse.ArgumentParser(description="Generate prediction patches and full reconstructed images for the report.")
    parser.add_argument('-c', '--config', required=True, help='Path to the training config YAML file.')
    parser.add_argument('-m', '--model', required=True, help='Path to the trained model checkpoint (.pth).')
    parser.add_argument('-o', '--output', default='report_visuals', help='Root directory to save the output visuals.')
    return parser.parse_args()

def create_qualitative_overlay(image, ground_truth_mask, prediction_mask):
    """
    Creates a color-coded overlay to visualize segmentation performance.
    - Green: True Positive
    - Red: False Positive
    - Yellow: False Negative
    """
    if image.ndim == 2:
        image_color = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        image_color = image.copy()

    # Ensure masks are binary (0 or 1)
    gt = (ground_truth_mask > 0).astype(np.uint8)
    pred = (prediction_mask > 0).astype(np.uint8)

    # Calculate TP, FP, FN
    tp = (gt & pred)
    fp = ((1 - gt) & pred)
    fn = (gt & (1 - pred))

    # Create the overlay image with colors
    overlay = np.zeros_like(image_color)
    overlay[tp == 1] = [0, 255, 0]    # Green for True Positives
    overlay[fp == 1] = [0, 0, 255]    # Red for False Positives
    overlay[fn == 1] = [0, 255, 255]  # Yellow for False Negatives (Green + Red)
    
    # Blend the original image with the overlay
    # A higher alpha for the overlay makes the colors pop
    blended = cv2.addWeighted(image_color, 0.6, overlay, 0.4, 0)
    
    return blended

def main():
    args = parse_args()
    set_seed(42)

    # --- 1. Load Configuration and Model ---
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    logging.info(f"Loading model from: {args.model}")
    conf = yaml.safe_load(open(args.config))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.load(args.model, map_location=device, weights_only=False)
    model.eval()

    # --- 2. Setup Data Paths and Parameters ---
    image_dir = conf['dataset']['feature_dirs']['image']
    mask_dir = conf['dataset']['feature_dirs']['mask']
    patch_h, patch_w = conf['dataset']['patch_size']
    stride_h, stride_w = patch_h // 2, patch_w // 2
    image_filenames = [os.path.basename(f) for f in glob(os.path.join(mask_dir, '*')) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff'))]
    transforms = A.Compose([A.Normalize(mean=(0.5,), std=(0.5,)), ToTensorV2()])

    # --- 3. Main Loop: Process Each Image ---
    logging.info(f"Generating visuals for {len(image_filenames)} images...")
    os.makedirs(args.output, exist_ok=True)
    
    with torch.no_grad():
        for filename in tqdm(image_filenames, desc="Processing Images"):
            image_name = os.path.splitext(filename)[0]
            
            # --- Load Full-Size Image and Ground Truth Mask ---
            image_path = os.path.join(image_dir, filename)
            mask_path = os.path.join(mask_dir, filename)
            
            original_image = np.array(Image.open(image_path).convert("L"))
            ground_truth_mask = np.array(Image.open(mask_path).convert("L"))
            H_orig, W_orig = original_image.shape

            # --- PADDING LOGIC ---
            pad_h = (stride_h - (H_orig - patch_h) % stride_h) % stride_h
            pad_w = (stride_w - (W_orig - patch_w) % stride_w) % stride_w
            padded_image = np.pad(original_image, ((0, pad_h), (0, pad_w)), mode='reflect')
            H_pad, W_pad = padded_image.shape

            # --- Prepare for Reconstruction ---
            sum_preds = torch.zeros((H_pad, W_pad), device=device, dtype=torch.float32)
            count_preds = torch.zeros((H_pad, W_pad), device=device, dtype=torch.float32)

            # --- Sliding Window on PADDED image ---
            for i in range(0, H_pad - patch_h + 1, stride_h):
                for j in range(0, W_pad - patch_w + 1, stride_w):
                    image_patch_np = padded_image[i:i+patch_h, j:j+patch_w]
                    transformed = transforms(image=image_patch_np)
                    input_tensor = transformed['image'].unsqueeze(0).to(device)
                    
                    pred_logits = model(input_tensor)
                    if isinstance(pred_logits, tuple): pred_logits = pred_logits[0]
                    pred_prob = torch.sigmoid(pred_logits).squeeze()

                    sum_preds[i:i+patch_h, j:j+patch_w] += pred_prob
                    count_preds[i:i+patch_h, j:j+patch_w] += 1

            # --- Reconstruct and CROP back to original size ---
            count_preds[count_preds == 0] = 1.0
            avg_preds = sum_preds / count_preds
            final_preds_prob = avg_preds[:H_orig, :W_orig]
            reconstructed_binary_np = (final_preds_prob.cpu().numpy() > 0.5).astype(np.uint8)

            # --- Create and Save the Visuals ---
            # Define output directories
            binary_pred_dir = os.path.join(args.output, "full_predictions_binary")
            overlay_dir = os.path.join(args.output, "full_predictions_overlay")
            gt_dir = os.path.join(args.output, "ground_truths")

            os.makedirs(binary_pred_dir, exist_ok=True)
            os.makedirs(overlay_dir, exist_ok=True)
            os.makedirs(gt_dir, exist_ok=True)

            # 1. Save the binary prediction
            Image.fromarray(reconstructed_binary_np * 255).save(
                os.path.join(binary_pred_dir, f"{image_name}_pred_binary.png")
            )

            # 2. Save the ground truth mask for reference
            Image.fromarray(ground_truth_mask).save(
                os.path.join(gt_dir, f"{image_name}_ground_truth.png")
            )
            
            # 3. Create and save the color-coded overlay
            overlay_image = create_qualitative_overlay(original_image, ground_truth_mask, reconstructed_binary_np)
            # Convert from BGR (OpenCV default) to RGB for saving with PIL
            overlay_image_rgb = cv2.cvtColor(overlay_image, cv2.COLOR_BGR2RGB)
            Image.fromarray(overlay_image_rgb).save(
                os.path.join(overlay_dir, f"{image_name}_pred_overlay.png")
            )

    logging.info(f"Finished generating visuals. All outputs saved in: {args.output}")

if __name__ == '__main__':
    main()