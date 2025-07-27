import os
import json
import argparse
import numpy as np
from tqdm import tqdm
from PIL import Image
import torch
from torch.utils.data import DataLoader
import sys

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.datasets import LaticiferPatchTest
from src.utils import patchify
from src.models import build_model

def load_model(model_dir):
    with open(os.path.join(model_dir, "metadata.json")) as f:
        metadata = json.load(f)
    model = torch.load(os.path.join(model_dir, "best_model.pth"), weights_only=False)
    model.eval()
    return model

def apply_clahe(img_np):
    import cv2
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(img_np)

def is_confident(prob_patch, threshold=0.9):
    # Calculate pixel-level confidence
    # E.g., for binary: probability very close to 1 or 0
    mask = (prob_patch > threshold) | (prob_patch < (1 - threshold))
    confident_fraction = mask.sum() / mask.size
    print(f"Confident fraction: {confident_fraction:.2f}")
    return confident_fraction > 0.9  # at least 90% of pixels are confident

def extract_and_save_confident_patches(
    model, image_path, out_image_dir, out_mask_dir,
    patch_size=512, stride=256, threshold=0.5, conf_thresh=0.9
):
    fname = os.path.splitext(os.path.basename(image_path))[0]
    img = Image.open(image_path).convert("L")
    img_np = np.array(img)
    img_np = apply_clahe(img_np)

    # Convert to tensor
    img_tensor = torch.tensor(img_np / 255.0, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    H, W = img_np.shape
    for y in range(0, H - patch_size + 1, stride):
        for x in range(0, W - patch_size + 1, stride):
            patch = img_tensor[:, :, y:y+patch_size, x:x+patch_size].to(device)

            with torch.no_grad():
                pred = model(patch).squeeze().cpu().numpy()
                #prob = torch.sigmoid(pred).squeeze().cpu().numpy()

            if is_confident(pred, threshold=conf_thresh):
                # Save image patch
                img_patch = img_np[y:y+patch_size, x:x+patch_size]
                mask_patch = (pred > threshold).astype(np.uint8) * 255

                out_img_name = f"{fname}_y{y}_x{x}.png"
                out_mask_name = f"{fname}_y{y}_x{x}.png"
                print(f"mask max: {mask_patch.max()}, min: {mask_patch.min()}")
                Image.fromarray(img_patch).save(os.path.join(out_image_dir, out_img_name))
                Image.fromarray(mask_patch).save(os.path.join(out_mask_dir, out_mask_name))

def main(args):
    model = load_model(args.model_dir)

    os.makedirs(args.output_images, exist_ok=True)
    os.makedirs(args.output_masks, exist_ok=True)

    image_list = [
        os.path.join(args.image_dir, f)
        for f in os.listdir(args.image_dir)
        if f.lower().endswith((".png", ".jpg", ".jpeg", ".tif", ".tiff"))
    ]

    for img_path in tqdm(image_list, desc="Extracting confident patches"):
        extract_and_save_confident_patches(
            model=model,
            image_path=img_path,
            out_image_dir=args.output_images,
            out_mask_dir=args.output_masks,
            patch_size=args.patch_size,
            stride=args.stride,
            threshold=args.threshold,
            conf_thresh=args.confidence
        )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract confident pseudo-label patches from unlabeled data")
    parser.add_argument("--model_dir", type=str, required=True)
    parser.add_argument("--image_dir", type=str, default="datasets/laticifers/enhanced_images")
    parser.add_argument("--output_images", type=str, default="confident_patches/images")
    parser.add_argument("--output_masks", type=str, default="confident_patches/masks")
    parser.add_argument("--patch_size", type=int, default=512)
    parser.add_argument("--stride", type=int, default=256)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--confidence", type=float, default=0.9, help="Confidence threshold (default=0.9)")
    args = parser.parse_args()
    main(args)