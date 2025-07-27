import os
import argparse
import torch
import json
import numpy as np
from PIL import Image
from tqdm import tqdm
import sys
import shutil
from torch.utils.data import DataLoader
import torch.cuda.amp

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.utils import reconstruct_from_patches
from src.models import build_model
from src.datasets import LaticiferPatchTest


def load_model_from_metadata(meta_dict, model_path):
    model = torch.load(os.path.join(model_path, "best_model.pth"), weights_only=False)
    model.eval()
    return model


def apply_clahe(image_np):
    import cv2
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(image_np)


def predict_with_dataset(
        model, 
        image_path, 
        output_path, 
        patch_size=(512, 512), 
        stride=(256, 256), 
        use_clahe=True,
        threshold=0.5
    ):
    tmp_root = "temp_test_dataset"
    os.makedirs(os.path.join(tmp_root, "enhanced"), exist_ok=True)
    os.makedirs(os.path.join(tmp_root, "mask"), exist_ok=True)  # Dummy GT mask

    fname = os.path.basename(image_path)
    img = Image.open(image_path).convert("L")
    img_np = np.array(img)

    # Optionally apply CLAHE
    if use_clahe:
        img_np = apply_clahe(img_np)

    # Save enhanced image
    Image.fromarray(img_np).save(os.path.join(tmp_root, "enhanced", fname))

    # Save dummy mask
    dummy_mask = Image.fromarray(np.zeros_like(img_np, dtype=np.uint8))
    dummy_mask.save(os.path.join(tmp_root, "mask", fname))

    feature_dirs = {
        "enhanced": os.path.join(tmp_root, "enhanced"),
        "mask": os.path.join(tmp_root, "mask")
    }

    dataset = LaticiferPatchTest(
        feature_dirs=feature_dirs,
        patch_size=patch_size,
        stride=stride,
        dist_transform=False,
        filenames=[fname]
    )

    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    scaler = torch.cuda.amp.GradScaler(enabled=False)  # scaler not really needed in inference, but keeping style
    
    for batch in dataloader:
        image_patches = batch['image_patches'].squeeze(0).to(device)
        coords = batch['coords']
        image_size = batch['image_size']
        original_size = batch['original_size']

        preds = []
        with torch.no_grad():
            for i in range(0, image_patches.size(0), 8):
                with torch.cuda.amp.autocast(enabled=device.type=='cuda'):
                    pred = model(image_patches[i:i+8])
                preds.append(pred.cpu())

        preds = torch.cat(preds, dim=0)
        pred_logits_full = reconstruct_from_patches(preds, coords, image_size, patch_size)
        pred_logits_full = pred_logits_full[:original_size[0], :original_size[1]].numpy()
        pred_mask = (pred_logits_full > threshold).astype(np.uint8) * 255

        # Save mask
        output_path = os.path.splitext(output_path)[0] + ".png"
        Image.fromarray(pred_mask).save(output_path)
        print(f"Saved: {output_path}")

    shutil.rmtree(tmp_root)
    return pred_mask

def main(args):
    # Load model metadata
    with open(os.path.join(args.model_dir, "metadata.json")) as f:
        metadata = json.load(f)

    model = load_model_from_metadata(metadata, args.model_dir)

    input_path = args.input
    output_dir = args.output
    os.makedirs(output_dir, exist_ok=True)

    if os.path.isdir(input_path):
        images = [
            os.path.join(input_path, fname)
            for fname in os.listdir(input_path)
            if fname.lower().endswith((".png", ".jpg", ".jpeg", ".tif", ".tiff"))
        ]
        for img_path in tqdm(images, desc="Processing images"):
            fname = os.path.basename(img_path)
            out_path = os.path.join(output_dir, fname)
            predict_with_dataset(model, img_path, out_path, threshold=args.threshold)
    else:
        out_path = os.path.join(output_dir, os.path.basename(input_path))
        predict_with_dataset(model, input_path, out_path, threshold=args.threshold)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Segment images using a trained model")
    parser.add_argument("--model_dir", type=str, required=True, help="Path to model directory containing best_model.pth and metadata.json")
    parser.add_argument("--input", type=str, required=True, help="Path to image file or directory")
    parser.add_argument("--output", type=str, required=True, help="Directory to save predicted masks")
    parser.add_argument("--threshold", type=float, default=0.5, help="Threshold for converting logits to binary mask (default: 0.5)")
    args = parser.parse_args()
    main(args)
