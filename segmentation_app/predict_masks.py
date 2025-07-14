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
# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.utils import reconstruct_from_patches, extract_patches
from src.models import build_model
from src.datasets import LaticiferPatchTest  # adjust if needed

def load_model_from_metadata(meta_dict, model_path):
    model = torch.load(os.path.join(model_path, "best_model.pth"), weights_only=False)
    model.eval()
    return model

def predict_with_dataset(model, image_path, output_path, patch_size=(512, 512), stride=(256, 256), use_clahe=True):
    # Setup temporary dataset
    tmp_root = "temp_test_dataset"
    os.makedirs(os.path.join(tmp_root, "gray_images"), exist_ok=True)
    os.makedirs(os.path.join(tmp_root, "masks"), exist_ok=True)

    # Convert and save image
    fname = os.path.basename(image_path)
    img = Image.open(image_path).convert("L")
    img.save(os.path.join(tmp_root, "gray_images", fname))

    # Dummy mask (zeros)
    dummy_mask = Image.fromarray(np.zeros_like(np.array(img), dtype=np.uint8))
    dummy_mask.save(os.path.join(tmp_root, "masks", fname))

    # Dataset and DataLoader
    dataset = LaticiferPatchTest([fname], root_dir=tmp_root, patch_size=patch_size, stride=stride, use_clahe=use_clahe)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    # Predict over all patches and reconstruct
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    for batch in dataloader:
        image_patches = batch['image_patches'].squeeze(0).to(device)  # (N, 1, H, W)
        coords = batch['coords']
        image_size = tuple(batch['image_size'])
        original_size = tuple(batch['original_size'])

        preds = []
        with torch.no_grad():
            for i in range(0, image_patches.size(0), 8):
                pred = model(image_patches[i:i+8])
                preds.append(pred.cpu())

        preds = torch.cat(preds, dim=0)  # (N, 1, H, W)
        pred_full = reconstruct_from_patches(preds, coords, image_size, patch_size)
        pred_full = pred_full[:original_size[0], :original_size[1]].numpy()
        pred_mask = (pred_full * 255).astype(np.uint8)

        # Save
        output_path = os.path.splitext(output_path)[0] + ".png"
        Image.fromarray(pred_mask).save(output_path)
        print(f"Saved: {output_path}")

    # Cleanup temp folder
    if os.path.exists(tmp_root):
        shutil.rmtree(tmp_root)

    return pred_mask


def main(args):
    # Load model metadata
    with open(os.path.join(args.model_dir, "metadata.json")) as f:
        metadata = json.load(f)

    # Load model
    model = load_model_from_metadata(metadata, args.model_dir)

    # Process input(s)
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
            predict_with_dataset(model, img_path, out_path)
    else:
        out_path = os.path.join(output_dir, os.path.basename(input_path))
        predict_with_dataset(model, input_path, out_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Segment images using a trained model")
    parser.add_argument("--model_dir", type=str, required=True, help="Path to model directory containing best_model.pth and metadata.json")
    parser.add_argument("--input", type=str, required=True, help="Path to image file or directory")
    parser.add_argument("--output", type=str, required=True, help="Directory to save predicted masks")
    args = parser.parse_args()
    main(args)
