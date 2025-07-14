import torch
import numpy as np
from torchvision import transforms
import sys
import os
import matplotlib.pyplot as plt
import albumentations as A
from albumentations.pytorch import ToTensorV2
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.utils import reconstruct_from_patches, extract_patches

import os
import cv2  # for saving images

def predict_image(model, image_np, patch_size=512, patch_stride=256, debug_patches_dir="test_dir"):
    """
    Predicts the segmentation mask for a single image.

    Args:
        model: PyTorch model (already loaded and in eval mode).
        image_np: NumPy array of the grayscale input image (H, W).
        patch_size: Size of each square patch.
        patch_stride: Stride between patches.
        debug_patches_dir: Optional path to save predicted patches for debugging.

    Returns:
        NumPy array of predicted mask (H, W), dtype=np.uint8.
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    if isinstance(patch_size, int):
        patch_size = (patch_size, patch_size)
    if isinstance(patch_stride, int):
        patch_stride = (patch_stride, patch_stride)

    # Normalize image to [0, 1]
    image = image_np.astype(np.float32) / 255.0
    H, W = image.shape

    # Convert to tensor and add channel/batch dims
    tensor_img = torch.from_numpy(image).unsqueeze(0).unsqueeze(0)  # shape: (1, 1, H, W)

    # Extract patches
    patches, coords = extract_patches(tensor_img, patch_size, patch_stride)
    if patches.dim() == 5 and patches.shape[2] == 1:
        patches = patches.squeeze(2)

    transform = A.Compose([
        A.Normalize(),
        ToTensorV2()
    ])

    preds = []
    batch_size = 8

    if debug_patches_dir:
        input_dir = os.path.join(debug_patches_dir, "input_patches")
        pred_dir = os.path.join(debug_patches_dir, "predicted_patches")
        os.makedirs(input_dir, exist_ok=True)
        os.makedirs(pred_dir, exist_ok=True)

    patch_idx = 0
    with torch.no_grad():
        for i in range(0, patches.size(0), batch_size):
            batch = patches[i:i+batch_size]  # (B, 1, H, W)
            batch_np = batch.squeeze(1).cpu().numpy()  # shape: (B, H, W)

            batch_tensor = []
            for j, img in enumerate(batch_np):
                # Save input patch (scaled back to 0-255 for visualization)
                if debug_patches_dir:
                    input_patch_img = (img * 255).astype(np.uint8)
                    y, x = coords[i + j]
                    fname_input = f"patch_{patch_idx}_y{y}_x{x}.png"
                    cv2.imwrite(os.path.join(input_dir, fname_input), input_patch_img)

                # Normalize patch for model input
                transformed = transform(image=img)
                batch_tensor.append(transformed['image'])  # shape: (1, H, W)

            batch_tensor = torch.stack(batch_tensor).to(device)
            pred = model(batch_tensor)  # output shape: (B, 1, H, W)
            #pred = torch.sigmoid(pred)  # apply sigmoid if needed
            preds.append(pred.cpu())

            if debug_patches_dir:
                pred_np = pred.squeeze(1).cpu().numpy()  # (B, H, W)
                for j in range(pred_np.shape[0]):
                    patch_pred = pred_np[j]
                    # Scale to 0-255 uint8 for saving
                    print(f"Min: {patch_pred.min()}, Max: {patch_pred.max()}")
                    patch_pred_img = (patch_pred * 255).astype(np.uint8)
                    y, x = coords[i + j]
                    fname_pred = f"patch_{patch_idx}_y{y}_x{x}.png"
                    cv2.imwrite(os.path.join(pred_dir, fname_pred), patch_pred_img)
                    patch_idx += 1

    preds = torch.cat(preds, dim=0)  # (N, 1, H, W)
    pred_full = reconstruct_from_patches(preds, coords, (H, W), patch_size)
    pred_full = pred_full[:H, :W].numpy()
    pred_mask = (pred_full * 255).astype(np.uint8)
    return pred_mask