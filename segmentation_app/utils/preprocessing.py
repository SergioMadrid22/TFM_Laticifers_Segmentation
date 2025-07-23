import cv2
import numpy as np
import torch
import numpy as np
import torch.nn.functional as F

def apply_clahe(image):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(image)


def predict_image(model, image_np, patch_size=512, stride=256, threshold=0.5, device="cpu"):
    """
    Run patch-based inference on a grayscale image using a sliding window approach.

    Args:
        model: The segmentation model (PyTorch nn.Module).
        image_np: Grayscale image as a NumPy array (H, W).
        patch_size: Size of each input patch (square).
        stride: Sliding window stride.
        threshold: Threshold to binarize output mask.
        device: 'cuda' or 'cpu'

    Returns:
        Binary mask (H, W) as a NumPy array.
    """
    model.to(device)
    model.eval()

    H, W = image_np.shape
    image_tensor = torch.tensor(image_np, dtype=torch.float32).unsqueeze(0).unsqueeze(0) / 255.0  # Shape: (1, 1, H, W)
    image_tensor = image_tensor.to(device)

    # Prepare output canvas
    output = torch.zeros((1, 1, H, W), dtype=torch.float32, device=device)
    count_map = torch.zeros((1, 1, H, W), dtype=torch.float32, device=device)

    # Slide over image
    for top in range(0, H - patch_size + 1, stride):
        for left in range(0, W - patch_size + 1, stride):
            patch = image_tensor[:, :, top:top + patch_size, left:left + patch_size]  # Shape: (1, 1, patch_size, patch_size)

            with torch.no_grad():
                pred = model(patch)
                pred = torch.sigmoid(pred)  # ensure [0,1] probability

            output[:, :, top:top + patch_size, left:left + patch_size] += pred
            count_map[:, :, top:top + patch_size, left:left + patch_size] += 1.0

    # Avoid division by zero
    count_map[count_map == 0] = 1.0
    output /= count_map

    binary_mask = (output > threshold).float()
    mask_np = binary_mask.squeeze().cpu().numpy() * 255  # Convert to uint8 image scale

    return mask_np.astype(np.uint8)
