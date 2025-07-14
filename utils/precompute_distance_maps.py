import os
import torch
import numpy as np
from glob import glob
from tqdm import tqdm
from PIL import Image
from scipy.ndimage import distance_transform_edt

import torchvision.transforms.functional as TF

# === CONFIGURATION ===
MASK_DIR = '/home/smadper/TFM/datasets/laticifers/masks'            # directory with your mask images
SAVE_PT_DIR = '/home/smadper/TFM/datasets/laticifers/distance_maps_pt'
SAVE_IMG_DIR = '/home/smadper/TFM/datasets/laticifers/distance_maps_img'
RESIZE_TO = None  # (H, W), or None to keep original size

# Create save directories if not exist
os.makedirs(SAVE_PT_DIR, exist_ok=True)
os.makedirs(SAVE_IMG_DIR, exist_ok=True)

# Load all mask paths
mask_paths = sorted(glob(os.path.join(MASK_DIR, '*.tif')))

# === PROCESS EACH MASK ===
for mask_path in tqdm(mask_paths, desc="Processing masks"):
    # Load mask as binary
    mask_img = Image.open(mask_path).convert('L')
    if RESIZE_TO:
        mask_img = mask_img.resize(RESIZE_TO, Image.NEAREST)

    mask_np = np.array(mask_img)
    mask_bin = (mask_np > 127).astype(np.uint8)

    # Compute distance map (from background to foreground)
    dist_map = distance_transform_edt(1 - mask_bin).astype(np.float32)

    # Normalize to [0, 1]
    max_val = dist_map.max()
    dist_map_norm = dist_map / (max_val + 1e-8)

    # Save normalized .pt tensor
    dist_tensor = torch.from_numpy(dist_map_norm).unsqueeze(0)  # [1, H, W]
    base_name = os.path.splitext(os.path.basename(mask_path))[0]
    save_pt_path = os.path.join(SAVE_PT_DIR, base_name + '.pt')
    torch.save(dist_tensor, save_pt_path)

    # Save normalized image as 16-bit PNG
    dist_img_16bit = (dist_map_norm * 65535).astype(np.uint16)
    save_img_path = os.path.join(SAVE_IMG_DIR, base_name + '.png')
    Image.fromarray(dist_img_16bit).save(save_img_path)