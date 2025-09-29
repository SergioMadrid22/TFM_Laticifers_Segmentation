import os
import numpy as np
from PIL import Image
from glob import glob
from tqdm import tqdm


MASK_DIR = '/data/smadper@alumno.upv.es/TFM/datasets/laticifers/masks'


def compute_class_proportions(mask_directory):
    """
    Computes the average proportion of foreground (laticifer) and background
    pixels across an entire dataset of binary masks.
    """
    total_pixels = 0
    foreground_pixels = 0

    mask_paths = sorted(glob(os.path.join(mask_directory, '*.tif')))
    if not mask_paths:
        print(f"Error: No .tif masks found in {mask_directory}")
        return

    print(f"Analyzing {len(mask_paths)} masks...")

    for mask_path in tqdm(mask_paths, desc="Processing masks"):
        try:
            mask_img = Image.open(mask_path).convert('L')
            mask_np = np.array(mask_img)
            
            # Binarize the mask
            mask_bin = (mask_np > 127).astype(np.uint8)
            
            # Update counts
            total_pixels += mask_bin.size
            foreground_pixels += mask_bin.sum()

        except Exception as e:
            print(f"Could not process {mask_path}: {e}")

    if total_pixels == 0:
        print("Error: No pixels were processed.")
        return

    # Calculate proportions
    background_pixels = total_pixels - foreground_pixels
    
    prop_foreground = (foreground_pixels / total_pixels) * 100
    prop_background = (background_pixels / total_pixels) * 100

    print("\n--- Dataset Class Imbalance Report ---")
    print(f"Total Pixels Analyzed: {total_pixels:,}")
    print(f"Foreground (Laticifer) Pixels: {foreground_pixels:,}")
    print(f"Background Pixels: {background_pixels:,}")
    print("-" * 35)
    print(f"Foreground Proportion: {prop_foreground:.4f}%")
    print(f"Background Proportion: {prop_background:.4f}%")
    print(f"Imbalance Ratio (Background:Foreground): {prop_background/prop_foreground:.1f} : 1")
    print("-" * 35)
    
    return prop_foreground, prop_background

if __name__ == '__main__':
    compute_class_proportions(MASK_DIR)