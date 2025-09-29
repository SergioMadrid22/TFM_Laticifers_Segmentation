import os
import numpy as np
import cv2
from PIL import Image
from glob import glob
from tqdm import tqdm

# --- CONFIGURATION ---
# The directory containing your original, thin ground truth masks
ORIGINAL_MASK_DIR = '/data/smadper@alumno.upv.es/TFM/datasets/laticifers/masks'

# The root directory where the new curriculum masks will be saved
CURRICULUM_ROOT_DIR = '/data/smadper@alumno.upv.es/TFM/datasets/laticifers/curriculum_masks'

# Define the curriculum levels you want to pre-compute
# Level '0' is the original, '1' is 1 iteration of dilation, etc.
CURRICULUM_LEVELS_TO_COMPUTE = [1, 2, 3, 4]

# --- SCRIPT ---
def precompute_dilated_masks(original_dir, save_root, levels):
    """
    Pre-computes and saves morphologically dilated masks for curriculum learning.
    """

    # Ensure the glob pattern matches your original mask format
    original_mask_paths = sorted(glob(os.path.join(original_dir, '*.tif')))
    if not original_mask_paths:
        print(f"Error: No .tif masks found in {original_dir}")
        return

    # Define the kernel for dilation (a simple 3x3 square)
    kernel = np.ones((3, 3), np.uint8)

    for level in levels:
        if level <= 0:
            continue # Level 0 is the original, no need to process

        # Create a subdirectory for this curriculum level
        level_save_dir = os.path.join(save_root, f'level_{level}')
        os.makedirs(level_save_dir, exist_ok=True)
        print(f"\n--- Generating masks for Curriculum Level {level} ---")
        print(f"Saving to: {level_save_dir}")

        for mask_path in tqdm(original_mask_paths, desc=f"Level {level}"):
            try:
                # Load the original mask
                mask_img = Image.open(mask_path).convert('L')
                mask_np = np.array(mask_img)
                mask_bin = (mask_np > 127).astype(np.uint8)

                # Apply dilation
                dilated_mask = cv2.dilate(mask_bin, kernel, iterations=level)

                # --- MODIFIED PART ---
                # Convert the numpy array back to a PIL Image
                # Multiply by 255 to save as a standard black and white image
                dilated_mask_img = Image.fromarray(dilated_mask * 255)
                
                # Get the original filename to keep it consistent
                base_name = os.path.basename(mask_path)
                
                # Construct the full save path with the .tif extension
                save_path = os.path.join(level_save_dir, base_name)

                # Save the image in TIFF format. Pillow automatically detects the format
                # from the file extension. You can also specify it explicitly.
                dilated_mask_img.save(save_path, format='TIFF')

            except Exception as e:
                print(f"Could not process {mask_path} for level {level}: {e}")
    
    print("\n--- Pre-computation complete! ---")

if __name__ == '__main__':
    precompute_dilated_masks(ORIGINAL_MASK_DIR, CURRICULUM_ROOT_DIR, CURRICULUM_LEVELS_TO_COMPUTE)