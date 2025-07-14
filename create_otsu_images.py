import os
import cv2
from skimage import filters
import numpy as np

# Define input and output directories
input_dir = "datasets/laticifers/enhanced_images"
output_dir = "datasets/laticifers/otsu_images"

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Supported image extensions
valid_extensions = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")

# Process each image in the input directory
for filename in os.listdir(input_dir):
    if not filename.lower().endswith(valid_extensions):
        continue

    # Read image in grayscale
    img_path = os.path.join(input_dir, filename)
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    if img is None:
        print(f"[Warning] Could not read {filename}. Skipping.")
        continue

    # Apply Otsu thresholding
    otsu_thresh = filters.threshold_otsu(img)
    binary_img = (img > otsu_thresh).astype(np.uint8) * 255

    # Save the binary image
    save_path = os.path.join(output_dir, filename)
    cv2.imwrite(save_path, binary_img)
    print(f"[OK] Saved Otsu binary: {save_path}")
