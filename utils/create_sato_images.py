import os
import cv2
import numpy as np
from skimage.filters import sato
from skimage import img_as_ubyte
from skimage.util import img_as_float
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

# Input directory with grayscale images
input_dir = "datasets/laticifers/enhanced_images"

# Output directory for Sato results
output_base_dir = "datasets/laticifers/sato_images"
os.makedirs(output_base_dir, exist_ok=True)

# Supported image extensions
valid_extensions = (".png", ".jpg", ".jpeg", ".tif", ".tiff",)

# Sigma values
sigma_values = list(range(15, 32, 2))

def process_image(filename):
    if not filename.lower().endswith(valid_extensions):
        return f"[Skipped] {filename}: Invalid extension"

    img_path = os.path.join(input_dir, filename)
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    if img is None:
        return f"[Warning] Could not read {filename}. Skipping."

    img_float = img_as_float(img)
    sato_img = sato(img_float, sigmas=sigma_values)
    sato_img_uint8 = img_as_ubyte(sato_img)

    save_path = os.path.join(output_base_dir, filename)
    cv2.imwrite(save_path, sato_img_uint8)

    return f"[Processed] {filename}"

def main():
    filenames = os.listdir(input_dir)

    with ProcessPoolExecutor() as executor:
        futures = {executor.submit(process_image, fname): fname for fname in filenames}

        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing images"):
            result = future.result()
            print(result)

if __name__ == "__main__":
    main()
