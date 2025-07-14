import os
import csv

# Set dataset root
DATASET_ROOT = "/home/smadper/TFM/datasets/laticifers"

gray_dir = os.path.join(DATASET_ROOT, "gray_images")
original_dir = os.path.join(DATASET_ROOT, "original_images")
label_dir = os.path.join(DATASET_ROOT, "labeled_images")
mask_dir = os.path.join(DATASET_ROOT, "masks")

# Output CSV file
output_file = os.path.join(DATASET_ROOT, "laticifer_dataset_index.csv")

# Get list of grayscale images
original_images = sorted([
    f for f in os.listdir(original_dir) if f.lower().endswith((".tif", ".png", ".jpg"))
])

# Create CSV
with open(output_file, mode="w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["original_img_path", "gray_img_path", "enhanced_img_path", "label_img_path", "mask_path", "is_labeled"])

    for img_name in original_images:
        # Base name without extension
        base_name, ext = os.path.splitext(img_name)
        tif_name = f"{base_name}.tif"

        # Check if labeled image exists
        labeled_path = os.path.join("labeled_images", tif_name)
        is_labeled = os.path.exists(os.path.join(DATASET_ROOT, labeled_path))

        # Write row
        writer.writerow([
            os.path.join("original_images", img_name),
            os.path.join("gray_images", tif_name),
            os.path.join("enhanced_images", tif_name),
            os.path.join("labeled_imaes", tif_name) if is_labeled else "",
            os.path.join("masks", tif_name),
            str(is_labeled)
        ])

print(f"Dataset index created at: {output_file}")
