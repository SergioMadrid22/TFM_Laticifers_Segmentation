import os
import json
import argparse
import numpy as np
from PIL import Image
import napari
from qtpy.QtWidgets import QFileDialog

def load_image(path):
    return np.array(Image.open(path).convert("L"))

def load_mask(path):
    return np.array(Image.open(path).convert("L")) // 255  # Convert to 0/1 mask

def save_mask(mask_np, path):
    mask_img = Image.fromarray((mask_np * 255).astype(np.uint8))
    mask_img.save(path)
    print(f"✅ Corrected mask saved to: {path}")

def run_gui(image_path, pred_mask_path, output_path):
    # Load image and predicted mask from disk (no inference here)
    img_np = load_image(image_path)
    pred_mask_np = load_mask(pred_mask_path)

    # Launch Napari
    viewer = napari.Viewer()
    viewer.add_image(img_np, name="Original Image", colormap='gray', contrast_limits=[0, 255])
    label_layer = viewer.add_labels(pred_mask_np, name="Predicted Mask", opacity=0.6)

    print("🖱 Use the brush and eraser tools in Napari to correct the mask.")
    print("💾 After you're done, close the viewer to save the corrected mask.")

    @viewer.window.add_dock_widget(name="Save Corrected Mask", area="right")
    def save_button():
        file_path, _ = QFileDialog.getSaveFileName(
            caption="Save corrected mask as",
            directory=output_path,
            filter="PNG files (*.png);;TIFF files (*.tif);;All files (*)"
        )
        if file_path:
            save_mask(label_layer.data.astype(np.uint8), file_path)

    napari.run()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Mask correction tool with Napari")
    parser.add_argument("--input", type=str, required=True, help="Path to input image")
    parser.add_argument("--pred_mask", type=str, required=True, help="Path to precomputed predicted mask")
    parser.add_argument("--output", type=str, default="corrected_mask.png", help="Path to save corrected mask")
    args = parser.parse_args()

    run_gui(args.input, args.pred_mask, args.output)
