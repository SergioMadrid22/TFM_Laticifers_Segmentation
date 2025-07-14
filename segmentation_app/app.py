import streamlit as st
import os
import torch
import cv2
import json
import numpy as np
from PIL import Image
import sys
import shutil
from torch.utils.data import DataLoader

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.datasets import LaticiferPatchTest
from src.utils import reconstruct_from_patches
from src.metrics import compute_dice, compute_iou

st.set_page_config(page_title="Segmentation GUI", layout="centered")
st.title("🔬 Laticifer Segmentation Tool")

MODEL_ROOT = "segmentation_app/checkpoints"

@st.cache_resource
def load_all_models_metadata():
    metadata_dict = {}
    for model_dir in os.listdir(MODEL_ROOT):
        model_path = os.path.join(MODEL_ROOT, model_dir)
        if not os.path.isdir(model_path):
            continue
        for exp_dir in os.listdir(model_path):
            exp_path = os.path.join(model_path, exp_dir)
            metadata_file = os.path.join(exp_path, "metadata.json")
            if os.path.exists(metadata_file):
                with open(metadata_file, "r") as f:
                    metadata = json.load(f)
                    key = f"{model_dir} / {exp_dir}"
                    metadata_dict[key] = {
                        "path": exp_path,
                        "metadata": metadata
                    }
    return metadata_dict


@st.cache_resource
def load_model_from_metadata(meta_dict, model_path):
    model = torch.load(os.path.join(model_path, "best_model.pth"), weights_only=False)
    model.eval()
    return model


def apply_clahe(image):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(image)


# --- Auto-compute best threshold ---
def find_best_threshold(pred_logits, gt_mask_np, num_thresholds=100):
    gt_tensor = torch.tensor(gt_mask_np / 255.0).unsqueeze(0).unsqueeze(0).float()
    thresholds = np.linspace(0, 1, num_thresholds)
    best_dice = -1
    best_thresh = 0.5

    for t in thresholds:
        pred_tensor = torch.tensor((pred_logits > t).astype(np.float32)).unsqueeze(0).unsqueeze(0)
        dice = compute_dice(pred_tensor, gt_tensor)
        if dice > best_dice:
            best_dice = dice
            best_thresh = t

    return best_thresh, best_dice


def predict_with_dataset(model, pil_image, patch_size=(512, 512), stride=(256, 256), use_clahe=True):
    tmp_root = "temp_test_dataset"
    os.makedirs(os.path.join(tmp_root, "gray_images"), exist_ok=True)
    os.makedirs(os.path.join(tmp_root, "masks"), exist_ok=True)

    fname = "uploaded_image.png"
    image_np = np.array(pil_image.convert("L"))
    pil_image.save(os.path.join(tmp_root, "gray_images", fname))

    dummy_mask = Image.fromarray(np.zeros_like(image_np, dtype=np.uint8))
    dummy_mask.save(os.path.join(tmp_root, "masks", fname))

    dataset = LaticiferPatchTest([fname], root_dir=tmp_root, patch_size=patch_size, stride=stride, use_clahe=use_clahe)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    pred_mask = None
    pred_logits = []
    for batch in dataloader:
        image_patches = batch['image_patches'].squeeze(0).to(device)
        coords = batch['coords']
        image_size = batch['image_size']
        original_size = batch['original_size']

        preds = []
        with torch.no_grad():
            for i in range(0, image_patches.size(0), 8):
                pred = model(image_patches[i:i+8])
                preds.append(pred.cpu())

        preds = torch.cat(preds, dim=0)
        pred_logits_full = reconstruct_from_patches(preds, coords, image_size, patch_size)
        pred_logits_full = pred_logits_full[:original_size[0], :original_size[1]].numpy()
        pred_mask = (pred_logits_full * 255).astype(np.uint8)
        pred_logits = pred_logits_full  # Save for metric evaluation

    if os.path.exists(tmp_root):
        shutil.rmtree(tmp_root)

    return pred_mask, pred_logits

def overlay_mask(image_gray, mask, color=(255, 0, 0), alpha=0.5):
    image_color = cv2.cvtColor(image_gray, cv2.COLOR_GRAY2RGB)
    mask_color = np.zeros_like(image_color)
    mask_color[:, :, 0] = color[0]
    mask_color[:, :, 1] = color[1]
    mask_color[:, :, 2] = color[2]

    mask_bool = mask > 127
    overlayed = np.where(mask_bool[..., None],
                         (1 - alpha) * image_color + alpha * mask_color,
                         image_color).astype(np.uint8)
    return overlayed

def overlay_two_masks(image_gray, mask1, mask2, color1=(255, 0, 0), color2=(0, 255, 0), alpha=0.5):
    image_color = cv2.cvtColor(image_gray, cv2.COLOR_GRAY2RGB)
    overlay = image_color.copy()

    overlay[mask1 > 127] = ((1 - alpha) * overlay[mask1 > 127] + alpha * np.array(color1)).astype(np.uint8)
    overlay[mask2 > 127] = ((1 - alpha) * overlay[mask2 > 127] + alpha * np.array(color2)).astype(np.uint8)

    return overlay

# Load all metadata once
model_registry = load_all_models_metadata()

# Select model
selected_key = st.selectbox("Select a trained model", list(model_registry.keys()))
selected_meta = model_registry[selected_key]["metadata"]
model = load_model_from_metadata(selected_meta, model_registry[selected_key]["path"])

# Display model details
st.markdown("### 📄 Model Details")
st.text(f"Model: {selected_meta.get('model_name')}")
st.text(f"Encoder: {selected_meta.get('encoder')}")
st.text(f"Pretrained Weights: {selected_meta.get('encoder_wights')}")
st.text(f"Loss Function: {selected_meta.get('loss_function')}")
st.text(f"Topographic Weighting: {selected_meta.get('topographic_weighting')}")
if selected_meta.get("combine_with"):
    st.text(f"Combined With: {selected_meta.get('combine_with')}")
st.text(f"Best Dice Score: {selected_meta.get('best_dice'):.4f} (Epoch {selected_meta.get('best_epoch')})")
st.text(f"Timestamp: {selected_meta.get('timestamp')}")

# Upload image
uploaded_file = st.file_uploader("Upload a grayscale image", type=["png", "jpg", "tif", "tiff"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("L")
    image_np = np.array(image)

    st.subheader("Original Image")
    st.image(image_np, use_container_width=True, clamp=True)

    clahe_preview = apply_clahe(image_np)
    st.subheader("CLAHE Preprocessed (Preview)")
    st.image(clahe_preview, use_container_width=True, clamp=True)

    pred_mask, pred_logits = predict_with_dataset(model, image)

    # --- Threshold slider ---
    st.subheader("Predicted Mask (Thresholded)")
    if 'threshold' not in st.session_state:
        st.session_state.threshold = 0.5

    col1, col2 = st.columns([3, 1])
    with col1:
        st.session_state.threshold = st.slider("Threshold", min_value=0.0, max_value=1.0, value=st.session_state.threshold, step=0.01, key="threshold_slider")

    with col2:
        auto_button = st.button("Auto-compute best threshold")
    
    st.markdown(
        "<small>ℹ️ <i>Auto-compute threshold is only available if a ground truth mask is uploaded.</i></small>",
        unsafe_allow_html=True
    )

    # Apply threshold to logits
    pred_mask_thresh = (pred_logits > st.session_state.threshold).astype(np.uint8) * 255
    st.image(pred_mask_thresh, use_container_width=True, clamp=True)

    # Overlay the thresholded mask
    st.subheader("Overlay: Thresholded Predicted Mask on Original Image")
    overlay_pred = overlay_mask(image_np, pred_mask_thresh)
    st.image(overlay_pred, use_container_width=True)

    # Allow downloading thresholded mask
    result_img = Image.fromarray(pred_mask_thresh.astype(np.uint8), mode='L')
    st.download_button("Download Thresholded Mask", result_img.tobytes(), file_name="segmented_thresholded.png")

    # Optionally upload ground truth
    gt_mask_file = st.file_uploader("Optionally upload a ground truth mask", type=["png", "jpg", "tif", "tiff"])
    if gt_mask_file is not None:
        gt_mask = Image.open(gt_mask_file).convert("L")
        gt_mask_np = np.array(gt_mask)

        st.subheader("Ground Truth Mask")
        st.image(gt_mask_np, use_container_width=True, clamp=True)

        st.subheader("Overlay: Predicted (Red) vs Ground Truth (Green)")
        overlay_combined = overlay_two_masks(image_np, pred_mask, gt_mask_np)
        st.image(overlay_combined, use_container_width=True)

        # --- Compute Metrics ---
        st.subheader("📊 Segmentation Metrics")
        pred_tensor = torch.tensor(pred_logits > st.session_state.threshold).unsqueeze(0).unsqueeze(0).float()
        gt_tensor = torch.tensor(gt_mask_np / 255.0).unsqueeze(0).unsqueeze(0).float()  # (1,1,H,W)

        dice = compute_dice(pred_tensor, gt_tensor)
        iou = compute_iou(pred_tensor, gt_tensor)

        st.markdown(f"- **Dice Score:** `{dice:.4f}`")
        st.markdown(f"- **IoU (Jaccard):** `{iou:.4f}`")

        if auto_button:
            best_thresh, best_dice = find_best_threshold(pred_logits, gt_mask_np)
            st.session_state.threshold = float(np.round(best_thresh, 3))
            st.success(f"Best threshold found: {best_thresh:.3f} with Dice: {best_dice:.4f}")
            st.rerun()
