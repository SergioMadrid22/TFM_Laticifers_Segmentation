import os
import json
import torch
from datetime import datetime

def patchify(image, patch_size, stride):
    """
    Extract patches from a 2D image.

    Args:
        image (np.ndarray): 2D image (H, W) or 3D (C, H, W).
        patch_size (int or tuple): size of each patch (patch_h, patch_w)
        stride (int or tuple): stride between patches (stride_h, stride_w)

    Returns:
        patches (List[np.ndarray]): list of patches
        coords (List[Tuple[int, int]]): top-left (y, x) coordinates of each patch
    """
    if isinstance(patch_size, int):
        patch_size = (patch_size, patch_size)
    if isinstance(stride, int):
        stride = (stride, stride)

    if image.ndim == 2:
        H, W = image.shape
        get_patch = lambda y, x: image[y:y+patch_size[0], x:x+patch_size[1]]
    elif image.ndim == 3:
        C, H, W = image.shape
        get_patch = lambda y, x: image[:, y:y+patch_size[0], x:x+patch_size[1]]
    else:
        raise ValueError("Image must be 2D or 3D (C, H, W)")

    patches = []
    coords = []

    for y in range(0, H - patch_size[0] + 1, stride[0]):
        for x in range(0, W - patch_size[1] + 1, stride[1]):
            patch = get_patch(y, x)
            if patch.shape[-2:] == patch_size:
                patches.append(patch)
                coords.append((y, x))

    return patches, coords


def save_metadata(save_path, model, conf, best_dice, best_cldice, best_val_loss, best_epoch):
    metadata = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model_name": conf['model']['name'],
        "experiment_name": conf['train']['experiment_name'],
        "encoder": conf['model'].get('encoder_name', 'N/A'),
        "encoder_weights": conf['model'].get('encoder_weights', 'N/A'),
        "best_dice": best_dice,
        "best_cldice": best_cldice,
        "best_val_loss": best_val_loss,
        "best_epoch": best_epoch,
        "loss_function": conf['loss']['name'],
        "topographic_weighting": conf['loss'].get('use_topographic', False),
        "combine_with": conf['loss'].get('combine_with', None),
        "main_weight": conf['loss']['weights']['main'],
        "combined_weight": conf['loss']['weights']['combined'],
        "topo_alpha": conf['loss'].get('topo', {}).get('alpha', None),
        "topo_beta": conf['loss'].get('topo', {}).get('beta', None),
        "train_params": conf['train'],
    }

    with open(os.path.join(save_path, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=4)


def reconstruct_from_patches2(patches, coords, image_size, patch_size):
    H, W = map(int, image_size)
    full_prob = torch.zeros((H, W), device=patches.device)
    count = torch.zeros((H, W), device=patches.device)
    for patch, (top, left) in zip(patches, coords):
        full_prob[top:top+patch_size[0], left:left+patch_size[1]] += patch.squeeze(0)
        count[top:top+patch_size[0], left:left+patch_size[1]] += 1
    return (full_prob / count.clamp(min=1e-7)).clamp(0, 1)


def reconstruct_from_patches(patches, coords, image_size, patch_size):
    H, W = map(int, image_size)
    full_prob = torch.zeros((H, W), device=patches.device)
    count = torch.zeros((H, W), device=patches.device)

    for patch, (top, left) in zip(patches, coords):
        patch_2d = patch.squeeze(0)  # shape (H_patch, W_patch)
        ph, pw = patch_2d.shape[-2], patch_2d.shape[-1]

        # Calculate actual slice sizes (in case patch goes out of image boundaries)
        ph_eff = min(ph, H - top)
        pw_eff = min(pw, W - left)

        # Crop patch accordingly
        patch_cropped = patch_2d[..., :ph_eff, :pw_eff]

        # Add patch to full_prob
        full_prob[top:top+ph_eff, left:left+pw_eff] += patch_cropped
        count[top:top+ph_eff, left:left+pw_eff] += 1

    return (full_prob / count.clamp(min=1e-7)).clamp(0, 1)


def extract_patches(image_tensor, patch_size=(512, 512), stride=(256, 256)):
    """
    Extracts patches from a 2D image tensor.

    Args:
        image_tensor: torch.Tensor of shape (1, 1, H, W)
        patch_size: tuple (patch_height, patch_width)
        stride: tuple (stride_height, stride_width)

    Returns:
        patches: Tensor of shape (N, 1, patch_h, patch_w)
        coords: list of (y, x) coordinates where each patch was taken from
    """
    _, _, H, W = image_tensor.shape
    ph, pw = patch_size
    sh, sw = stride

    # Pad image if needed to fit patching
    pad_h = (ph - H % ph) % ph if H % ph != 0 else 0
    pad_w = (pw - W % pw) % pw if W % pw != 0 else 0

    padded = torch.nn.functional.pad(image_tensor, (0, pad_w, 0, pad_h), mode='reflect')
    _, _, H_pad, W_pad = padded.shape

    patches = []
    coords = []

    for y in range(0, H_pad - ph + 1, sh):
        for x in range(0, W_pad - pw + 1, sw):
            patch = padded[:, :, y:y + ph, x:x + pw]
            patches.append(patch)
            coords.append((y, x))

    patches = torch.cat(patches, dim=0)  # Concatenate along batch dimension
    return patches.unsqueeze(1), coords  # shape: (N, 1, ph, pw), coords: list of (y, x)

