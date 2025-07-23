import torch.nn.functional as F
import numpy as np
from scipy.ndimage import distance_transform_edt
import numpy as np
from scipy.spatial.distance import directed_hausdorff
import cv2
import torch
from clDice.cldice_metric.cldice import clDice as compute_cldice

# METRICS
def compute_iou(preds, targets, threshold=0.5, eps=1e-7):
    preds = (preds > threshold).float()
    intersection = (preds * targets).sum(dim=(1, 2, 3))
    union = preds.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3)) - intersection
    iou = (intersection + eps) / (union + eps)
    return iou.mean().item()

def compute_dice(preds, targets, threshold=0.5, eps=1e-7):
    preds = (preds > threshold).float()
    intersection = (preds * targets).sum(dim=(1, 2, 3))
    union = preds.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3))
    dice = (2. * intersection + eps) / (union + eps)
    return dice.mean().item()


def compute_tversky(preds, targets, alpha=0.5, beta=0.5, threshold=0.5, eps=1e-7):
    preds = (preds > threshold).float()
    targets = targets.float()

    tp = (preds * targets).sum(dim=(1, 2, 3))
    fn = ((1 - preds) * targets).sum(dim=(1, 2, 3))
    fp = (preds * (1 - targets)).sum(dim=(1, 2, 3))

    tversky = (tp + eps) / (tp + alpha * fp + beta * fn + eps)
    return tversky.mean().item()

def compute_hausdorff(preds, targets, threshold=0.5):
    """
    Compute the average symmetric Hausdorff distance for a batch.
    Assumes preds and targets are tensors of shape (B, 1, H, W).
    """
    preds = (preds > threshold).cpu().numpy().astype(np.uint8)
    targets = targets.cpu().numpy().astype(np.uint8)

    batch_size = preds.shape[0]
    distances = []

    for i in range(batch_size):
        pred_coords = np.argwhere(preds[i, 0])
        target_coords = np.argwhere(targets[i, 0])

        if len(pred_coords) == 0 or len(target_coords) == 0:
            distances.append(np.nan)  # Undefined for empty mask
            continue

        hd_forward = directed_hausdorff(pred_coords, target_coords)[0]
        hd_backward = directed_hausdorff(target_coords, pred_coords)[0]
        symmetric_hd = max(hd_forward, hd_backward)

        distances.append(symmetric_hd)

    return np.nanmean(distances)

def compute_hd95(preds, targets, threshold=0.5):
    """
    Compute the average symmetric 95th percentile Hausdorff distance (HD95) for a batch.
    """
    preds = (preds > threshold).cpu().numpy().astype(np.uint8)
    targets = targets.cpu().numpy().astype(np.uint8)

    batch_size = preds.shape[0]
    distances = []

    for i in range(batch_size):
        pred = preds[i, 0]
        target = targets[i, 0]

        if pred.sum() == 0 or target.sum() == 0:
            distances.append(np.nan)
            continue

        # Get edges
        pred_edges = pred - cv2.erode(pred, None)
        target_edges = target - cv2.erode(target, None)

        # Distance transforms
        dt_pred = distance_transform_edt(1 - pred_edges)
        dt_target = distance_transform_edt(1 - target_edges)

        # Surface distances
        surface_distances_pred_to_target = dt_target[pred_edges == 1]
        surface_distances_target_to_pred = dt_pred[target_edges == 1]

        if len(surface_distances_pred_to_target) == 0 or len(surface_distances_target_to_pred) == 0:
            distances.append(np.nan)
            continue

        hd95 = np.percentile(np.hstack((surface_distances_pred_to_target, surface_distances_target_to_pred)), 95)
        distances.append(hd95)

    return np.nanmean(distances)

def compute_metrics(preds, targets, threshold=0.5, alpha=0.5, beta=0.5):
    """
    Computes multiple segmentation metrics in one call.

    Args:
        preds (torch.Tensor): Predicted masks of shape (B, 1, H, W).
        targets (torch.Tensor): Ground truth masks of shape (B, 1, H, W).
        threshold (float): Threshold to binarize predictions.
        alpha (float): Tversky loss alpha parameter.
        beta (float): Tversky loss beta parameter.

    Returns:
        dict: Dictionary with computed metrics.
    """
    metrics = {}

    metrics['Dice'] = compute_dice(preds, targets, threshold)
    metrics['clDice'] = compute_cldice(preds.squeeze(), targets.squeeze(), threshold=threshold)
    metrics['IoU'] = compute_iou(preds, targets, threshold)
    metrics['Tversky'] = compute_tversky(preds, targets, alpha, beta, threshold)
    metrics['Hausdorff'] = compute_hausdorff(preds, targets, threshold)
    metrics['HD95'] = compute_hd95(preds, targets, threshold)

    return metrics

# LOSSES
def compute_soft_dice(pred, target, eps=1e-7):
    pred = pred.view(-1)
    target = target.view(-1)
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum()
    return (2. * intersection + eps) / (union + eps)

def dice_loss(pred, target, eps=1e-7):
    dice = compute_soft_dice(pred, target, eps)
    return 1 - dice

def compute_soft_tversky(pred, target, alpha=0.5, beta=0.5, eps=1e-7):
    pred = pred.view(-1)
    target = target.view(-1)

    tp = (pred * target).sum()
    fn = ((1 - pred) * target).sum()
    fp = (pred * (1 - target)).sum()

    return (tp + eps) / (tp + alpha * fp + beta * fn + eps)

def tversky_loss(pred, target, alpha=0.5, beta=0.5, eps=1e-7):
    tversky = compute_soft_tversky(pred, target, alpha, beta, eps)
    return 1 - tversky


def focal_tversky_loss(pred, target, alpha=0.5, beta=0.5, gamma=0.75, eps=1e-7):
    """Focal Tversky Loss: penalizes harder-to-segment pixels more."""
    tversky = compute_soft_tversky(pred, target, alpha, beta, eps)
    return (1 - tversky) ** gamma


def dice_loss(pred, target, eps=1e-7):
    return 1 - compute_soft_dice(pred, target, eps)


def compute_soft_dice(pred, target, eps=1e-7):
    pred = pred.view(-1)
    target = target.view(-1)
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum()
    return (2. * intersection + eps) / (union + eps)


def compute_tversky(preds, targets, alpha=0.5, beta=0.5, threshold=0.5, eps=1e-7):
    preds = (preds > threshold).float()
    targets = targets.float()

    tp = (preds * targets).sum(dim=(1, 2, 3))
    fn = ((1 - preds) * targets).sum(dim=(1, 2, 3))
    fp = (preds * (1 - targets)).sum(dim=(1, 2, 3))

    tversky = (tp + eps) / (tp + alpha * fp + beta * fn + eps)
    return tversky.mean().item()


def compute_soft_tversky(pred, target, alpha=0.5, beta=0.5, eps=1e-7):
    pred = pred.view(-1)
    target = target.view(-1)

    tp = (pred * target).sum()
    fn = ((1 - pred) * target).sum()
    fp = (pred * (1 - target)).sum()

    return (tp + eps) / (tp + alpha * fp + beta * fn + eps)


def tversky_loss(pred, target, alpha=0.5, beta=0.5, eps=1e-7):
    return 1 - compute_soft_tversky(pred, target, alpha, beta, eps)


def focal_tversky_loss(pred, target, alpha=0.5, beta=0.5, gamma=0.75, eps=1e-7):
    tversky = compute_soft_tversky(pred, target, alpha, beta, eps)
    return (1 - tversky) ** gamma


def topographic_loss(preds, targets, distance_maps, base_loss='bce', alpha=2.0, beta=1.0, eps=1e-7):
    """
    Distance-weighted pixel-wise loss (aka Topographic Loss).

    Args:
        preds: (B, 1, H, W) - predicted probabilities.
        targets: (B, 1, H, W) - binary ground truth masks.
        distance_maps: (B, 1, H, W) - precomputed distance maps.
        base_loss: 'bce', 'dice', or 'focal_tversky'.
        alpha: power factor for distance weighting.
        beta: global scaling multiplier for the loss.
        eps: small constant for numerical stability.

    Returns:
        Scalar loss value.
    """
    # Normalize distance map to [0, 1] per sample
    dist_norm = distance_maps / (distance_maps.amax(dim=(2, 3), keepdim=True) + eps)
    weights = (1.0 + dist_norm) ** alpha

    if base_loss == 'bce':
        loss_map = F.binary_cross_entropy(preds, targets, reduction='none')

    elif base_loss == 'dice':
        intersection = preds * targets
        loss_map = 1 - (2. * intersection + eps) / (preds + targets + eps)

    elif base_loss == 'focal_tversky':
        alpha_t, beta_t, gamma = 0.7, 0.3, 0.75
        tp = (preds * targets).sum(dim=(2, 3))
        fp = ((1 - targets) * preds).sum(dim=(2, 3))
        fn = (targets * (1 - preds)).sum(dim=(2, 3))
        tversky = (tp + eps) / (tp + alpha_t * fp + beta_t * fn + eps)
        return (1 - tversky).pow(gamma).mean()

    else:
        raise ValueError(f"Unsupported base_loss: {base_loss}")

    weighted_loss = weights * loss_map
    return beta * weighted_loss.mean()


def combined_topo_tversky_dice(
    preds, 
    targets, 
    distance_maps, 
    topo_alpha=2.0,      # exponent on distance weighting
    tv_alpha=0.7,        # Tversky false‐positive weight
    tv_beta=0.3,         # Tversky false‐negative weight
    tv_gamma=0.75,       # Focusing parameter for focal Tversky
    dice_weight=0.5,     # weight for plain Dice term
    beta=1.0,            # global scaling for the topo loss
    eps=1e-7
):
    """
    Combined topographic‐weighted focal‐Tversky + Dice loss.

    Assumes:
    - `distance_maps` are already normalized to [0, 1].
    - `preds` and `targets` are (B, 1, H, W) with values in [0, 1].
    """

    # 1) compute distance-based weighting
    weight_map = (1.0 + distance_maps) ** topo_alpha  # (B, 1, H, W)

    # 2) compute soft Tversky
    tv = compute_soft_tversky(preds, targets, alpha=tv_alpha, beta=tv_beta, eps=eps)
    focal_tv = (1 - tv).pow(tv_gamma)  # scalar

    # 3) topographic component (scalar * map, then mean)
    topo_focal_tv = beta * (weight_map * focal_tv.unsqueeze(-1).unsqueeze(-1)).mean()

    # 4) Dice loss
    dice_l = 1 - compute_soft_dice(preds, targets, eps)

    # 5) Combine
    loss = topo_focal_tv + dice_weight * dice_l
    return loss

import torch.nn.functional as F

def bce_dice_topographic_loss(preds, targets, distance_maps, alpha=2.0, dice_weight=0.5, beta=1.0, eps=1e-7):
    """
    Combines:
    - BCE with distance weighting (topographic loss)
    - Soft Dice loss

    Args:
        preds: (B, 1, H, W) - predicted probabilities
        targets: (B, 1, H, W) - binary ground truth
        distance_maps: (B, 1, H, W) - precomputed distance maps (not normalized)
        alpha: exponent for distance weighting
        dice_weight: scaling factor for dice component
        beta: global scaling for BCE part
        eps: small number for numerical stability

    Returns:
        Scalar loss (float)
    """

    # Normalize distance maps to [0, 1] per sample
    dist_norm = distance_maps / (distance_maps.amax(dim=(2, 3), keepdim=True) + eps)
    weights = (1.0 + dist_norm) ** alpha  # (B, 1, H, W)

    # Compute BCE with distance weighting
    bce_map = F.binary_cross_entropy(preds, targets, reduction='none')  # (B, 1, H, W)
    weighted_bce = weights * bce_map
    topo_bce = beta * weighted_bce.mean()

    # Compute soft Dice loss
    dice_l = dice_loss(preds, targets, eps=eps)

    # Combine
    loss = topo_bce + dice_weight * dice_l
    return loss


def hausdorff_loss(preds, targets, eps=1e-7):
    """
    Compute a differentiable approximation of Hausdorff Distance Loss.
    Based on: https://arxiv.org/abs/1904.10030 (Karimi et al.)
    """
    if not (preds.shape == targets.shape):
        raise ValueError("Shape mismatch between preds and targets")

    preds = preds.squeeze(1)  # (B, H, W)
    targets = targets.squeeze(1)

    loss = 0.0
    batch_size = preds.shape[0]

    for i in range(batch_size):
        pred = preds[i]
        target = targets[i]

        pred_np = pred.detach().cpu().numpy()
        target_np = target.detach().cpu().numpy()

        # Distance transforms on boundaries
        dist_target = torch.tensor(distance_transform_edt(1 - target_np), device=pred.device, dtype=pred.dtype)
        dist_pred = torch.tensor(distance_transform_edt(1 - pred_np), device=pred.device, dtype=pred.dtype)

        # Hausdorff loss from both directions
        term_1 = pred * dist_target
        term_2 = target * dist_pred

        loss += (term_1.mean() + term_2.mean()) / 2

    return loss / batch_size
