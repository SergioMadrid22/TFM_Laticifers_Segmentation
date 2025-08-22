import torch
import torch.nn.functional as F
import numpy as np
import cv2
from scipy.ndimage import distance_transform_edt
from scipy.spatial.distance import directed_hausdorff
from losses import compute_soft_dice
from clDice.cldice_metric.cldice import clDice as compute_cldice
from clDice.cldice_loss.cldice import soft_cldice


def compute_iou(preds, targets, threshold=0.5, eps=1e-7):
    preds = (preds > threshold).float()
    intersection = (preds * targets).sum(dim=(1, 2, 3))
    union = preds.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3)) - intersection
    return ((intersection + eps) / (union + eps)).mean().item()


def compute_dice(preds, targets, threshold=0.5, eps=1e-7):
    preds = (preds > threshold).float()
    intersection = (preds * targets).sum(dim=(1, 2, 3))
    union = preds.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3))
    return ((2. * intersection + eps) / (union + eps)).mean().item()


def compute_tversky(preds, targets, alpha=0.5, beta=0.5, threshold=0.5, eps=1e-7):
    preds = (preds > threshold).float()
    targets = targets.float()
    tp = (preds * targets).sum(dim=(1, 2, 3))
    fn = ((1 - preds) * targets).sum(dim=(1, 2, 3))
    fp = (preds * (1 - targets)).sum(dim=(1, 2, 3))
    return ((tp + eps) / (tp + alpha * fp + beta * fn + eps)).mean().item()


def compute_hausdorff(preds, targets, threshold=0.5):
    """Symmetric Hausdorff Distance (non-differentiable)."""
    preds = (preds > threshold).cpu().numpy().astype(np.uint8)
    targets = targets.cpu().numpy().astype(np.uint8)

    distances = []
    for i in range(preds.shape[0]):
        pred_coords = np.argwhere(preds[i, 0])
        target_coords = np.argwhere(targets[i, 0])
        if len(pred_coords) == 0 or len(target_coords) == 0:
            distances.append(np.nan)
            continue
        hd_forward = directed_hausdorff(pred_coords, target_coords)[0]
        hd_backward = directed_hausdorff(target_coords, pred_coords)[0]
        distances.append(max(hd_forward, hd_backward))
    return np.nanmean(distances)


def compute_hd95(preds, targets, threshold=0.5):
    """Symmetric 95th percentile Hausdorff Distance (HD95)."""
    preds = (preds > threshold).cpu().numpy().astype(np.uint8)
    targets = targets.cpu().numpy().astype(np.uint8)

    distances = []
    for i in range(preds.shape[0]):
        pred, target = preds[i, 0], targets[i, 0]
        if pred.sum() == 0 or target.sum() == 0:
            distances.append(np.nan)
            continue

        # Edges
        pred_edges = pred - cv2.erode(pred, None)
        target_edges = target - cv2.erode(target, None)

        dt_pred = distance_transform_edt(1 - pred_edges)
        dt_target = distance_transform_edt(1 - target_edges)

        surf_pred_to_target = dt_target[pred_edges == 1]
        surf_target_to_pred = dt_pred[target_edges == 1]

        if len(surf_pred_to_target) == 0 or len(surf_target_to_pred) == 0:
            distances.append(np.nan)
            continue

        hd95 = np.percentile(np.hstack((surf_pred_to_target, surf_target_to_pred)), 95)
        distances.append(hd95)

    return np.nanmean(distances)


def to_float(val):
    return float(val.detach().cpu()) if torch.is_tensor(val) else float(val)


def compute_metrics(preds, targets, threshold=0.5, alpha=0.5, beta=0.5):
    preds, targets = preds.detach(), targets.detach()
    return {
        'Dice':       to_float(compute_dice(preds, targets, threshold)),
        'softDice':   to_float(compute_soft_dice(preds, targets)),
        'clDice':     to_float(compute_cldice(preds.squeeze(), targets.squeeze(), threshold)),
        'softclDice': to_float(soft_cldice()(targets, preds)),
        'IoU':        to_float(compute_iou(preds, targets, threshold)),
        'Tversky':    to_float(compute_tversky(preds, targets, alpha, beta, threshold)),
        'Hausdorff':  to_float(compute_hausdorff(preds, targets, threshold)),
        'HD95':       to_float(compute_hd95(preds, targets, threshold)),
    }