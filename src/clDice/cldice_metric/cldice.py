from skimage.morphology import skeletonize
import numpy as np
import torch

def cl_score(v, s):
    """Compute skeleton volume overlap (precision/sensitivity)."""
    return np.sum(v & s) / (np.sum(s) + 1e-8)  # Add epsilon to avoid div by zero

def clDice(v_p, v_l, threshold=0.5):
    """
    Computes clDice metric after thresholding predictions and ground truth.

    Args:
        v_p (ndarray or Tensor): Predicted image
        v_l (ndarray or Tensor): Ground truth image
        threshold (float): Threshold to binarize images

    Returns:
        float: clDice score ∈ [0, 1]
    """
    # Convert PyTorch tensors to numpy arrays if necessary
    if isinstance(v_p, torch.Tensor):
        v_p = v_p.detach().cpu().numpy()
    if isinstance(v_l, torch.Tensor):
        v_l = v_l.detach().cpu().numpy()

    # Binarize inputs if not already bool
    if not v_p.dtype == bool:
        v_p = v_p >= threshold
    if not v_l.dtype == bool:
        v_l = v_l >= threshold

    if v_p.ndim == 2:
        skel_p = skeletonize(v_p)
        skel_l = skeletonize(v_l)
    else:
        raise ValueError("Input masks must be 2D arrays.")

    tprec = cl_score(v_p, skel_l)
    tsens = cl_score(v_l, skel_p)

    return (2 * tprec * tsens) / (tprec + tsens + 1e-8)
