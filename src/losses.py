import torch
from scipy.ndimage import distance_transform_edt
import torch.nn.functional as F
from clDice.cldice_loss.cldice import soft_cldice, soft_dice_cldice
import torch.nn as nn

def masked_mse_loss(prediction, original_image, mask):
    """
    Computes the Mean Squared Error (MSE) loss only on the regions that were
    masked out during the Unet_MIM's forward pass.
    
    Args:
        prediction (torch.Tensor): The autoencoder's reconstructed output.
        original_image (torch.Tensor): The original, unmasked input image.
        mask (torch.Tensor): The binary mask used for masking (1 where pixels were removed).
    """
    # The loss is the squared difference, but only where the mask is 1.
    # We multiply by the mask to zero out the loss for the unmasked regions.
    loss = F.mse_loss(prediction * mask, original_image * mask, reduction='sum')
    
    # We normalize the loss by the number of masked pixels to get a stable mean.
    # Add a small epsilon to avoid division by zero if a mask is all zeros.
    num_masked_pixels = mask.sum() + 1e-7
    
    return loss / num_masked_pixels

def compute_soft_dice(pred, target, eps=1e-7):
    pred, target = pred.reshape(-1), target.reshape(-1)
    inter = (pred * target).sum()
    return (2. * inter + eps) / (pred.sum() + target.sum() + eps)


def dice_loss(pred, target, eps=1e-7):
    return 1 - compute_soft_dice(pred, target, eps)


def compute_soft_tversky(pred, target, alpha=0.5, beta=0.5, eps=1e-7):
    pred, target = pred.reshape(-1), target.reshape(-1)
    tp = (pred * target).sum()
    fn = ((1 - pred) * target).sum()
    fp = (pred * (1 - target)).sum()
    return (tp + eps) / (tp + alpha * fp + beta * fn + eps)


def tversky_loss(pred, target, alpha=0.5, beta=0.5, eps=1e-7):
    return 1 - compute_soft_tversky(pred, target, alpha, beta, eps)


def focal_tversky_loss(pred, target, alpha=0.5, beta=0.5, gamma=0.75, eps=1e-7):
    tv = compute_soft_tversky(pred, target, alpha, beta, eps)
    return (1 - tv).pow(gamma)


def hausdorff_loss(preds, targets, eps=1e-7):
    """Approximate differentiable Hausdorff loss (Karimi et al. 2019)."""
    preds, targets = preds.squeeze(1), targets.squeeze(1)
    loss = 0.0
    for pred, target in zip(preds, targets):
        pred_np, target_np = pred.detach().cpu().numpy(), target.detach().cpu().numpy()
        dist_t = torch.tensor(distance_transform_edt(1 - target_np), device=pred.device, dtype=pred.dtype)
        dist_p = torch.tensor(distance_transform_edt(1 - pred_np), device=pred.device, dtype=pred.dtype)
        loss += (pred * dist_t).mean() + (target * dist_p).mean()
    return loss / (2 * preds.shape[0])


def weighted_tversky_loss(pred, target, weight_map, alpha=0.5, beta=0.5, eps=1e-7):
    """
    Computes a Tversky loss where the False Positive and False Negative terms
    are weighted by a distance map. This penalizes errors in critical regions more.
    """
    # Reshape to 1D tensors
    pred = pred.reshape(-1)
    target = target.reshape(-1)
    weight_map = weight_map.reshape(-1)

    # True Positives are not weighted
    tp = (pred * target).sum()
    
    # False Positives and False Negatives are weighted
    fp_map = pred * (1 - target)
    fn_map = (1 - pred) * target
    
    weighted_fp = (fp_map * weight_map).sum()
    weighted_fn = (fn_map * weight_map).sum()

    tversky_index = (tp + eps) / (tp + alpha * weighted_fp + beta * weighted_fn + eps)
    return 1 - tversky_index


def weighted_dice_loss(pred, target, weight_map, eps=1e-7):
    """ A special case of weighted Tversky loss with alpha=0.5 and beta=0.5. """
    return weighted_tversky_loss(pred, target, weight_map, alpha=0.5, beta=0.5, eps=eps)


def weighted_focal_tversky_loss(pred, target, weight_map, alpha=0.5, beta=0.5, gamma=0.75, eps=1e-7):
    """ A focal version of the weighted Tversky loss. """
    tversky_loss_val = weighted_tversky_loss(pred, target, weight_map, alpha, beta, eps)
    return tversky_loss_val ** gamma

def weighted_bce_loss(preds, targets, weight_map, beta=1.0):
    """
    Computes BCE loss with a pixel-wise weighting.
    Uses the numerically stable `with_logits` version.
    
    Args:
        preds: (B, C, H, W) - Raw logits from the model.
        targets: (B, C, H, W) - Ground truth masks.
        weight_map: (B, C, H, W) - The map of pixel-wise weights.
        beta: (float) - A global scaling factor for the loss.
    """
    # Get a pixel-wise loss map without reduction
    bce_map = F.binary_cross_entropy_with_logits(preds, targets, reduction='none')
    
    # Apply the weights and compute the mean
    weighted_bce = weight_map * bce_map
    return beta * weighted_bce.mean()


# ------------------------
# FACTORY
# ------------------------
def get_loss_function(conf):
    """
    Constructs a loss function based on a configuration dictionary.
    Now includes a case for the 'masked_mse' loss for pre-training.
    """
    name = conf['loss']['name'].lower()
    
    # --- NEW: Special case for the Deep Closing pre-training loss ---
    if name == 'masked_mse':
        # This loss is simple and doesn't need combinations or weighting
        return masked_mse_loss

    # --- (The rest of the factory function remains the same) ---
    use_topo = conf['loss'].get('use_topographic', False)
    combine_name = conf['loss'].get('combine_with', None)
    
    w_main = conf['loss']['weights'].get('main', 1.0)
    w_comb = conf['loss']['weights'].get('combined', 0.0)
    
    topo_a = conf['loss']['topo'].get('alpha', 2.0)
    topo_b = conf['loss']['topo'].get('beta', 1.0)
    
    cldice_a = conf['loss'].get('cldice_alpha', 0.5)
    tversky_a = conf['loss'].get('tversky_alpha', 0.3)
    tversky_b = conf['loss'].get('tversky_beta', 0.7)
    focal_tversky_gamma = conf['loss'].get('focal_tversky_gamma', 0.75)
    eps = 1e-7

    loss_constructors = {
        'bce': lambda: nn.BCEWithLogitsLoss(),
        'dice': lambda: lambda p, t: dice_loss(p.sigmoid(), t, eps=eps),
        'tversky': lambda: lambda p, t: tversky_loss(p.sigmoid(), t, alpha=tversky_a, beta=tversky_b, eps=eps),
        'focal_tversky': lambda: lambda p, t: focal_tversky_loss(p.sigmoid(), t, alpha=tversky_a, beta=tversky_b, gamma=focal_tversky_gamma, eps=eps),
        'hausdorff': lambda: lambda p, t: hausdorff_loss(p.sigmoid(), t, eps=eps),
        'cldice': lambda: lambda p, t: soft_cldice()(t, p.sigmoid()),
        'dice_cldice': lambda: lambda p, t: soft_dice_cldice(alpha=cldice_a)(t, p.sigmoid())
    }

    weighted_loss_constructors = {
        # --- CORRECTION: Pass raw logits 'p' to weighted_bce_loss ---
        'bce': lambda: lambda p, t, w: weighted_bce_loss(p, t, w, beta=topo_b),
        'dice': lambda: lambda p, t, w: weighted_dice_loss(p.sigmoid(), t, w, eps=eps),
        'tversky': lambda: lambda p, t, w: weighted_tversky_loss(p.sigmoid(), t, w, alpha=tversky_a, beta=tversky_b, eps=eps),
        'focal_tversky': lambda: lambda p, t, w: weighted_focal_tversky_loss(p.sigmoid(), t, w, alpha=tversky_a, beta=tversky_b, gamma=focal_tversky_gamma, eps=eps),
    }

    def final_loss_fn(preds, targets, **kwargs):
        dist = kwargs.get('dist', None) # Accept optional distance map
        
        is_weighted = use_topo and name in weighted_loss_constructors
        constructors = weighted_loss_constructors if is_weighted else loss_constructors

        if name not in constructors:
            raise ValueError(f"Loss '{name}' not supported.")
        main_loss = constructors[name]()

        combined_loss = None
        if combine_name and w_comb > 0:
            if combine_name not in loss_constructors:
                 raise ValueError(f"Combined loss '{combine_name}' not supported.")
            combined_loss = loss_constructors[combine_name]()

        weight_map = None
        if is_weighted:
            if dist is None:
                raise ValueError(f"Distance map must be provided for topographic loss '{name}'.")
            dist_norm = dist / (dist.amax(dim=(2, 3), keepdim=True) + eps)
            weight_map = (1.0 + dist_norm) ** topo_a

        if is_weighted:
            loss = w_main * main_loss(preds, targets, weight_map)
        else:
            loss = w_main * main_loss(preds, targets)

        if combined_loss:
            loss += w_comb * combined_loss(preds, targets)
            
        return loss

    return final_loss_fn