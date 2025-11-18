import torch
from scipy.ndimage import distance_transform_edt
import torch.nn.functional as F
from clDice.cldice_loss.cldice import soft_cldice, soft_dice_cldice
import torch.nn as nn
import numpy as np
from scipy.ndimage import distance_transform_edt as dist
import scipy.ndimage as ndimage

import losses_skeletonrecallloss as utils_srl
import losses_cbdice as utils_cbdice
import warnings

import sys
sys.path.append('/data/smadper@alumno.upv.es/TFM/malis') 
import malis

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


class CESkeletonRecallLoss(nn.Module):
    """
    Combines Cross-Entropy with Skeleton Recall Loss.
    Adapted from Kirchhoff et al. (2024) and TopoMortar.
    Prioritizes ensuring the ground truth skeleton is covered by the prediction.
    """
    def __init__(self, lambda_srl=1.0):
        super().__init__()
        self.lambda_srl = lambda_srl
        self.ce_loss_fn = nn.BCEWithLogitsLoss()
        
    def forward(self, preds, targets):
        # Standard BCE Loss
        ce_loss = self.ce_loss_fn(preds, targets)
        
        # Skeleton Recall Loss part
        probs = preds.sigmoid()
        
        # Skeletons are computed from the hard ground truth
        with torch.no_grad():
            skeletons = utils_srl.compute_skeletons(targets).to(targets.device)
        
        # Calculate recall on the skeleton pixels
        intersection = (probs * targets * skeletons).sum()
        gt_skeleton_sum = (targets * skeletons).sum()
        
        recall = (intersection + 1e-7) / (gt_skeleton_sum + 1e-7)
        srl = 1-recall # We want to maximize recall, so we minimize its negative
        
        return ce_loss + self.lambda_srl * srl

class CBDiceLoss(nn.Module):
    """
    Centerline Boundary Dice Loss.
    Adapted from Shi et al. (2024) and TopoMortar.
    Combines CE, Dice, and a term that penalizes the distance between the
    predicted boundary and the true centerline.
    """
    def __init__(self, alpha=1.0, beta=1.0):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.ce_loss_fn = nn.BCEWithLogitsLoss()
        self.dice_loss_fn = dice_loss
        self.t_skeletonize = utils_cbdice.Skeletonize(probabilistic=False, simple_point_detection='EulerCharacteristic')

    def forward(self, preds, targets):
        probs = preds.sigmoid()
        
        # --- Standard CE and Dice losses ---
        ce_loss = self.ce_loss_fn(preds, targets)
        dice = self.dice_loss_fn(probs, targets)

        # --- cbDice specific calculations ---
        pred_prob_fg = probs.squeeze(1) # Remove channel dim
        
        with torch.no_grad():
            target_fg = targets.squeeze(1).float()
            pred_hard_fg = (pred_prob_fg > 0.5).float()
            
            skel_pred_hard = self.t_skeletonize(pred_hard_fg.unsqueeze(1)).squeeze(1)
            skel_true = self.t_skeletonize(target_fg.unsqueeze(1)).squeeze(1)

        skel_pred_prob = skel_pred_hard * pred_prob_fg

        q_vl, q_slvl, q_sl = utils_cbdice.get_weights(target_fg, skel_true, dim=2, prob_flag=False)
        q_vp, q_spvp, q_sp = utils_cbdice.get_weights(pred_prob_fg, skel_pred_prob, dim=2, prob_flag=True)

        w_tprec = (torch.sum(q_sp * q_vl) + 1e-7) / (torch.sum(utils_cbdice.combine_tensors(q_spvp, q_slvl, q_sp)) + 1e-7)
        w_tsens = (torch.sum(q_sl * q_vp) + 1e-7) / (torch.sum(utils_cbdice.combine_tensors(q_slvl, q_spvp, q_sl)) + 1e-7)

        cb_dice = 2.0 * (w_tprec * w_tsens) / (w_tprec + w_tsens)
        cb_dice_loss = 1-cb_dice # Minimize negative cb_dice

        # --- Final combined loss from the paper ---
        w1 = 0.5
        w2 = (self.alpha / (2 * (self.alpha + self.beta)))
        w3 = (self.beta / (2 * (self.alpha + self.beta)))

        return w1 * ce_loss + w2 * dice + w3 * cb_dice_loss


class MSETopoWinLoss(nn.Module):
    """
    TOPO Windowed Loss, adapted from Oner et al. (2021) and the TopoMortar implementation.
    This loss is designed to prevent incorrect merges of parallel structures.
    
    WARNING: This loss is computationally very expensive and requires the 'malis' library.
    """
    def __init__(self, window_size=128, weights=[1.0, 1.0], alpha=1, d_max=20, d_min=10):
        super().__init__()        
        self.window = window_size
        self.weights = weights
        self.alpha = alpha
        self.Dmax = d_max
        self.Dmin = d_min
        self.malis_lr_pos = 0.1

    def forward(self, preds, targets):
        # The original paper's model outputs a distance map, so we use the raw logits.
        # It does not use a sigmoid/softmax.
        preds = torch.relu(preds) # As per original implementation
        
        pred_np = preds.cpu().detach().numpy()
        target_np = targets.cpu().detach().numpy()
        B, C, H, W = pred_np.shape

        # 1. Global MSE component (against a distance map of the target)
        mse_loss = 0
        for b in range(B):
            dist_map = np.clip(dist(target_np[b, 0]), a_min=None, a_max=self.Dmax)
            dist_map_tensor = torch.from_numpy(dist_map).to(preds.device)
            mse_term = (preds[b, 0] - dist_map_tensor)**2
            # Weight by class imbalance
            mse_loss += ((targets[b, 0] * self.weights[1] * mse_term) + ((1 - targets[b, 0]) * self.weights[0] * mse_term)).sum()
        
        num_pixels = targets.numel() # Total number of pixels in the batch
        mse_loss /= num_pixels

        # 2. Windowed Topological (MALIS) component
        weights_n = np.zeros_like(pred_np, dtype=np.float32)
        weights_p = np.zeros_like(pred_np, dtype=np.float32)
        
        for k in range(H // self.window):
            for j in range(W // self.window):
                # ... (This complex logic is copied verbatim from the TopoMortar source) ...
                # It calculates MALIS weights in windows.
                pred_win = pred_np[:,:,k*self.window:(k+1)*self.window, j*self.window:(j+1)*self.window]
                target_win = target_np[:,:,k*self.window:(k+1)*self.window, j*self.window:(j+1)*self.window]

                nodes = np.arange(self.window * self.window).reshape(self.window, self.window)
                edges_h = np.vstack([nodes[:,:-1].ravel(), nodes[:,1:].ravel()])
                edges_v = np.vstack([nodes[:-1,:].ravel(), nodes[1:,:].ravel()])
                edges = np.hstack([edges_h, edges_v]).astype(np.uint64)

                costs_h = (pred_win[:,:,:,:-1] + pred_win[:,:,:,1:]).reshape(B, -1)
                costs_v = (pred_win[:,:,:-1,:] + pred_win[:,:,1:,:]).reshape(B, -1)
                costs = np.hstack([costs_h, costs_v]).astype(np.float32)

                gt_h = (target_win[:,:,:,:-1] + target_win[:,:,:,1:]).reshape(B, -1)
                gt_v = (target_win[:,:,:-1,:] + target_win[:,:,1:,:]).reshape(B, -1)
                gt_costs = np.hstack([gt_h, gt_v]).astype(np.float32)

                costs_n = costs.copy(); costs_p = costs.copy()
                costs_n[gt_costs > self.Dmax] = self.Dmax
                costs_p[gt_costs < self.Dmin] = 0

                for i in range(B):
                    sg_gt = ndimage.label(ndimage.binary_dilation((target_win[i, 0] == 0), iterations=5) == 0)[0]
                    
                    edge_weights_n = malis.malis_loss_weights(sg_gt.astype(np.uint64).flatten(), edges[0], edges[1], costs_n[i], 0).astype(np.float64)
                    edge_weights_p = malis.malis_loss_weights(sg_gt.astype(np.uint64).flatten(), edges[0], edges[1], costs_p[i], 1).astype(np.float64)

                    if np.sum(edge_weights_n) > 0: edge_weights_n /= np.sum(edge_weights_n)
                    if np.sum(edge_weights_p) > 0: edge_weights_p /= np.sum(edge_weights_p)

                    edge_weights_n[gt_costs[i] >= self.Dmin] = 0
                    edge_weights_p[gt_costs[i] < self.Dmax] = 0

                    def get_node_weights(edge_weights):
                        ew_h, ew_v = np.split(edge_weights, 2)
                        ew_h = ew_h.reshape(self.window, self.window - 1)
                        ew_v = ew_v.reshape(self.window - 1, self.window)
                        node_weights = np.zeros((self.window, self.window), dtype=np.float32)
                        node_weights[:, :-1] += ew_h; node_weights[:, 1:] += ew_h
                        node_weights[:-1, :] += ew_v; node_weights[1:, :] += ew_v
                        return node_weights

                    weights_n[i, 0, k*self.window:(k+1)*self.window, j*self.window:(j+1)*self.window] = get_node_weights(edge_weights_n)
                    weights_p[i, 0, k*self.window:(k+1)*self.window, j*self.window:(j+1)*self.window] = get_node_weights(edge_weights_p)

        weights_n = torch.from_numpy(weights_n).to(preds.device)
        weights_p = torch.from_numpy(weights_p).to(preds.device)

        loss_n = preds.pow(2)
        loss_p = (self.Dmax - preds).pow(2)
        topo_loss = (loss_n * weights_n).sum() + (self.malis_lr_pos * loss_p * weights_p).sum()
        topo_loss /= num_pixels
        #print(topo_loss, mse_loss)
        return self.alpha * mse_loss + topo_loss

# ------------------------
# FACTORY
# ------------------------
def get_loss_function(conf):
    """
    Constructs a loss function based on a configuration dictionary.
    Now includes a case for the 'masked_mse' loss for pre-training.
    """
    name = conf['loss']['name'].lower()
    
    # Special case for the Deep Closing pre-training loss ---
    if name == 'masked_mse':
        return masked_mse_loss

    use_topo = conf['loss'].get('use_topographic', False)
    combine_name = conf['loss'].get('combine_with', None)
    
    w_main = conf['loss']['weights'].get('main', 1.0)
    w_comb = conf['loss']['weights'].get('combined', 0.0)

    eps = 1e-7

    # --- Define all possible loss constructors ---
    loss_constructors = {
        'bce': lambda: nn.BCEWithLogitsLoss(),
        'dice': lambda: lambda p, t: dice_loss(p.sigmoid(), t, eps=eps),
        'tversky': lambda: lambda p, t: tversky_loss(p.sigmoid(), t, alpha=conf['loss']['tversky_alpha'], beta=conf['loss']['tversky_beta'], eps=eps),
        'focal_tversky': lambda: lambda p, t: focal_tversky_loss(p.sigmoid(), t, alpha=conf['loss']['tversky_alpha'], beta=conf['loss']['tversky_beta'], gamma=conf['loss']['focal_tversky_gamma'], eps=eps),
        'hausdorff': lambda: lambda p, t: hausdorff_loss(p.sigmoid(), t, eps=eps),
        'cldice': lambda: lambda p, t: soft_cldice()(t, p.sigmoid()),
        'dice_cldice': lambda: lambda p, t: soft_dice_cldice(alpha=conf['loss']['cldice_alpha'])(t, p.sigmoid()),
        'ce_srl': lambda: CESkeletonRecallLoss(lambda_srl=conf['loss'].get('lambda_srl', 1.0)),
        'cbdice': lambda: CBDiceLoss(alpha=conf['loss'].get('cbdice_alpha', 1.0), beta=conf['loss'].get('cbdice_beta', 1.0)),
        'topo_win': lambda: MSETopoWinLoss(
            weights=conf['loss'].get('topo_weights', [1.0, 1.0]),
            alpha=conf['loss'].get('topo_alpha_mse', 1e-3) # Use a different key to avoid conflict
        )
    }

    weighted_loss_constructors = {
        'bce': lambda: lambda p, t, w: weighted_bce_loss(p, t, w, beta=conf['loss']['topo']['beta']),
        'dice': lambda: lambda p, t, w: weighted_dice_loss(p.sigmoid(), t, w, eps=eps),
        'tversky': lambda: lambda p, t, w: weighted_tversky_loss(p.sigmoid(), t, w, alpha=conf['loss']['tversky_alpha'], beta=conf['loss']['tversky_beta'], eps=eps),
        'focal_tversky': lambda: lambda p, t, w: weighted_focal_tversky_loss(p.sigmoid(), t, w, alpha=conf['loss']['tversky_alpha'], beta=conf['loss']['tversky_beta'], gamma=conf['loss']['focal_tversky_gamma'], eps=eps),
    }

    # --- Another special case for the standalone TOPO Windowed Loss ---
    if name == 'topo_win':
        if combine_name or use_topo:
            warnings.warn("'topo_win' is a standalone loss and should not be combined or weighted. Ignoring 'combine_with' and 'use_topographic'.")
        # It expects raw logits, so we return it directly.
        return loss_constructors['topo_win']()

    # --- Main factory logic for all other composable losses ---
    def final_loss_fn(preds, targets, **kwargs):
        dist = kwargs.get('dist', None)
        
        is_weighted = use_topo and name in weighted_loss_constructors
        constructors = weighted_loss_constructors if is_weighted else loss_constructors

        if name not in constructors:
            raise ValueError(f"Loss '{name}' not supported in this configuration.")
        main_loss_fn = constructors[name]()

        combined_loss_fn = None
        if combine_name and w_comb > 0:
            if combine_name not in loss_constructors:
                 raise ValueError(f"Combined loss '{combine_name}' not supported.")
            combined_loss_fn = loss_constructors[combine_name]()

        weight_map = None
        if is_weighted:
            if dist is None:
                raise ValueError(f"Distance map must be provided for topographic loss '{name}'.")
            dist_norm = dist / (dist.amax(dim=(2, 3), keepdim=True) + eps)
            weight_map = (1.0 + dist_norm) ** conf['loss']['topo']['alpha']

        # --- Calculate final loss ---
        if is_weighted:
            loss = w_main * main_loss_fn(preds, targets, weight_map)
        else:
            loss = w_main * main_loss_fn(preds, targets)

        if combined_loss_fn:
            loss += w_comb * combined_loss_fn(preds, targets)
            
        return loss

    return final_loss_fn