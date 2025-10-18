import torch
import torch.nn as nn
import torch.nn.functional as F
import monai
import numpy as np
import random
import skimage.measure

# =========================================================================
#  ------ PART 1: MASKED AUTOENCODER (Unet_MIM from DeepClosing.py) ------
# =========================================================================
# This is the network we will pre-train to learn the shape of laticifers.

class Unet_MIM(nn.Module):
    "Unet with Masked Image Modeling, adapted from the DeepClosing paper."
    def __init__(self, in_channels=1, out_channels=1, mask_ratio=(0.0, 0.25), patch_size=([2,8],[2,8])):
        super().__init__()
        self.patch_size = patch_size
        self.mask_ratio = mask_ratio
        
        # We use a standard MONAI U-Net as the core architecture
        self.net = monai.networks.nets.UNet(
            spatial_dims=2,
            in_channels=in_channels,
            out_channels=out_channels,
            channels=(32, 64, 128, 256, 512),
            strides=(2, 2, 2, 2),
            num_res_units=2
        )
        self.final_act = nn.Sigmoid()

    def get_mask_params(self):
        # Handle random patch sizes and mask ratios from a range
        if isinstance(self.patch_size[0], list):
            patch_h = random.choice(list(range(*self.patch_size[0])))
            patch_w = random.choice(list(range(*self.patch_size[1])))
        else:
            patch_h, patch_w = self.patch_size

        if isinstance(self.mask_ratio, tuple):
            mask_ratio = np.random.uniform(*self.mask_ratio)
        else:
            mask_ratio = self.mask_ratio
            
        return mask_ratio, (patch_h, patch_w)

    def forward(self, imgs):
        # This forward pass is for the self-supervised pre-training
        mask_ratio, patch_size = self.get_mask_params()
        masked_img, mask = self.random_masking(imgs, mask_ratio, patch_size)

        pred_logits = self.net(masked_img)
        pred_prob = self.final_act(pred_logits)
        
        # The loss is calculated only on the parts that were masked out
        loss = F.mse_loss(pred_prob * mask, imgs * mask)

        return {
            "loss": loss, 
            "prediction": pred_prob, 
            "original_image": imgs,
            "masked_image": masked_img, 
            "mask": mask
        }

    def random_masking(self, input_tensor, mask_ratio, patch_size):
        # This function creates the random blocky mask
        B, C, H, W = input_tensor.shape
        p_h, p_w = patch_size
        
        new_H = int(np.ceil(H / p_h) * p_h)
        new_W = int(np.ceil(W / p_w) * p_w)
        
        mask = torch.ones(1, 1, new_H, new_W, device=input_tensor.device)

        h, w = new_H // p_h, new_W // p_w
        num_patches = h * w
        len_keep = int(num_patches * (1 - mask_ratio))
        
        shuffled_indices = torch.randperm(num_patches, device=input_tensor.device)
        keep_indices = shuffled_indices[:len_keep]

        # Create a mask where kept patches are 0 and removed patches are 1
        patch_mask = torch.ones(1, num_patches, 1, device=input_tensor.device)
        patch_mask[:, keep_indices, :] = 0
        
        # Reshape back to image format
        patch_mask = patch_mask.reshape(1, h, w, 1, 1, 1)
        patch_mask = patch_mask.expand(1, h, w, p_h, p_w, 1)
        patch_mask = patch_mask.permute(0, 5, 1, 3, 2, 4).reshape(1, 1, new_H, new_W)

        # Crop back to original image size
        final_mask = patch_mask[:, :, :H, :W]
        
        masked_img = input_tensor * (1 - final_mask)
        return masked_img, final_mask

# =========================================================================
#  ------ PART 2: SIMPLE POINT EROSION (from Simple_Point_Erosion_Module.py) ------
# =========================================================================
# This is the learnable, topology-preserving "Erosion" part of Deep Closing.

def whether_center_point_is_simple_point(patch):
    """Checks if a 3x3 patch's center is a 'simple point' (can be removed without changing topology)."""
    p1 = patch.copy(); p2 = patch.copy()
    
    _, num_fg_8 = skimage.measure.label(p1, connectivity=2, return_num=True)
    _, num_bg_4 = skimage.measure.label(1 - p1, connectivity=1, return_num=True)
    
    p2[1, 1] = 1 - p2[1, 1] # Flip the center pixel
    
    _, num_fg_8_flipped = skimage.measure.label(p2, connectivity=2, return_num=True)
    _, num_bg_4_flipped = skimage.measure.label(1 - p2, connectivity=1, return_num=True)
    
    return num_fg_8 == num_fg_8_flipped and num_bg_4 == num_bg_4_flipped

class Simple_Point_Erosion_Module(nn.Module):
    """
    Performs a topology-preserving erosion using a pre-computed lookup table
    of 'simple points', as described in the DeepClosing paper.
    """
    def __init__(self, target_hw=(512, 512), device=torch.device("cuda:0")):
        super().__init__()
        self.H, self.W = target_hw
        self.device = device
        self.binary_power = (2**torch.arange(8, -1, -1, device=self.device)).float()
        self.lookup_table = self._construct_lookup_table().to(device)
        self._build_order_masks()

    def _construct_lookup_table(self):
        lookup_table = nn.Embedding(2**9, 1)
        lookup_table.weight.data.fill_(0)
        for i in range(2**9):
            binary_string = bin(i)[2:].zfill(9)
            patch = np.array([int(x) for x in binary_string]).reshape(3, 3)
            if whether_center_point_is_simple_point(patch):
                lookup_table.weight.data[i] = 1.0
        return lookup_table

    def _build_order_masks(self):
        self.order_mask_list = []
        masks = [torch.zeros(self.H, self.W) for _ in range(4)]
        for r in range(self.H):
            for c in range(self.W):
                idx = (r % 2) * 2 + (c % 2)
                masks[idx][r, c] = 1
        for m in masks:
            self.order_mask_list.append(m.unsqueeze(0).unsqueeze(0).to(self.device))

    def _get_simple_points_for_order(self, T, i):
        if T.shape[-2:] != (self.H, self.W):
            self.H, self.W = T.shape[-2:]
            self._build_order_masks()

        B, C, H, W = T.shape
        # F.unfold expects a 4D tensor
        patches = F.unfold(T, kernel_size=3, padding=1, stride=1)
        patches = patches.permute(0, 2, 1).reshape(B * H * W, 9)
        
        # --- MODIFICATION 2: Use the pre-computed self.binary_power ---
        # Now it's a valid FloatTensor @ FloatTensor multiplication
        patch_indices = (patches @ self.binary_power).long()
        
        simple_point_labels = self.lookup_table(patch_indices).reshape(B, 1, H, W)
        return simple_point_labels * self.order_mask_list[i]

    @torch.no_grad()
    def erode(self, T, M_T, max_k=1000):
        T = T.to(self.device); M_T = M_T.to(self.device)
        last_sum = T.sum()
        for k in range(max_k):
            for i in range(4): # Four sub-iterations for directional stability
                simple_points = self._get_simple_points_for_order(T, i)
                T = T - simple_points * T * M_T # Remove simple points only within the mask M_T
            
            current_sum = T.sum()
            if current_sum == last_sum:
                break
            last_sum = current_sum
        return T