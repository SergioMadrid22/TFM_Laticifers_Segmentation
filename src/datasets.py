from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
import os
from PIL import Image
import torch
import cv2


class LaticiferPatchTrain(Dataset):
    def __init__(
        self,
        feature_dirs,
        patch_size=(512, 512),
        patches_per_image=20,
        positive_ratio=0.8,
        dist_transform=False,
        fg_threshold=0.03,
        filenames=None,
        curriculum_level=0
    ):
        self.feature_dirs = feature_dirs
        self.patch_size = patch_size
        self.patches_per_image = patches_per_image
        self.positive_ratio = positive_ratio
        self.dist_transform = dist_transform
        self.fg_threshold = fg_threshold
        self.curriculum_level = curriculum_level

        self.filenames = filenames if filenames is not None else sorted(os.listdir(self.feature_dirs['mask']))
        additional_targets = {
            key: 'image' for key in feature_dirs if key not in ['mask', 'image', 'distance']
        }
        if dist_transform:
            additional_targets['distance'] = 'mask'

        self.transforms = A.Compose(
            [
                # --- 1. Basic Geometric Transformations ---
                # More robust than RandomRotate90, as it can be any angle.
                # ShiftScaleRotate is a powerful combo transform.
                A.Affine(
                    scale=(0.9, 1.1),           # Equivalent to scale_limit=0.1
                    translate_percent=0.0625,   # Equivalent to shift_limit=0.0625
                    rotate=(-45, 45),           # Equivalent to rotate_limit=45
                    p=0.8,
                    border_mode=cv2.BORDER_REFLECT_101
                ),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),

                # --- 2. Advanced Non-Rigid Deformations ---
                # These are crucial for biological tissue that can stretch or warp.
                # We use OneOf to apply only one of these powerful distortions at a time.
                A.OneOf([
                    A.ElasticTransform(
                        alpha=120,
                        sigma=120 * 0.05,
                        p=0.5,
                        border_mode=cv2.BORDER_REFLECT_101
                    ),
                    A.GridDistortion(p=0.5, border_mode=cv2.BORDER_REFLECT_101),
                    A.OpticalDistortion(distort_limit=0.5, p=0.5, border_mode=cv2.BORDER_REFLECT_101)
                ], p=0.0),

                # --- 3. Photometric and Quality Transformations ---
                # This is the most important addition. It simulates real-world
                # variations in lighting, staining, focus, and sensor noise.
                A.OneOf([
                    A.RandomBrightnessContrast(brightness_limit=[-0.2, 0.2], contrast_limit=[-0.2, 0.2], p=0.7),
                    A.RandomGamma(gamma_limit=(80, 120), p=0.5),
                ], p=0.9),

                A.OneOf([
                    A.GaussianBlur(blur_limit=(3, 7), p=0.5),
                    A.GaussNoise(p=0.5),
                ], p=0.4),
                
                # --- 4. Final Preprocessing ---
                # Normalization and conversion to PyTorch Tensor must be last.
                A.Normalize(mean=(0.5,) * len(feature_dirs), std=(0.5,) * len(feature_dirs)),
                ToTensorV2()
        ],
            additional_targets=additional_targets
        )

        self.samples = []
        for i in range(len(self.filenames)):
            self.samples.extend([i] * patches_per_image)

    def __len__(self):
        return len(self.samples)

    def _load_feature(self, dir_path, fname, key=None):
        if key == 'distance':
            fname = os.path.splitext(fname)[0] + '.pt'
            path = os.path.join(dir_path, fname)
            return torch.load(path, weights_only=True).squeeze(0).numpy()
        else:
            path = os.path.join(dir_path, fname)
            img = np.array(Image.open(path).convert("L"))#.astype(np.float32)
            if key in ['sato'] or (img.max() > 1 and img.max() < 255): # A heuristic to catch unnormalized images
                if img.max() > 0: # Avoid division by zero for black images
                    img = (img / img.max() * 255).astype(np.uint8)
            return img

    def _random_patch_coords(self, H, W):
        ph, pw = self.patch_size
        top = np.random.randint(0, H - ph + 1)
        left = np.random.randint(0, W - pw + 1)
        return top, left

    def _find_positive_patch(self, mask):
        max_tries = 50
        H, W = mask.shape
        ph, pw = self.patch_size
        for _ in range(max_tries):
            top, left = self._random_patch_coords(H, W)
            patch = mask[top:top + ph, left:left + pw]
            if patch.sum() / patch.size >= self.fg_threshold:
                return top, left
        print("POSITIVE PATCH SEARCH FAILED!")
        return self._random_patch_coords(H, W)

    def _apply_curriculum(self, mask):
        """Applies morphological dilation to the mask based on curriculum level."""
        if self.curriculum_level <= 0:
            return mask
        kernel = np.ones((3, 3), np.uint8)
        mask_dilated = cv2.dilate(mask.astype(np.uint8), kernel, iterations=self.curriculum_level)
        return mask_dilated.clip(0, 1)

    def __getitem__(self, idx):
        file_idx = self.samples[idx]
        fname = self.filenames[file_idx]

        mask = self._load_feature(self.feature_dirs['mask'], fname, key='mask') // 255
        mask = self._apply_curriculum(mask)

        H, W = mask.shape

        choose_positive = np.random.rand() < self.positive_ratio
        if choose_positive:
            top, left = self._find_positive_patch(mask)
        else:
            top, left = self._random_patch_coords(H, W)

        features = {
            key: self._load_feature(path, fname, key=key)
            for key, path in self.feature_dirs.items()
            if key not in ['mask', 'distance']
        }

        cropped_features = {
            key: feat[top:top+self.patch_size[0], left:left+self.patch_size[1]]
            for key, feat in features.items()
        }
        mask_crop = mask[top:top+self.patch_size[0], left:left+self.patch_size[1]]

        if self.dist_transform:
            dist = self._load_feature(self.feature_dirs['distance'], fname, key='distance')
            dist_crop = dist[top:top+self.patch_size[0], left:left+self.patch_size[1]]
            cropped_features['distance'] = dist_crop

        augmented = self.transforms(
            image=cropped_features['image'],
            mask=mask_crop,
            **{k: v for k, v in cropped_features.items() if k != 'image'}
        )

        feature_tensors = [augmented['image']]
        for k in cropped_features:
            if k != 'image' and k != 'distance':
                feat = augmented[k]
                if feat.ndim == 2:
                    feat = feat.unsqueeze(0)
                feature_tensors.append(feat)

        result = {
            'inputs': torch.cat(feature_tensors, dim=0),
            'masks': augmented['mask'].unsqueeze(0).float()
        }

        if self.dist_transform:
            result['dist_maps'] = augmented['distance'].unsqueeze(0).float()

        return result

class LaticiferPatchTest(Dataset):
    def __init__(
        self,
        feature_dirs,
        patch_size=(512, 512),
        stride=(256, 256),
        dist_transform=False,
        filenames=None
    ):
        self.feature_dirs = feature_dirs
        self.patch_size = patch_size
        self.stride = stride
        self.dist_transform = dist_transform
        self.filenames = filenames if filenames is not None else sorted(os.listdir(self.feature_dirs['mask']))

        additional_targets = {
            key: 'image' for key in feature_dirs if key not in ['mask', 'image', 'distance']
        }
        if dist_transform:
            additional_targets['distance'] = 'mask'

        self.transforms = A.Compose(
            [
                A.Normalize(mean=(0.5,) * len(feature_dirs), std=(0.5,) * len(feature_dirs)),
                ToTensorV2()
            ],
            additional_targets=additional_targets
        )

    def __len__(self):
        return len(self.filenames)

    def _load_feature(self, dir_path, fname, key=None):
        if key == 'distance':
            fname = os.path.splitext(fname)[0] + '.pt'
            path = os.path.join(dir_path, fname)
            return torch.load(path, weights_only=True).squeeze(0).numpy()
        else:
            path = os.path.join(dir_path, fname)
            img = np.array(Image.open(path).convert("L")).astype(np.float32) #/ 255.0
            #if key == 'sato':
            #    img = (img - img.min()) / (img.max() - img.min() + 1e-5) * 255.0
            return img

    def __getitem__(self, idx):
        fname = self.filenames[idx]
        mask = self._load_feature(self.feature_dirs['mask'], fname, key='mask') // 255
        H, W = mask.shape

        features = {
            key: self._load_feature(path, fname, key=key)
            for key, path in self.feature_dirs.items()
            if key not in ['mask', 'distance']
        }

        if self.dist_transform:
            distance = self._load_feature(self.feature_dirs['distance'], fname, key='distance')

        # Padding
        pad_h = (self.patch_size[0] - H % self.patch_size[0]) % self.patch_size[0]
        pad_w = (self.patch_size[1] - W % self.patch_size[1]) % self.patch_size[1]

        padded_features = {
            key: np.pad(f, ((0, pad_h), (0, pad_w)), mode='constant')
            for key, f in features.items()
        }

        mask = np.pad(mask, ((0, pad_h), (0, pad_w)), mode='constant')
        if self.dist_transform:
            distance = np.pad(distance, ((0, pad_h), (0, pad_w)), mode='constant')
            padded_features['distance'] = distance

        H_p, W_p = mask.shape
        patches, mask_patches, dist_patches, coords = [], [], [], []

        for top in range(0, H_p - self.patch_size[0] + 1, self.stride[0]):
            for left in range(0, W_p - self.patch_size[1] + 1, self.stride[1]):
                patch_crops = {
                    key: feat[top:top+self.patch_size[0], left:left+self.patch_size[1]]
                    for key, feat in padded_features.items()
                }
                mask_crop = mask[top:top+self.patch_size[0], left:left+self.patch_size[1]]

                augmented = self.transforms(
                    image=patch_crops['image'],
                    mask=mask_crop,
                    **{k: v for k, v in patch_crops.items() if k != 'image'}
                )

                feature_tensors = [augmented['image']]
                for k in patch_crops:
                    if k != 'image' and k != 'distance':
                        feat = augmented[k]
                        if feat.ndim == 2:
                            feat = feat.unsqueeze(0)
                        feature_tensors.append(feat)

                input_tensor = torch.cat(feature_tensors, dim=0)
                mask_tensor = augmented['mask'].unsqueeze(0).float()

                patches.append(input_tensor)
                mask_patches.append(mask_tensor)

                if self.dist_transform:
                    dist_tensor = augmented['distance'].unsqueeze(0).float()
                    dist_patches.append(dist_tensor)

                coords.append((top, left))

        result = {
            'image_patches': torch.stack(patches),
            'mask_patches': torch.stack(mask_patches),
            'coords': coords,
            'image_size': (H_p, W_p),
            'original_size': (H, W),
            'image_idx': idx,
            'filename': fname
        }

        if self.dist_transform:
            result['dist_patches'] = torch.stack(dist_patches)

        return result


def get_patch_dataloaders(conf, train_filenames, val_filenames):
    patch_size = tuple(conf['dataset'].get('patch_size', (512, 512)))
    num_patches = conf['dataset'].get('num_patches', 20)
    stride = tuple(conf['dataset'].get('stride', (patch_size[0] // 2, patch_size[1] // 2)))
    num_workers = conf['dataset'].get('num_workers', 4)
    dist_transform = conf['loss'].get('use_topographic', False)
    fg_threshold = conf['dataset'].get('fg_threshold', 0.0)
    positive_ratio = conf['dataset'].get('positive_ratio', 0.0)

    feature_dirs = conf['dataset']['feature_dirs']

    # Train dataset
    train_dataset = LaticiferPatchTrain(
        feature_dirs=feature_dirs,
        filenames=train_filenames,
        patch_size=patch_size,
        patches_per_image=num_patches,
        positive_ratio=positive_ratio,
        dist_transform=dist_transform,
        fg_threshold=fg_threshold
    )

    # Val dataset
    val_dataset = LaticiferPatchTest(
        feature_dirs=feature_dirs,
        filenames=val_filenames,
        patch_size=patch_size,
        stride=stride,
        dist_transform=dist_transform
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=conf['train']['batch_size'],
        shuffle=True,
        num_workers=num_workers
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=conf['test']['batch_size'],
        shuffle=False,
        num_workers=num_workers
    )

    return train_loader, val_loader
