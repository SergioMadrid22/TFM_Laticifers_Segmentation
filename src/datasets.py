from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
import os
from PIL import Image
from sklearn.model_selection import train_test_split
import pandas as pd
import random
from torchvision import transforms
import torch
import cv2

def apply_clahe(image_np, clip_limit=2.0, tile_grid_size=(8, 8)):
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    return clahe.apply(image_np)

class LaticiferDataset(Dataset):
    def __init__(self, df, root_dir, image_size=(1024, 1024), augment=False):
        self.df = df
        self.root = root_dir
        self.augment = augment
        self.image_size = image_size

        self.transforms = A.Compose([
            A.Resize(*image_size),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.3),
            A.ElasticTransform(p=0.3),
            A.GaussianBlur(p=0.2),
            A.Normalize(),
            ToTensorV2()
        ]) if augment else A.Compose([
            A.Resize(*image_size),
            A.Normalize(),
            ToTensorV2()
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        gray_path = os.path.join(self.root, row['gray_img_path'])
        enhanced_path = os.path.join(self.root, row['enhanced_img_path'])
        mask_path = os.path.join(self.root, row['mask_path'])

        image = np.array(Image.open(enhanced_path).convert("L"))
        mask = np.array(Image.open(mask_path).convert("L")) // 255

        augmented = self.transforms(image=image, mask=mask)
        return augmented['image'], augmented['mask'].unsqueeze(0).float()

'''
class LaticiferPatchTrain(Dataset):
    def __init__(
        self, filenames, root_dir, patch_size=(512, 512), num_patches=20,
        augment=True, dist_transform=False, use_clahe=False, num_channels=3
    ):
        self.filenames = filenames
        self.root = root_dir
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.augment = augment
        self.dist_transform = dist_transform
        self.use_clahe = use_clahe
        self.num_channels = num_channels

        normalize = A.Normalize(
            mean=(0.485, 0.456, 0.406) if num_channels == 3 else (0.5,),
            std=(0.229, 0.224, 0.225) if num_channels == 3 else (0.5,)
        )

        self.transforms = A.Compose([
            A.RandomCrop(*patch_size),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.3),
            A.ElasticTransform(p=0.3),
            A.GaussianBlur(p=0.2),
            normalize,
            ToTensorV2()
        ], additional_targets={'distance': 'mask'} if self.dist_transform else None)

        self.samples = []
        for i in range(len(self.filenames)):
            self.samples.extend([i] * num_patches)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        file_idx = self.samples[idx]
        fname = self.filenames[file_idx]

        img_path = os.path.join(self.root, "enhanced_images", fname)
        mask_path = os.path.join(self.root, "masks", fname)
        dist_path = os.path.join(self.root, "distance_maps_pt", fname.replace(".tif", ".pt"))

        image = np.array(Image.open(img_path).convert("L"))
        if self.use_clahe:
            image = apply_clahe(image)

        if self.num_channels == 3:
            image = np.stack([image] * 3, axis=-1)  # Convert to RGB shape [H, W, 3]

        mask = np.array(Image.open(mask_path).convert("L")) // 255

        if self.dist_transform:
            distance = torch.load(dist_path, weights_only=True).squeeze(0).numpy()
            augmented = self.transforms(image=image, mask=mask, distance=distance)
        else:
            augmented = self.transforms(image=image, mask=mask)

        image_tensor = augmented['image']
        mask_tensor = augmented['mask'].unsqueeze(0).float()

        if self.dist_transform:
            distance_tensor = augmented['distance'].unsqueeze(0).float()
            return image_tensor, mask_tensor, distance_tensor

        return image_tensor, mask_tensor
'''
'''
class LaticiferPatchTest(Dataset):
    def __init__(
        self, filenames, root_dir, patch_size=(512, 512), stride=(256, 256),
        dist_transform=False, use_clahe=True, num_channels=3
    ):
        self.filenames = filenames
        self.root_dir = root_dir
        self.patch_size = patch_size
        self.stride = stride
        self.dist_transform = dist_transform
        self.use_clahe = use_clahe
        self.num_channels = num_channels

        normalize = A.Normalize(
            mean=(0.485, 0.456, 0.406) if num_channels == 3 else (0.5,),
            std=(0.229, 0.224, 0.225) if num_channels == 3 else (0.5,)
        )

        self.transforms = A.Compose([
            normalize,
            ToTensorV2()
        ], additional_targets={'distance': 'mask'} if self.dist_transform else None)

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        fname = self.filenames[idx]
        img_path = os.path.join(self.root_dir, "enhanced_images", fname)
        mask_path = os.path.join(self.root_dir, "masks", fname)
        dist_path = os.path.join(self.root_dir, "distance_maps_pt", fname.replace(".tif", ".pt"))

        image = np.array(Image.open(img_path).convert("L"))
        if self.use_clahe:
            image = apply_clahe(image)

        if self.num_channels == 3:
            image = np.stack([image] * 3, axis=-1)

        mask = np.array(Image.open(mask_path).convert("L")) // 255
        if self.dist_transform:
            distance = torch.load(dist_path, weights_only=True).squeeze(0).numpy()

        original_size = image.shape[:2]
        pad_h = (self.patch_size[0] - original_size[0] % self.patch_size[0]) % self.patch_size[0]
        pad_w = (self.patch_size[1] - original_size[1] % self.patch_size[1]) % self.patch_size[1]

        image = np.pad(image, ((0, pad_h), (0, pad_w), (0, 0)) if self.num_channels == 3 else ((0, pad_h), (0, pad_w)), mode='constant')
        mask = np.pad(mask, ((0, pad_h), (0, pad_w)), mode='constant')
        if self.dist_transform:
            distance = np.pad(distance, ((0, pad_h), (0, pad_w)), mode='constant')

        H, W = image.shape[:2]
        patches, mask_patches, dist_patches, coords = [], [], [], []

        for top in range(0, H - self.patch_size[0] + 1, self.stride[0]):
            for left in range(0, W - self.patch_size[1] + 1, self.stride[1]):
                img_patch = image[top:top+self.patch_size[0], left:left+self.patch_size[1]]  # [H, W, C] or [H, W]
                mask_patch = mask[top:top+self.patch_size[0], left:left+self.patch_size[1]]
                if self.dist_transform:
                    dist_patch = distance[top:top+self.patch_size[0], left:left+self.patch_size[1]]
                    transformed = self.transforms(image=img_patch, mask=mask_patch, distance=dist_patch)
                else:
                    transformed = self.transforms(image=img_patch, mask=mask_patch)

                patches.append(transformed['image'])
                mask_patches.append(transformed['mask'].unsqueeze(0).float())
                if self.dist_transform:
                    dist_patches.append(transformed['distance'].unsqueeze(0).float())
                coords.append((top, left))

        result = {
            'image_patches': torch.stack(patches),
            'mask_patches': torch.stack(mask_patches),
            'coords': coords,
            'image_size': (H, W),
            'original_size': original_size,
            'image_idx': idx,
            'filename': fname
        }
        if self.dist_transform:
            result['dist_patches'] = torch.stack(dist_patches)

        return result
'''

class LaticiferPatchTrain(Dataset):
    def __init__(
        self,
        feature_dirs,
        patch_size=(512, 512),
        patches_per_image=20,
        positive_ratio=0.8,
        dist_transform=False,
        fg_threshold=0.03,  # Fraction of patch area required to be vessel
        filenames=None
    ):
        self.feature_dirs = feature_dirs
        self.patch_size = patch_size
        self.patches_per_image = patches_per_image
        self.positive_ratio = positive_ratio
        self.dist_transform = dist_transform
        self.fg_threshold = fg_threshold
        self.filenames = filenames if filenames is not None else sorted(os.listdir(self.feature_dirs['mask']))

        additional_targets = {
            key: 'image' for key in feature_dirs if key not in ['mask', 'enhanced', 'distance']
        }
        if dist_transform:
            additional_targets['distance'] = 'mask'

        self.transforms = A.Compose(
            [
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.ElasticTransform(alpha=150, sigma=5, p=0.3, border_mode=cv2.BORDER_REFLECT_101),  # Removed alpha_affine
                A.GridDistortion(p=0.2, border_mode=cv2.BORDER_REFLECT_101),
                A.RandomBrightnessContrast(p=0.3),
                A.GaussNoise(std_range=(0.02, 0.1), mean_range=(0.0, 0.0), p=0.2),
                A.GaussianBlur(blur_limit=(3, 5), p=0.2),
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
            img = np.array(Image.open(path).convert("L")).astype(np.float32) #/ 255.0
            #if key == 'sato':
            #    img = (img - img.min()) / (img.max() - img.min() + 1e-5) * 255.0
            return img
        
    def _random_patch_coords(self, H, W):
        ph, pw = self.patch_size
        top = np.random.randint(0, H - ph + 1)
        left = np.random.randint(0, W - pw + 1)
        return top, left

    def _find_positive_patch(self, mask):
        """Find a patch where the vessel area exceeds the threshold."""
        max_tries = 50
        H, W = mask.shape
        ph, pw = self.patch_size

        for _ in range(max_tries):
            top, left = self._random_patch_coords(H, W)
            patch = mask[top:top + ph, left:left + pw]
            #print(f"patch proportion: {patch.sum() / patch.size}")
            if patch.sum() / patch.size >= self.fg_threshold:
                return top, left
        print("FAILED!")
        return self._random_patch_coords(H, W)  # fallback if none found

    def __getitem__(self, idx):
        file_idx = self.samples[idx]
        fname = self.filenames[file_idx]

        mask = self._load_feature(self.feature_dirs['mask'], fname, key='mask') // 255
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
            image=cropped_features['enhanced'],
            mask=mask_crop,
            **{k: v for k, v in cropped_features.items() if k != 'enhanced'}
        )

        feature_tensors = [augmented['image']]
        for k in cropped_features:
            if k != 'enhanced' and k != 'distance':
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
            key: 'image' for key in feature_dirs if key not in ['mask', 'enhanced', 'distance']
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
                    image=patch_crops['enhanced'],
                    mask=mask_crop,
                    **{k: v for k, v in patch_crops.items() if k != 'enhanced'}
                )

                feature_tensors = [augmented['image']]
                for k in patch_crops:
                    if k != 'enhanced' and k != 'distance':
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


def get_dataloaders(conf):
    dataset_root = conf['dataset']['root']
    dataset_csv = conf['dataset']['dataset_csv']
    image_size = conf['dataset']['image_size']
    num_workers = conf['dataset']['num_workers']
    batch_size=conf['train']['batch_size']
    
    df = pd.read_csv(os.path.join(dataset_root, dataset_csv))
    df = df[df["is_labeled"] == True].reset_index(drop=True)
    train_df, val_df = train_test_split(df, test_size=0.1, random_state=42)
    train_dataset = LaticiferDataset(train_df, dataset_root, image_size, augment=True)
    test_dataset = LaticiferDataset(val_df, dataset_root, image_size, augment=False)

    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=conf['dataset']['num_workers']
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )
    return train_loader, test_loader

'''
def get_patch_dataloaders(conf):
    dataset_root = conf['dataset']['root']
    patch_size = conf['dataset'].get('patch_size', (512, 512))
    num_patches = conf['dataset'].get('num_patches', 20)
    stride = conf['dataset'].get('stride', (patch_size[0] // 2, patch_size[1] // 2))
    num_workers = conf['dataset'].get('num_workers', 4)
    dist_transform = conf['dataset'].get('dist_transform', False)

    # Get filenames only from masks folder (to ensure labels exist)
    mask_dir = os.path.join(dataset_root, "masks")
    all_filenames = [f for f in os.listdir(mask_dir) if f.endswith(".tif")]
    all_filenames.sort()  # Optional: sort to ensure deterministic split

    # Split filenames into train and validation
    train_filenames, val_filenames = train_test_split(
        all_filenames, test_size=0.1, random_state=42
    )

    # Create datasets
    train_dataset = LaticiferPatchTrain(
        filenames=train_filenames,
        root_dir=dataset_root,
        patch_size=patch_size,
        num_patches=num_patches,
        augment=True,
        dist_transform=dist_transform,
        num_channels=conf['dataset'].get('num_channels', 1)
    )
    val_dataset = LaticiferPatchTest(
        filenames=val_filenames,
        root_dir=dataset_root,
        patch_size=patch_size,
        stride=stride,
        dist_transform=dist_transform,
        num_channels=conf['dataset'].get('num_channels', 1)
    )

    # Create dataloaders
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
'''

def get_patch_dataloaders(conf):
    patch_size = tuple(conf['dataset'].get('patch_size', (512, 512)))
    num_patches = conf['dataset'].get('num_patches', 20)
    stride = tuple(conf['dataset'].get('stride', (patch_size[0] // 2, patch_size[1] // 2)))
    num_workers = conf['dataset'].get('num_workers', 4)
    dist_transform = conf['dataset'].get('dist_transform', False)
    fg_threshold = conf['dataset'].get('fg_threshold', 0.04)
    positive_ratio = conf['dataset'].get('positive_ratio', 0.8)

    feature_dirs = conf['dataset']['feature_dirs']
    mask_dir = feature_dirs['mask']

    # Get available filenames from mask folder
    all_filenames = [f for f in os.listdir(mask_dir) if f.endswith(".tif")]
    all_filenames.sort()

    train_filenames, val_filenames = train_test_split(
        all_filenames, test_size=0.1, random_state=42
    )

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
