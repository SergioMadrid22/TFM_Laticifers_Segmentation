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
class LaticiferPatchTrain(Dataset):
    def __init__(self, df, root_dir, patch_size=(512, 512), num_patches=20,
                 augment=True, fg_threshold=0.03, fg_ratio=0.7, dist_transform=False):
        """
        Args:
          df: DataFrame with image and mask paths.
          fg_threshold: minimum fraction of foreground pixels to count a patch as "positive".
          fg_ratio: proportion of positive vs. random patches.
        """
        self.df = df.reset_index(drop=True)
        self.root = root_dir
        self.patch_h, self.patch_w = patch_size
        self.num_patches = num_patches
        self.augment = augment
        self.fg_threshold = fg_threshold
        self.fg_ratio = fg_ratio

        self.transforms = A.Compose([
            A.RandomCrop(self.patch_h, self.patch_w),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.3),
            A.ElasticTransform(p=0.3),
            A.GaussianBlur(p=0.2),
            A.Normalize(),
            ToTensorV2()
        ]) if augment else A.Compose([
            A.RandomCrop(self.patch_h, self.patch_w),
            A.Normalize(),
            ToTensorV2()
        ])

        # Build sample list with bias toward foreground
        self.samples = []
        for img_idx in range(len(self.df)):
            gray = np.array(Image.open(os.path.join(root_dir, self.df.loc[img_idx, 'enhanced_img_path'])).convert('L'))
            mask = np.array(Image.open(os.path.join(root_dir, self.df.loc[img_idx, 'mask_path'])).convert('L')) // 255

            h, w = gray.shape
            max_tries = self.num_patches * 5
            patches = 0
            tries = 0

            # 1) FG-biased patches
            while patches < int(self.num_patches * self.fg_ratio) and tries < max_tries:
                top = random.randint(0, h - self.patch_h)
                left = random.randint(0, w - self.patch_w)
                m_patch = mask[top:top+self.patch_h, left:left+self.patch_w]
                if m_patch.mean() >= self.fg_threshold:
                    self.samples.append((img_idx, top, left))
                    patches += 1
                tries += 1

            # 2) Random patches
            for _ in range(self.num_patches - patches):
                top = random.randint(0, h - self.patch_h)
                left = random.randint(0, w - self.patch_w)
                self.samples.append((img_idx, top, left))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_idx, top, left = self.samples[idx]
        row = self.df.loc[img_idx]

        img = np.array(Image.open(os.path.join(self.root, row['enhanced_img_path'])).convert('L'))
        mask = np.array(Image.open(os.path.join(self.root, row['mask_path'])).convert('L')) // 255

        img_patch = img[top:top+self.patch_h, left:left+self.patch_w]
        mask_patch = mask[top:top+self.patch_h, left:left+self.patch_w]

        augmented = self.transforms(image=img_patch, mask=mask_patch)
        image = augmented['image']
        mask = augmented['mask'].unsqueeze(0).float()

        return image, mask
'''
