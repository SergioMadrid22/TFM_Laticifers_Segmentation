from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
import os
from PIL import Image
from sklearn.model_selection import train_test_split
import pandas as pd
import random
import torch

def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

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
    def __init__(self, df, root_dir, patch_size=(512, 512), num_patches=20, augment=True):
        self.df = df
        self.root = root_dir
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.augment = augment

        self.transforms = A.Compose([
            A.RandomCrop(*patch_size),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.3),
            A.ElasticTransform(p=0.3),
            A.GaussianBlur(p=0.2),
            A.Normalize(),
            ToTensorV2()
        ])

        self.samples = []
        for idx in range(len(self.df)):
            for _ in range(num_patches):
                self.samples.append(idx)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        row_idx = self.samples[idx]
        row = self.df.iloc[row_idx]

        enhanced_path = os.path.join(self.root, row['enhanced_img_path'])
        mask_path = os.path.join(self.root, row['mask_path'])

        image = np.array(Image.open(enhanced_path).convert("L"))
        mask = np.array(Image.open(mask_path).convert("L")) // 255

        augmented = self.transforms(image=image, mask=mask)
        image_tensor = augmented['image']
        mask_tensor = augmented['mask'].unsqueeze(0).float()

        return image_tensor, mask_tensor


class LaticiferPatchTest(Dataset):
    def __init__(self, df, root_dir, patch_size=(512, 512), stride=(256, 256)):
        self.df = df.reset_index(drop=True)
        self.root = root_dir
        self.patch_size = patch_size
        self.stride = stride
        self.transforms = A.Compose([
            A.Normalize(),
            ToTensorV2()
        ])

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.root, row['enhanced_img_path'])
        mask_path = os.path.join(self.root, row['mask_path'])

        image = np.array(Image.open(img_path).convert("L"))
        mask = np.array(Image.open(mask_path).convert("L")) // 255
        original_size = image.shape  # (H, W)

        pad_h = (self.patch_size[0] - image.shape[0] % self.patch_size[0]) % self.patch_size[0]
        pad_w = (self.patch_size[1] - image.shape[1] % self.patch_size[1]) % self.patch_size[1]

        image = np.pad(image, ((0, pad_h), (0, pad_w)), mode='constant')
        mask = np.pad(mask, ((0, pad_h), (0, pad_w)), mode='constant')

        H, W = image.shape
        patches = []
        mask_patches = []
        coords = []

        for top in range(0, H - self.patch_size[0] + 1, self.stride[0]):
            for left in range(0, W - self.patch_size[1] + 1, self.stride[1]):
                img_patch = image[top:top+self.patch_size[0], left:left+self.patch_size[1]]
                mask_patch = mask[top:top+self.patch_size[0], left:left+self.patch_size[1]]
                transformed = self.transforms(image=img_patch, mask=mask_patch)
                patches.append(transformed['image'])  # [1, H, W]
                mask_patches.append(transformed['mask'].unsqueeze(0).float())
                coords.append((top, left))

        return {
            'image_patches': torch.stack(patches),
            'mask_patches': torch.stack(mask_patches),
            'coords': coords,
            'image_size': (H, W),             # padded size
            'original_size': original_size,   # original size
            'image_idx': idx
        }
    

def get_dataloaders(batch_size, dataset_root, image_size=(512, 512), dataset_csv="laticifer_dataset_index.csv", num_workers=4):
    df = pd.read_csv(os.path.join(dataset_root, dataset_csv))
    df = df[df["is_labeled"] == True].reset_index(drop=True)
    train_df, val_df = train_test_split(df, test_size=0.1, random_state=42)
    train_dataset = LaticiferDataset(train_df, dataset_root, image_size, augment=True)
    test_dataset = LaticiferDataset(val_df, dataset_root, image_size, augment=False)

    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers
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
    dataset_csv = conf['dataset']['dataset_csv']
    patch_size = conf['dataset']['patch_size']
    num_patches = conf['dataset']['num_patches']
    stride = conf['dataset']['stride']
    num_workers = conf['dataset']['num_workers']

    df = pd.read_csv(os.path.join(dataset_root, dataset_csv))
    df = df[df["is_labeled"] == True].reset_index(drop=True)
    
    train_df, val_df = train_test_split(df, test_size=0.1, random_state=42)
    train_dataset = LaticiferPatchTrain(
        df=train_df,
        root_dir=dataset_root,
        patch_size=patch_size,
        num_patches=num_patches,
        augment=True
    )
    test_dataset = LaticiferPatchTest(
        df=val_df, 
        root_dir=dataset_root,
        patch_size=patch_size,
        stride=stride
    )

    train_loader = DataLoader(
        train_dataset, 
        batch_size=conf['train']['batch_size'], 
        shuffle=True, 
        num_workers=num_workers
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=conf['test']['batch_size'],
        shuffle=False,
        num_workers=num_workers
    )
    return train_loader, test_loader