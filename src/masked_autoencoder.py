import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
import random

### --- Augmentation Utilities --- ###
class GrayscaleAugmentations:
    def __init__(self):
        self.augment = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(degrees=90),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.95, 1.05)),
        ])

    def __call__(self, img):
        return self.augment(img)

### --- Dataset with Augmentations --- ###
class UnlabeledImageDataset(Dataset):
    def __init__(self, image_paths, patch_size=512):
        self.image_paths = image_paths
        self.patch_size = patch_size
        self.augment = GrayscaleAugmentations()
        self.to_tensor = transforms.ToTensor()

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert("L")
        w, h = img.size
        if w < self.patch_size or h < self.patch_size:
            # If image is smaller, resize up (optional, or skip)
            img = img.resize((max(w, self.patch_size), max(h, self.patch_size)))

        # Random crop
        left = random.randint(0, img.width - self.patch_size)
        top = random.randint(0, img.height - self.patch_size)
        img = img.crop((left, top, left + self.patch_size, top + self.patch_size))

        img = self.augment(img)
        img = self.to_tensor(img)
        return img


### --- Encoder / Decoder --- ###
class Encoder(nn.Module):
    def __init__(self, in_channels=1, base_channels=64):
        super().__init__()
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, base_channels, 3, padding=1), nn.ReLU(inplace=True)
        )
        self.enc2 = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(base_channels, base_channels*2, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(base_channels*2, base_channels*2, 3, padding=1), nn.ReLU(inplace=True)
        )
        self.enc3 = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(base_channels*2, base_channels*4, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(base_channels*4, base_channels*4, 3, padding=1), nn.ReLU(inplace=True)
        )
        self.enc4 = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(base_channels*4, base_channels*8, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(base_channels*8, base_channels*8, 3, padding=1), nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.enc1(x)
        x = self.enc2(x)
        x = self.enc3(x)
        x = self.enc4(x)
        return x

class Decoder(nn.Module):
    def __init__(self, out_channels=1, base_channels=64):
        super().__init__()
        self.up4 = nn.ConvTranspose2d(base_channels*8, base_channels*4, 2, stride=2)
        self.dec4 = nn.Sequential(
            nn.Conv2d(base_channels*4, base_channels*4, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(base_channels*4, base_channels*4, 3, padding=1), nn.ReLU(inplace=True)
        )

        self.up3 = nn.ConvTranspose2d(base_channels*4, base_channels*2, 2, stride=2)
        self.dec3 = nn.Sequential(
            nn.Conv2d(base_channels*2, base_channels*2, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(base_channels*2, base_channels*2, 3, padding=1), nn.ReLU(inplace=True)
        )

        self.up2 = nn.ConvTranspose2d(base_channels*2, base_channels, 2, stride=2)
        self.dec2 = nn.Sequential(
            nn.Conv2d(base_channels, base_channels, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, base_channels, 3, padding=1), nn.ReLU(inplace=True)
        )

        #self.up1 = nn.ConvTranspose2d(base_channels, base_channels, 2, stride=2)
        #self.dec1 = nn.Sequential(
        #    nn.Conv2d(base_channels, base_channels, 3, padding=1), nn.ReLU(inplace=True),
        #    nn.Conv2d(base_channels, base_channels, 3, padding=1), nn.ReLU(inplace=True)
        #)

        self.final = nn.Conv2d(base_channels, out_channels, 1)

    def forward(self, x):
        x = self.up4(x)
        x = self.dec4(x)
        x = self.up3(x)
        x = self.dec3(x)
        x = self.up2(x)
        x = self.dec2(x)
        x = self.final(x)
        return x


### --- Masked Autoencoder --- ###
class MaskedAutoencoder(nn.Module):
    def __init__(self, in_channels=1, base_channels=64, mask_ratio=0.5, patch_size=32):
        super().__init__()
        self.encoder = Encoder(in_channels, base_channels)
        self.decoder = Decoder(in_channels, base_channels)
        self.mask_ratio = mask_ratio
        self.patch_size = patch_size

    def forward(self, x):
        B, C, H, W = x.shape
        mask = self.generate_mask(B, H, W, self.patch_size, self.mask_ratio).to(x.device)
        masked_x = x * mask
        z = self.encoder(masked_x)
        recon = self.decoder(z)
        loss = F.mse_loss(recon, x)
        return loss, recon, mask

    def generate_mask(self, B, H, W, patch_size, mask_ratio):
        mask = torch.ones((B, 1, H, W))
        num_patches = (H // patch_size) * (W // patch_size)
        num_mask = int(mask_ratio * num_patches)

        for b in range(B):
            idxs = random.sample(range(num_patches), num_mask)
            for idx in idxs:
                i = (idx // (W // patch_size)) * patch_size
                j = (idx % (W // patch_size)) * patch_size
                mask[b, :, i:i+patch_size, j:j+patch_size] = 0
        return mask
    
def update_mask_ratio(epoch, step_size=10, base_ratio=0.5, step_increment=0.1, max_ratio=0.8):
    steps_completed = epoch // step_size
    new_ratio = base_ratio + steps_completed * step_increment
    return min(new_ratio, max_ratio)


### --- Training --- ###
def train_mae(model, dataloader, epochs, lr=1e-4, device='cuda'):
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    best_loss = float('inf')
    best_encoder_state = None

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        model.mask_ratio = update_mask_ratio(epoch, step_size=10, base_ratio=0.5, step_increment=0.1, max_ratio=0.8)
        for images in dataloader:
            images = images.to(device)
            loss, recon, mask = model(images)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * images.size(0)

        avg_loss = epoch_loss / len(dataloader.dataset)
        print(f"Epoch [{epoch+1}/{epochs}] | Loss: {avg_loss:.4f} | Mask ratio: {model.mask_ratio}")

        # Save best encoder
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_encoder_state = model.encoder.state_dict().copy()
            torch.save(best_encoder_state, "mae_encoder_best3.pth")
            torch.save(mae.state_dict(), "mae_full_best3.pth")

    return model.encoder


### --- Usage --- ###
if __name__ == "__main__":
    root_dir = "/home/smadper/TFM/datasets/laticifers/enhanced_images"
    image_paths = [os.path.join(root_dir, fname) for fname in os.listdir(root_dir)
                   if fname.endswith(('.png', '.jpg', '.tif'))]

    dataset = UnlabeledImageDataset(image_paths)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=2)

    mae = MaskedAutoencoder(in_channels=1, base_channels=64, mask_ratio=0.5)
    pretrained_encoder = train_mae(mae, dataloader, epochs=100, lr=1e-4)

    torch.save(pretrained_encoder.state_dict(), "mae_encoder3.pth")
    torch.save(mae.state_dict(), "mae_full3.pth")