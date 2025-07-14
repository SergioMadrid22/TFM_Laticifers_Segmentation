import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from models import MAE
from datasets import UnlabeledPatchDataset
import torch.nn.functional as F
from tqdm import tqdm

def mae_loss(pred, target, patch_size=16):
    B, _, H, W = target.shape
    target_patches = F.unfold(target, kernel_size=patch_size, stride=patch_size)
    target_patches = target_patches.transpose(1, 2)  # (B, N, P^2)
    return F.mse_loss(pred, target_patches)


def train_mae():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = UnlabeledPatchDataset("/home/smadper/TFM/datasets/laticifers/enhanced_images")
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=4)

    model = MAE().to(device)
    optimizer = Adam(model.parameters(), lr=1e-4)

    for epoch in range(50):
        model.train()
        total_loss = 0
        for imgs in dataloader:
            imgs = imgs.to(device)
            preds, _ = model(imgs)
            loss = mae_loss(preds, imgs)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch+1}, Loss: {total_loss / len(dataloader):.4f}")
        torch.save(model.state_dict(), f"mae_epoch_{epoch+1}.pth")

if __name__ == "__main__":
    train_mae()
