import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from models import SimCLR
from datasets import SimCLRDataset
import os
import numpy as np
from tqdm import tqdm

def nt_xent_loss(z1, z2, temperature=0.5):
    """
    Normalized Temperature-scaled Cross Entropy Loss (NT-Xent) for SimCLR.
    """
    B = z1.size(0)
    z = torch.cat([z1, z2], dim=0)  # [2B, D]
    z = F.normalize(z, dim=1)

    sim = torch.matmul(z, z.T)  # [2B, 2B]
    sim /= temperature

    # Mask out self-similarity
    mask = torch.eye(2 * B, dtype=torch.bool, device=z.device)
    sim.masked_fill_(mask, float('-inf'))

    # Create labels: for each i in [0, B), positive is i + B; for each i in [B, 2B), positive is i - B
    labels = torch.cat([torch.arange(B, 2 * B), torch.arange(0, B)]).to(z.device)

    # Apply cross entropy loss
    loss = F.cross_entropy(sim, labels)
    return loss



def train_simclr(data_dir, epochs=100, batch_size=64, lr=3e-4):
    image_paths = [os.path.join(data_dir, fname) for fname in os.listdir(data_dir)]
    dataset = SimCLRDataset(image_paths)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True)

    model = SimCLR(backbone='resnet18').cuda()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for x1, x2 in loader:
            x1, x2 = x1.cuda(), x2.cuda()
            z1, _ = model(x1)
            z2, _ = model(x2)
            loss = nt_xent_loss(z1, z2)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch+1}, Loss: {total_loss / len(loader):.4f}")

    torch.save(model.encoder.state_dict(), "simclr_encoder.pth")

if __name__ == "__main__":
    data_path = "/home/smadper/TFM/datasets/laticifers/enhanced_images"
    train_simclr(data_path, 50, 16, 0.0003)