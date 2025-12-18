import xarray as xr
import numpy as np
import torch
from torch_geometric.data import HeteroData, Dataset
from tqdm import tqdm
import torch.nn as nn 
from torch_geometric.loader import DataLoader as GeoDataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
import matplotlib.pyplot as plt 
from pathlib import Path

class Trainer:
    def __init__(self, model, dataset, device, batch_size=2, lr=0.001):
        self.device = device
        self.model = model.to(self.device)
        self.dataset = dataset
        self.batch_size = batch_size

        train_size = int(0.8 * len(dataset))
        val_size = int(0.1 * len(dataset))

        self.train_dataset = torch.utils.data.Subset(dataset, range(0, train_size))
        self.val_dataset = torch.utils.data.Subset(dataset, range(train_size, train_size + val_size))
        self.test_dataset = torch.utils.data.Subset(dataset, range(train_size + val_size, len(dataset)))

        self.train_loader = GeoDataLoader(self.train_dataset, batch_size=batch_size, shuffle=True)
        self.val_loader = GeoDataLoader(self.val_dataset, batch_size=batch_size, shuffle=False)
        self.test_loader = GeoDataLoader(self.test_dataset, batch_size=batch_size, shuffle=False)

        self.optimizer = Adam(self.model.parameters(), lr=lr, weight_decay=1e-5)
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5, patience=5)
        self.criterion = nn.MSELoss()

        self.train_losses = []
        self.val_losses = []

    def train_epoch(self):
        self.model.train()
        total_loss = 0
        for batch in tqdm(self.train_loader, desc = "Training"):
            batch = batch.to(self.device) 
            self.optimizer.zero_grad()
            pred = self.model(batch)
            target = batch['high'].y
            loss = self.criterion(pred, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            total_loss += loss.item()
        return total_loss / len(self.train_loader)

    def validate(self):
        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validation"):
                batch = batch.to(self.device)
                pred = self.model(batch)
                target = batch['high'].y
                loss = self.criterion(pred, target)
                total_loss += loss.item()
        return total_loss / len(self.val_loader)

    def train(self, epochs=10, save_dir="checkpoints"):
        save_path = Path(save_dir)
        save_path.mkdir(exist_ok=True)
        for epoch in range(epochs):
            print(f"\nEpoch {epoch+1}/{epochs}")
            train_loss = self.train_epoch()
            val_loss = self.validate()
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.scheduler.step(val_loss)
            print(f"Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
        self.plot_losses(save_path / "loss_curve.png")

    def test(self, return_predictions=False):
        self.model.eval()
        total_loss = 0.0
        preds = []
        targets = []
        with torch.no_grad():
            for batch in tqdm(self.test_loader, desc="Testing"):
                batch = batch.to(self.device)
                pred = self.model(batch)
                target = batch['high'].y
                loss = self.criterion(pred, target)
                total_loss += loss.item()
                if return_predictions:
                    preds.append(pred.cpu())
                    targets.append(target.cpu())
        avg_loss = total_loss / len(self.test_loader)
        print(f"Test Loss (MSE): {avg_loss:.6f}")
        if return_predictions:
            return avg_loss, torch.cat(preds), torch.cat(targets)

        return avg_loss

    def plot_losses(self, save_path):
        plt.figure(figsize=(8,5))
        plt.plot(self.train_losses, label="Train Loss")
        plt.plot(self.val_losses, label="Val Loss")
        plt.xlabel("Epoch")
        plt.ylabel("MSE Loss")
        plt.legend()
        plt.grid(True)
        plt.savefig(save_path, dpi=150)
        plt.close()
