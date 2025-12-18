from pathlib import Path
import sys

notebook_dir = Path(__file__).resolve().parent if "__file__" in globals() else Path().resolve()
ROOT = notebook_dir.parent

sys.path.insert(0, "/home/s448562/LAB4")
print(ROOT)
import os

import torch
from trainer import Trainer

from gnn4cd_model import GNN4CD_Model
import json 
from PhysicsNeMo.examples.weather.corrdiff.datasets.hrrrmini import HRRRMiniDataset

device = torch.device("cuda" if torch.cuda.is_available() else "mps")

from graphBuilder import BipartiteGraph

from train_test import Trainer, Tester
from torch_geometric.loader import DataLoader
import torch.nn as nn
from accelerate import Accelerator

print("Device: ", device)


'''
- data.x_dict['low']: shape (n_coarse, seq_l, n_features) → (64, xx, 26)
    - 64 coarse nodes (8x8 grid)
    - seq_l = xx timesteps per node ??
    - 26 input features per timestep (ERA5 variables) + 2 invariant
- data.x_dict['high']: shape (n_fine, n_high_features) -> (4096,4)
    - 4096 fine nodes (64x64 grid)
    - 4 output features per node (HRRR variables)
- y_high: ground truth labels for the fine nodes, same shape as x_dict['high']
- normalization stats: precomputed mean and standard deviation for each variable:
    - ERA5 input (low-res)
    - HRRR output (high-res)
    - invariant (e.g. latitude, longitude, elevation)

'''

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


stats_path = "/home/s448562/LAB4/data_corrdiff_mini/stats.json"
data_path = "/home/s448562/LAB4/data_corrdiff_mini/hrrr_mini_train.nc"

#stats_path = "/Users/emely/Uni/Lab4/Lab4/data_corrdiff_mini/stats.json"
#data_path = "/Users/emely/Uni/Lab4/Lab4/data_corrdiff_mini/hrrr_mini_train.nc"

ds = HRRRMiniDataset(data_path=data_path, stats_path=stats_path)

dataset = BipartiteGraph(
        ds = ds,
        device = device,
        coarse_shape=(8,8),
        fine_shape=(64,64),
        neighbors = 6,
        seq_len = None
)


model = GNN4CD_Model(
    encoding_dim = 64,
    n_input_features = 28,  
    h_hid = 32,  
    n_layers = 2,
    high_in = 4,  
    low2high_out = 32,
    high_out = 32,
    n_output_vars = 4
).to(device)

args = type('', (), {})() 
args.epochs = 5
args.alpha = 0.75
args.output_path = "./output/"
args.loss_fn = "MSE"
args.model_type = "Rall"   
args.log_file = "test.log"

subset_size = 5000

train_size = int(0.9 * len(dataset))
val_size = int(0.05 * len(dataset))
test_size = len(dataset) - train_size - val_size 

train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, val_size, test_size])

train_dataset = torch.utils.data.Subset(train_dataset, list(range(len(train_dataset)-subset_size, len(train_dataset))))
val_dataset = torch.utils.data.Subset(val_dataset, list(range(len(val_dataset)-subset_size//10, len(val_dataset))))
test_dataset = torch.utils.data.Subset(test_dataset,list(range(min(len(test_dataset), subset_size // 10))))

train_loader = DataLoader(train_dataset, batch_size=10, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=10, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=10, shuffle=False)

trainer = Trainer()
tester = Tester()

optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
loss_fn = nn.MSELoss()
accelerator = Accelerator(mixed_precision="fp16")

model, optimizer, train_loader, val_loader, test_loader = accelerator.prepare(model, optimizer, train_loader, val_loader, test_loader)

trainer.train_R_Rall(
    model = model,
    dataloader_train=train_loader,
    dataloader_val=val_loader,
    optimizer=optimizer,
    loss_fn=loss_fn,
    lr_scheduler = None,
    accelerator= accelerator,
    args = args
)

pr, times = tester.test(model=model,dataloader=test_loader,args=args,accelerator=accelerator)
