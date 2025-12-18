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
#from graphBuilderMain import BipartiteGraph
from graphBuilder import BipartiteGraph

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


subset_size = 5000  
subset_dataset = torch.utils.data.Subset(dataset, list(range(len(dataset)-subset_size, len(dataset))))

trainer = Trainer(model, subset_dataset,device, batch_size=15, lr=0.001)

trainer.train(epochs=5)

test_loss, y_pred, y_true = trainer.test(return_predictions=True)



