from pathlib import Path
import sys
import os

notebook_dir = Path(__file__).resolve().parent if "__file__" in globals() else Path().resolve()
ROOT = notebook_dir.parent
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split, Subset

from PhysicsNeMo.examples.weather.corrdiff.datasets.hrrrmini import HRRRMiniDataset
from consistency_model import ConsistencyDownscalingModel, ConsistencyLoss
from accelerate import Accelerator

print(f"Root directory: {ROOT}")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
stats_path = r"C:\Users\ryadw\OneDrive\Desktop\stats.json"
data_path = r"C:\Users\ryadw\OneDrive\Desktop\hrrr_mini_train.nc"
