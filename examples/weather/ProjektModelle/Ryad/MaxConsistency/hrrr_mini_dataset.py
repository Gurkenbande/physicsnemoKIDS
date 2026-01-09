import torch
from torch.utils.data import Dataset, DataLoader, random_split
import xarray as xr
import numpy as np
import json
from pathlib import Path

print ("start")

class SimpleHRRRMiniDataset(Dataset):
    def __init__(self, data_path: str, stats_path: str, normalize: bool = True):
        self.data_path = Path(data_path)
        self.stats_path = Path(stats_path)
        self.normalize = normalize
        
        if not self.data_path.exists():
            raise FileNotFoundError(f"Data file not found: {self.data_path}")
        if not self.stats_path.exists():
            raise FileNotFoundError(f"Stats file not found: {self.stats_path}")
        
        with open(self.stats_path, 'r') as f:
            self.stats = json.load(f)
        
        print(f"Loading dataset from {self.data_path}...")
        self.ds = xr.open_dataset(self.data_path)
        self.n_samples = len(self.ds.time)
        
        print(f"Dataset loaded: {self.n_samples} samples")
    
    def __len__(self):
        return self.n_samples
    
    def _normalize_data(self, data: np.ndarray, var_name: str, is_input: bool = True) -> np.ndarray:
        if not self.normalize:
            return data
        
        stats_key = 'input' if is_input else 'output'
        
        if stats_key in self.stats and var_name in self.stats[stats_key]:
            mean = self.stats[stats_key][var_name].get('mean', 0.0)
            std = self.stats[stats_key][var_name].get('std', 1.0)
            data = (data - mean) / (std + 1e-8)
        
        return data
    
    def __getitem__(self, idx: int):
        sample = self.ds.isel(time=idx)

        input_vars = ['u10m', 'v10m', 't2m', 'sp', 'msl', 'tcwv']
        
        x_list = []
        for var in input_vars:
            if var in sample:
                data = sample[var].values.astype(np.float32)
                data = self._normalize_data(data, var, is_input=True)
                
                if data.ndim == 1:
                    H = W = int(np.sqrt(len(data)))
                    data = data.reshape(H, W)
                
                x_list.append(data)

        while len(x_list) < 28:
            x_list.append(np.zeros_like(x_list[0]))
        
        x = np.stack(x_list[:28], axis=0)
        output_vars = ['u10m', 'v10m', 't2m', 'precip']
        
        y_list = []
        for var in output_vars:
            var_name = f'hrrr_{var}' if f'hrrr_{var}' in sample else var
            
            if var_name in sample:
                data = sample[var_name].values.astype(np.float32)
                data = self._normalize_data(data, var, is_input=False)
                
                if data.ndim == 1:
                    H = W = int(np.sqrt(len(data)))
                    data = data.reshape(H, W)
                
                y_list.append(data)
        
        y = np.stack(y_list, axis=0) if y_list else np.zeros((4, 64, 64), dtype=np.float32)
        
        x = torch.from_numpy(x)
        y = torch.from_numpy(y)
        
        return x, y


def create_dataloaders(data_path: str, stats_path: str, batch_size: int = 10,
                      train_ratio: float = 0.9, val_ratio: float = 0.05, 
                      num_workers: int = 0):
    dataset = SimpleHRRRMiniDataset(
        data_path=data_path,
        stats_path=stats_path,
        normalize=True
    )
    
    n = len(dataset)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    n_test = n - n_train - n_val
    
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [n_train, n_val, n_test]
    )
    print(f"trainieren:{len(train_dataset)}")
    print(f"val:{len(val_dataset)}")
    print(f"testen:{len(test_dataset)}") 

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    
    return train_loader, val_loader, test_loader       