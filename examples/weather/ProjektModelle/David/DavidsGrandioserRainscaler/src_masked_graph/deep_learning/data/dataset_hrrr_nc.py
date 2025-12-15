import os
import numpy as np
import torch
from torch.utils.data import Dataset
import xarray as xr


class DatasetHRRRNC(Dataset):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt

        nc_path = opt['dataroot_H']
        engine = opt.get('nc_engine', 'h5netcdf')
        group_in = opt.get('nc_group_in', 'input')
        group_out = opt.get('nc_group_out', 'output')

        if not os.path.isfile(nc_path):
            raise FileNotFoundError(f"NetCDF file not found: {nc_path}")

        self.ds_in = xr.open_dataset(nc_path, group=group_in, engine=engine)
        self.ds_out = xr.open_dataset(nc_path, group=group_out, engine=engine)

        self.in_vars = list(self.ds_in.data_vars)
        self.out_vars = list(self.ds_out.data_vars)

        if 'sample' not in self.ds_in.dims or 'sample' not in self.ds_out.dims:
            raise ValueError("Both input and output groups must have a 'sample' dimension.")

        self.n_samples = self.ds_in.sizes['sample']
        if self.ds_out.sizes['sample'] != self.n_samples:
            raise ValueError("Input and output groups have different number of samples.")

        self.scale = opt.get('scale', 8)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        lr_list = [self.ds_in[v].isel(sample=idx).values for v in self.in_vars]
        lr = np.stack(lr_list, axis=0).astype(np.float32) 

        hr_list = [self.ds_out[v].isel(sample=idx).values for v in self.out_vars]
        hr = np.stack(hr_list, axis=0).astype(np.float32)  
        return {
            'L': torch.from_numpy(lr),
            'H': torch.from_numpy(hr),
            'idx': idx
        }