"""input (x): 26 ERA5 variables from 8x8 low-resolution grid
      surface: u10m, v10m, t2m, tcwv, sp, msl
      upper levels: u/v/z/t/q at 1000/850/500/250 hPa

output(y): 4HRRR variables from 64x64 high-resolution grid
      2t: 2-meter temperatur
      10u,10v: 10-meter winds
      tp: total precipitation

each grid point becomes a node

bipartite graph structure
  1. coarse nodes (64 nodes from 8x8 grid, represent 25km resolution, contain ERA5 input variables(26 features))
  2. fine nodes(4096 nodes from 64x64 grid, target HRRR output variables(4 features), represent 3km resolution, start with zeros)

edges: high-to-high: captures high-resolution spatial relationships
       low-to-high: connects high-res-node to its nearest low-res-node neighbors -> downscaling


  """

import xarray as xr
import numpy as np
import torch
from torch_geometric.data import HeteroData
from PhysicsNeMo.examples.weather.corrdiff.datasets.hrrrmini import HRRRMiniDataset
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class BipartiteGraph():
    def __init__(self, netcdf_path: str, stats: dict, seq_l: int = 26, 
                    coarse_shape=(8,8), fine_shape=(64,64), neighbors=3):
        super().__init__()
        self.path = netcdf_path
        self.seq_l = seq_l
        self.stats = stats
        self.neighbors= neighbors

        self.ds_input = xr.open_dataset(self.path, group='input', engine='netcdf4')
        self.ds_output = xr.open_dataset(self.path, group='output', engine='netcdf4')

        self.input_vars = list(self.ds_input.data_vars)
        self.output_vars = list(self.ds_output.data_vars)

        self.n_samples = self.ds_input.dims['sample']

        self.coarse_shape = coarse_shape
        self.fine_shape = fine_shape
        self.n_coarse = coarse_shape[0] * coarse_shape[1]
        self.n_fine = fine_shape[0] * fine_shape[1]

        y_c = np.linspace(0,1,self.coarse_shape[0])
        x_c = np.linspace(0,1,self.coarse_shape[1])
        yy_c, xx_c = np.meshgrid(y_c, x_c, indexing='ij')
        self.coarse_positions = np.stack([yy_c.flatten(), xx_c.flatten()], axis=1).astype(np.float32)

        y_f = np.linspace(0,1,self.fine_shape[0])
        x_f = np.linspace(0,1,self.fine_shape[1])
        yy_f, xx_f = np.meshgrid(y_f, x_f, indexing='ij')
        self.fine_positions = np.stack([yy_f.flatten(), xx_f.flatten()], axis=1).astype(np.float32)

        self.build_edges(neighbors)

    def __len__(self):
        return self.n_samples

    def build_edges(self, k_neighbors=4):
        Hf, Wf = self.fine_shape
        Hc, Wc = self.coarse_shape
        scale_y = Hc / Hf
        scale_x = Wc / Wf

        # Low -> High 
        edges_low_to_high = []

        for hi in range(Hf):
            for wi in range(Wf):
                high_idx = hi * Wf + wi
                
                yc = hi * scale_y
                xc = wi * scale_x

                possible_ys = np.clip([int(np.floor(yc)), int(np.ceil(yc))], 0, Hc-1)
                possible_xs = np.clip([int(np.floor(xc)), int(np.ceil(xc))], 0, Wc-1)

                neighbors = [y*Wc + x for y in possible_ys for x in possible_xs]

                for low_idx in neighbors[:k_neighbors]:
                    edges_low_to_high.append([low_idx, high_idx])

        self.edge_index_low_to_high = torch.tensor(np.array(edges_low_to_high).T, dtype=torch.long)

        # High -> High 
        edges_high_within = []
        for hi in range(Hf):
            for wi in range(Wf):
                high_idx = hi * Wf + wi
                for dh in [-1, 0, 1]:
                    for dw in [-1, 0, 1]:
                        nh, nw = hi + dh, wi + dw
                        if 0 <= nh < Hf and 0 <= nw < Wf:
                            nbr_idx = nh * Wf + nw
                            edges_high_within.append([high_idx, nbr_idx])

        self.edge_index_high_within = torch.tensor(np.array(edges_high_within).T, dtype=torch.long)


    def __getitem__(self, idx):
        seq_len = self.seq_l
        end = idx
        start = max(0, end - seq_len + 1)
        seq_indices = list(range(start, end + 1))
        if len(seq_indices) < seq_len:
            seq_indices = [seq_indices[0]] * (seq_len - len(seq_indices)) + seq_indices

        coarse_seq_frames = []
        for t in seq_indices:
            step_vars = []
            for var in self.input_vars:
                arr = self.ds_input[var].isel(sample=t).values.astype(np.float32).reshape(-1)
                
                arr = (arr - self.stats['input'][var]['mean']) / self.stats['input'][var]['std']
                step_vars.append(arr)
            coarse_seq_frames.append(np.stack(step_vars, axis=-1))
        coarse_seq = np.stack(coarse_seq_frames, axis=0).transpose(1,0,2)  # (n_coarse, seq_l, n_features)
        x_low = torch.tensor(coarse_seq, dtype=torch.float32)

        high_in = len(self.output_vars)
        x_high = torch.zeros((self.n_fine, high_in), dtype=torch.float32)

        fine_targets = []
        for var in self.output_vars:
            arr = self.ds_output[var].isel(sample=idx).values.astype(np.float32).reshape(-1)
            arr = (arr - self.stats['output'][var]['mean']) / self.stats['output'][var]['std']
            fine_targets.append(arr)
        y_high = torch.tensor(np.stack(fine_targets, axis=-1), dtype=torch.float32)

        data = HeteroData()
        data['low'].x = x_low
        data['high'].x = x_high
        data['high'].y = y_high
        data['low'].pos = torch.tensor(self.coarse_positions, dtype=torch.float32)
        data['high'].pos = torch.tensor(self.fine_positions, dtype=torch.float32)
        data[('low','to','high')].edge_index = self.edge_index_low_to_high
        data[('high','within','high')].edge_index = self.edge_index_high_within
        data.num_nodes_dict = {'low': self.n_coarse, 'high': self.n_fine}
        data.idx = idx
        data['high'].train_mask = torch.ones(self.n_fine, dtype=torch.bool)
       # data['high'].w = torch.ones(self.n_fine, dtype=torch.float32)
        data.t = torch.tensor([idx], dtype=torch.long)

        return data

