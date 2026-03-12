import numpy as np
import torch
from torch_geometric.data import HeteroData

class BipartiteGraph(torch.utils.data.Dataset):
    def __init__(self, ds, device, coarse_shape=(8, 8), fine_shape=(64, 64), neighbors=4, seq_len=None, n_neighbors=1):
      
        super().__init__()
        self.device = device
        self.seq_len = seq_len
        self.ds = ds
        self.coarse_shape = coarse_shape
        self.fine_shape = fine_shape
        self.n_coarse = coarse_shape[0] * coarse_shape[1]
        self.n_fine = fine_shape[0] * fine_shape[1]
        self.neighbors = neighbors
        self.n_neighbors = n_neighbors

        y_c = np.linspace(0, 1, coarse_shape[0])
        x_c = np.linspace(0, 1, coarse_shape[1])
        yy_c, xx_c = np.meshgrid(y_c, x_c, indexing="ij")
        self.coarse_positions = torch.tensor(np.stack([yy_c.flatten(), xx_c.flatten()], axis=1),dtype=torch.float32, device=self.device)

        y_f = np.linspace(0, 1, fine_shape[0])
        x_f = np.linspace(0, 1, fine_shape[1])
        yy_f, xx_f = np.meshgrid(y_f, x_f, indexing="ij")
        self.fine_positions = torch.tensor(np.stack([yy_f.flatten(), xx_f.flatten()], axis=1),dtype=torch.float32, device=self.device)
        self.x_high_template = torch.zeros((self.n_fine, 4),dtype=torch.float32,device=self.device)

        self.build_edges()

    def __len__(self):
        return len(self.ds) if self.seq_len is None else len(self.ds) - self.seq_len

    def _build_single_graph(self, idx):
        y, x = self.ds[idx] 
        x = torch.as_tensor(x, dtype=torch.float32, device=self.device)
        y = torch.as_tensor(y, dtype=torch.float32, device=self.device)
        num_channels, Hf, Wf = x.shape
        Hc, Wc = self.coarse_shape
        x_low = (x.reshape(num_channels, Hc, Hf // Hc, Wc, Wf // Wc).mean(dim=(2, 4)).permute(1, 2, 0).reshape(self.n_coarse, num_channels))
        x_high = self.x_high_template.clone()
        y_high = y.reshape(4, self.n_fine).permute(1, 0)

        return x_low, x_high, y_high


    def build_edges(self):
        with torch.no_grad():
            # low to high edges
            dist = torch.cdist(self.fine_positions, self.coarse_positions)
            knn = dist.topk(k=self.neighbors, largest=False).indices
            high_idx = torch.arange(self.n_fine, device=self.device).repeat_interleave(self.neighbors)
            low_idx = knn.reshape(-1)
            self.edge_index_low_to_high = torch.stack([low_idx, high_idx], dim=0)

            # fine to fine edges
            Hf, Wf = self.fine_shape
            hi, wi = torch.meshgrid(torch.arange(Hf), torch.arange(Wf), indexing="ij")
            hi = hi.flatten()
            wi = wi.flatten()

            max_radius = max(Hf, Wf)
            all_offsets = [
                (dh, dw)
                for dh in range(-max_radius, max_radius + 1)
                for dw in range(-max_radius, max_radius + 1)
                if not (dh == 0 and dw == 0)
            ]
            all_offsets.sort(key=lambda d: d[0]**2 + d[1]**2)

            assert self.n_neighbors % 2 == 0
            n_pairs = self.n_neighbors // 2

            selected = []
            seen = set()
            for dh, dw in all_offsets:
                if (dh, dw) in seen:
                    continue
                selected.append((dh, dw))
                selected.append((-dh, -dw))
                seen.add((dh, dw))
                seen.add((-dh, -dw))
                if len(selected) // 2 >= n_pairs:
                    break

            edges = []
            for dh, dw in selected:
                nh = hi + dh
                nw = wi + dw
                mask = (0 <= nh) & (nh < Hf) & (0 <= nw) & (nw < Wf)
                src = hi[mask] * Wf + wi[mask]
                dst = nh[mask] * Wf + nw[mask]
                edges.append(torch.stack([src, dst]))

            self.edge_index_high_within = torch.cat(edges, dim=1).to(self.device)


    def __getitem__(self, idx):
        if self.seq_len is None:
            x_low, x_high, y_high = self._build_single_graph(idx)
        else:
            x_lows = []
            for t in range(idx, idx + self.seq_len):
                x_low_t, _, _ = self._build_single_graph(t)
                x_lows.append(x_low_t)

            x_low = torch.stack(x_lows, dim=1)
            _, x_high, y_high = self._build_single_graph(idx + self.seq_len)

        data = HeteroData()
        data["low"].x = x_low
        data["high"].x = x_high
        data["high"].y = y_high
        data["low"].pos = self.coarse_positions
        data["high"].pos = self.fine_positions
        data['high'].train_mask = torch.ones(data['high'].x.shape[0], dtype=torch.bool, device=self.device)
        data[("low", "to", "high")].edge_index = self.edge_index_low_to_high
        data[("high", "within", "high")].edge_index = self.edge_index_high_within
        data.num_nodes_dict = {"low": self.n_coarse, "high": self.n_fine}
        data.t = torch.tensor(idx+self.seq_len if self.seq_len else idx, dtype=torch.float32, device=self.device)
        return data