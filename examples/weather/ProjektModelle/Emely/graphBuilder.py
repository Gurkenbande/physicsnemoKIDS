import numpy as np
import torch
from torch_geometric.data import HeteroData


class BipartiteGraph(torch.utils.data.Dataset):
    def __init__(
        self,
        ds,
        device,
        coarse_shape=(8, 8),
        fine_shape=(64, 64),
        neighbors=4,
        seq_len=None,
    ):
        super().__init__()
        self.device = device
        self.seq_len = seq_len
        self.ds = ds
        self.coarse_shape = coarse_shape
        self.fine_shape = fine_shape
        self.n_coarse = coarse_shape[0] * coarse_shape[1]
        self.n_fine = fine_shape[0] * fine_shape[1]
        self.neighbors = neighbors

        # positions
        y_c = np.linspace(0, 1, coarse_shape[0])
        x_c = np.linspace(0, 1, coarse_shape[1])
        yy_c, xx_c = np.meshgrid(y_c, x_c, indexing="ij")
        self.coarse_positions = torch.tensor(
            np.stack([yy_c.flatten(), xx_c.flatten()], axis=1), dtype=torch.float32, device=self.device
        )

        y_f = np.linspace(0, 1, fine_shape[0])
        x_f = np.linspace(0, 1, fine_shape[1])
        yy_f, xx_f = np.meshgrid(y_f, x_f, indexing="ij")
        self.fine_positions = torch.tensor(
            np.stack([yy_f.flatten(), xx_f.flatten()], axis=1), dtype=torch.float32, device=self.device
        )

        self.build_edges(neighbors)

    def __len__(self):
        return len(self.ds) if self.seq_len is None else len(self.ds) - self.seq_len
    
    def _build_single_graph(self, idx):
        y, x = self.ds[idx]

        x = torch.tensor(x, dtype=torch.float32, device=self.device)
        num_channels, Hf, Wf = x.shape

        x_low = x.reshape(
            num_channels, self.coarse_shape[0], Hf // self.coarse_shape[0],
            self.coarse_shape[1], Wf // self.coarse_shape[1]
        ).mean(dim=(2, 4)).reshape(num_channels, self.n_coarse).T

        x_high = torch.zeros((self.n_fine, 4), dtype=torch.float32, device=self.device)
        y_high = torch.tensor(y.reshape(4, self.n_fine).T, dtype=torch.float32, device=self.device)

        return x_low, x_high, y_high

    def build_edges(self, k_neighbors=4):
        Hf, Wf = self.fine_shape
        Hc, Wc = self.coarse_shape
        scale_y = Hc / Hf
        scale_x = Wc / Wf

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

        self.edge_index_low_to_high = torch.tensor(np.array(edges_low_to_high).T, dtype=torch.long, device=self.device)

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

        self.edge_index_high_within = torch.tensor(np.array(edges_high_within).T, dtype=torch.long, device=self.device)

    def __getitem__(self, idx):
        if self.seq_len is None:
            x_low, x_high, y_high = self._build_single_graph(idx)
        else:
            x_lows = [self._build_single_graph(t)[0] for t in range(idx, idx + self.seq_len)]
            x_low = torch.stack(x_lows, dim=1)
            _, x_high, y_high = self._build_single_graph(idx + self.seq_len)

        data = HeteroData()
        data["low"].x = x_low
        data["high"].x = x_high
        data["high"].y = y_high
        data["low"].pos = self.coarse_positions
        data["high"].pos = self.fine_positions
        data['high'].train_mask = torch.ones(data['high'].x.shape[0], dtype=torch.bool)
        data[("low", "to", "high")].edge_index = self.edge_index_low_to_high
        data[("high", "within", "high")].edge_index = self.edge_index_high_within
        data.num_nodes_dict = {"low": self.n_coarse, "high": self.n_fine}
        if self.seq_len is not None:
            data.t = torch.tensor(idx + self.seq_len, dtype=torch.float32, device=self.device)
        else:
            data.t = torch.tensor(idx, dtype=torch.float32, device=self.device)

        return data