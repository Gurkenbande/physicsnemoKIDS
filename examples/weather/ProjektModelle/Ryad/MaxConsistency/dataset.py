from pathlib import Path
import torch
import zarr
import torch.nn.functional as F
from torch.utils.data import Dataset
from typing import Optional


class WeatherDownscalingDataset:
    
    def __init__(
        self,
        mode: str,
        device: torch.device,
        data_path_sub: str = "/tmp/cwa_dataset_3months.zarr",
        data_path_full: str = "/data/42-julia-hpc-rz-lsx/sih25nq/downscaling/CorrDiff/cwa_dataset/cwa_dataset.zarr",
        input_key: str = "era5",
        target_key: str = "cwb",
        stats_file: Optional[str] = None,
        x_mean_fixed: Optional[torch.Tensor] = None,
        x_std_fixed: Optional[torch.Tensor] = None,
        y_mean_fixed: Optional[torch.Tensor] = None,
        y_std_fixed: Optional[torch.Tensor] = None,
    ):
        assert mode in ["sub", "full"]
        self.mode = mode
        self.device = device
        self.input_key = input_key
        self.target_key = target_key

    
        self.data_path = Path(
            data_path_sub if mode == "sub" else data_path_full
        )

        if not self.data_path.exists():
            raise FileNotFoundError(f"Zarr path not found: {self.data_path}")

        
        self.root = zarr.open(self.data_path, mode="r")

        print("WeatherDownscalingDataset")
        print("Mode:", self.mode)
        print("Zarr path:", self.data_path)
        print("Available Zarr keys:", list(self.root.keys()))

        if self.input_key not in self.root:
            raise KeyError(
                f"Input key '{self.input_key}' not found. "
                f"Available keys: {list(self.root.keys())}"
            )
        if self.target_key not in self.root:
            raise KeyError(
                f"Target key '{self.target_key}' not found. "
                f"Available keys: {list(self.root.keys())}"
            )

        self.x_low = self.root[self.input_key]
        self.y_high = self.root[self.target_key]

        if self.x_low.shape[0] != self.y_high.shape[0]:
            raise ValueError(
                "Input/Target length mismatch: "
                f"{self.x_low.shape[0]} vs {self.y_high.shape[0]}"
            )

        self.length = self.x_low.shape[0]

        self.n_input_features = self.x_low.shape[1]
        self.n_output_vars = self.y_high.shape[1]
        self.coarse_shape = (56, 56)
        self.fine_shape = (448, 448)
        # self.coarse_shape = tuple(self.x_low.shape[-2:])
        # self.fine_shape = tuple(self.y_high.shape[-2:])

        print("Input shape :", self.x_low.shape)
        print("Target shape:", self.y_high.shape)
        print("Dataset size:", self.length)
        print("Input channels :", self.n_input_features)
        print("Output channels:", self.n_output_vars)
        print("Coarse shape:", self.coarse_shape)
        print("Fine shape  :", self.fine_shape)

        default_stats_file = Path(f"normalization_stats_{self.mode}.pt")
        self.stats_file = Path(stats_file) if stats_file is not None else default_stats_file

        fixed_given = all(v is not None for v in [x_mean_fixed, x_std_fixed, y_mean_fixed, y_std_fixed])
        loaded_from_file = False

        if fixed_given:
            self.x_mean = self._as_channel_tensor(x_mean_fixed, self.n_input_features, "x_mean")
            self.x_std = self._as_channel_tensor(x_std_fixed, self.n_input_features, "x_std")
            self.y_mean = self._as_channel_tensor(y_mean_fixed, self.n_output_vars, "y_mean")
            self.y_std = self._as_channel_tensor(y_std_fixed, self.n_output_vars, "y_std")
            print("\nUsing fixed normalization statistics from constructor arguments.")
        elif self.stats_file.exists():
            payload = torch.load(self.stats_file, map_location="cpu")
            self.x_mean = self._as_channel_tensor(payload["x_mean"], self.n_input_features, "x_mean")
            self.x_std = self._as_channel_tensor(payload["x_std"], self.n_input_features, "x_std")
            self.y_mean = self._as_channel_tensor(payload["y_mean"], self.n_output_vars, "y_mean")
            self.y_std = self._as_channel_tensor(payload["y_std"], self.n_output_vars, "y_std")
            loaded_from_file = True
            print(f"\nUsing fixed normalization statistics from file: {self.stats_file}")
        else:
            self._compute_channelwise_stats()
            self.stats_file.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "x_mean": self.x_mean,
                    "x_std": self.x_std,
                    "y_mean": self.y_mean,
                    "y_std": self.y_std,
                },
                self.stats_file,
            )
            print(f"\nSaved fixed normalization statistics to: {self.stats_file}")

        self.x_std = torch.clamp(self.x_std, min=1e-6)
        self.y_std = torch.clamp(self.y_std, min=1e-6)

        print(f"X stats shape: mean={tuple(self.x_mean.shape)}, std={tuple(self.x_std.shape)}")
        print(f"Y stats shape: mean={tuple(self.y_mean.shape)}, std={tuple(self.y_std.shape)}")
        if loaded_from_file:
            print("Stats source: file (fixed)")
        elif fixed_given:
            print("Stats source: constructor arguments (fixed)")
        else:
            print("Stats source: computed once and frozen to file")


    @staticmethod
    def _as_channel_tensor(values, channels: int, name: str) -> torch.Tensor:
        tensor = torch.as_tensor(values, dtype=torch.float32)
        if tensor.ndim != 1 or tensor.numel() != channels:
            raise ValueError(f"{name} must be 1D with length {channels}, got shape {tuple(tensor.shape)}")
        return tensor


    def _compute_channelwise_stats(self):
        print("\nCalculating channel-wise normalization statistics...")

        max_samples = 500
        stride = max(1, self.length // max_samples)

        x_sum = torch.zeros(self.n_input_features, dtype=torch.float64)
        x_sum_sq = torch.zeros(self.n_input_features, dtype=torch.float64)
        y_sum = torch.zeros(self.n_output_vars, dtype=torch.float64)
        y_sum_sq = torch.zeros(self.n_output_vars, dtype=torch.float64)

        x_count = torch.zeros(self.n_input_features, dtype=torch.float64)
        y_count = torch.zeros(self.n_output_vars, dtype=torch.float64)
        x_nan_count = torch.zeros(self.n_input_features, dtype=torch.float64)
        y_nan_count = torch.zeros(self.n_output_vars, dtype=torch.float64)

        used_samples = 0
        for i in range(0, self.length, stride):
            x = torch.tensor(self.x_low[i], dtype=torch.float32)
            y = torch.tensor(self.y_high[i], dtype=torch.float32)

            x_valid = torch.isfinite(x)
            y_valid = torch.isfinite(y)

            x_clean = torch.where(x_valid, x, torch.zeros_like(x))
            y_clean = torch.where(y_valid, y, torch.zeros_like(y))

            x_sum += x_clean.sum(dim=(1, 2), dtype=torch.float64)
            x_sum_sq += (x_clean * x_clean).sum(dim=(1, 2), dtype=torch.float64)
            y_sum += y_clean.sum(dim=(1, 2), dtype=torch.float64)
            y_sum_sq += (y_clean * y_clean).sum(dim=(1, 2), dtype=torch.float64)

            x_count += x_valid.sum(dim=(1, 2), dtype=torch.float64)
            y_count += y_valid.sum(dim=(1, 2), dtype=torch.float64)
            x_nan_count += (~x_valid).sum(dim=(1, 2), dtype=torch.float64)
            y_nan_count += (~y_valid).sum(dim=(1, 2), dtype=torch.float64)
            used_samples += 1

        x_count = torch.clamp(x_count, min=1.0)
        y_count = torch.clamp(y_count, min=1.0)

        self.x_mean = (x_sum / x_count).to(torch.float32)
        self.y_mean = (y_sum / y_count).to(torch.float32)

        x_var = (x_sum_sq / x_count) - self.x_mean.to(torch.float64).pow(2)
        y_var = (y_sum_sq / y_count) - self.y_mean.to(torch.float64).pow(2)

        self.x_std = torch.sqrt(torch.clamp(x_var.to(torch.float32), min=1e-12))
        self.y_std = torch.sqrt(torch.clamp(y_var.to(torch.float32), min=1e-12))

        print(f"Used samples for stats: {used_samples}")
        print(f"X NaN count (sum over channels): {int(x_nan_count.sum().item())}")
        print(f"Y NaN count (sum over channels): {int(y_nan_count.sum().item())}")
        


    def get_dataset(self):
        """Gibt ein torch.utils.data.Dataset zurück."""
        return _TorchDownscalingDataset(
            self.x_low, self.y_high,
            x_mean=self.x_mean,
            x_std=self.x_std,
            y_mean=self.y_mean,
            y_std=self.y_std
        )

    def get_feature_info(self):
        
        return {
            "n_input_features": self.n_input_features,
            "n_output_vars": self.n_output_vars,
            "coarse_shape": self.coarse_shape,
            "fine_shape": self.fine_shape,
        }

    def __len__(self):
        return self.length


class _TorchDownscalingDataset(Dataset):
    
    def __init__(self, x_low, y_high, x_mean=0, x_std=1, y_mean=0, y_std=1):
        self.x_low = x_low
        self.y_high = y_high
        self.x_mean = x_mean
        self.x_std = x_std
        self.y_mean = y_mean
        self.y_std = y_std

    def __len__(self):
        return self.x_low.shape[0]

    def __getitem__(self, idx):
        x = torch.as_tensor(self.x_low[idx], dtype=torch.float32)
        y = torch.as_tensor(self.y_high[idx], dtype=torch.float32)

        # Impute NaN/Inf mit channel-wise Mittelwerten
        x_valid = torch.isfinite(x)
        y_valid = torch.isfinite(y)
        x = torch.where(x_valid, x, self.x_mean[:, None, None])
        y = torch.where(y_valid, y, self.y_mean[:, None, None])
        
        # Verwende globale normalisierungs-stats
        x = (x - self.x_mean[:, None, None]) / self.x_std[:, None, None]
        y = (y - self.y_mean[:, None, None]) / self.y_std[:, None, None]
        
        x = F.interpolate(
            x.unsqueeze(0),     
            size=(56, 56),
            mode="bilinear",
            align_corners=False
        ).squeeze(0)
        y = y[:, :448, :448]
        return x, y
    
    
    
    
    
    
    
    
    
    
    
    
    
    """
    
    def __init__(
        self,
        mode: str,
        device: torch.device,
        neighbors: int = 4,
        seq_len: int | None = None,
    ):
        assert mode in ["sub", "full"]
        self.mode = mode
        self.device = device
        self.neighbors = neighbors
        self.seq_len = seq_len

        self._load_raw_dataset()
        self._set_resolution()
        self._build_graph_dataset()

    def _load_raw_dataset(self):
        if self.mode == "sub":
            self.data_path = Path("/tmp/data_corrdiff_3months.zarr")
            self.ds = get_zarr_dataset(data_path=self.data_path)

        elif self.mode == "full":
            self.data_path = Path("/tmp/data_corrdiff.zarr")
            self.ds = get_zarr_dataset(data_path=self.data_path)

    def _set_resolution(self):
        self.coarse_shape = (56, 56)
        self.fine_shape = (448, 448)
        self.n_input_features = 20
        self.n_output_vars = 4

    def _build_graph_dataset(self):
        self.graph_dataset = BipartiteGraph(
            ds=self.ds,
            device=self.device,
            coarse_shape=self.coarse_shape,
            fine_shape=self.fine_shape,
            neighbors=self.neighbors,
            seq_len=self.seq_len,
        )

    def get_dataset(self):
        return self.graph_dataset

    def get_feature_info(self):
        return {
            "n_input_features": self.n_input_features,
            "n_output_vars": self.n_output_vars,
            "coarse_shape": self.coarse_shape,
            "fine_shape": self.fine_shape,
        }

    def longitude(self):
        return self.ds.longitude()

    def latitude(self):
        return self.ds.latitude()

    def __len__(self):
        return len(self.graph_dataset) 

print("a")               
"""