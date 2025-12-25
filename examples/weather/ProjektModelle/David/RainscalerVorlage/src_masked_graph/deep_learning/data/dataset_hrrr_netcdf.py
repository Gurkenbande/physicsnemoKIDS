import numpy as np
import torch
import torch.utils.data as data
import xarray as xr


class DatasetHRRRNetCDF(data.Dataset):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.nc_path = opt["dataroot_nc"]

        self.input_group = opt.get("input_group", "input")
        self.output_group = opt.get("output_group", "output")

        self.input_vars = list(opt["input_vars"])
        self.output_vars = list(opt["output_vars"])

        self.sample_dim = opt.get("sample_dim", "sample")
        self.coord_var = opt.get("coord_var", "coord")

        self.nan_to_num = opt.get("nan_to_num", None)
        self.normalize = bool(opt.get("normalize", False))

        self.input_mean = opt.get("input_mean", None)
        self.input_std = opt.get("input_std", None)
        self.output_mean = opt.get("output_mean", None)
        self.output_std = opt.get("output_std", None)

        self.engine = opt.get("engine", "h5netcdf")

        self.ds_in = xr.open_dataset(self.nc_path, group=self.input_group, engine=self.engine)
        self.ds_out = xr.open_dataset(self.nc_path, group=self.output_group, engine=self.engine)
        self.ds_meta = xr.open_dataset(self.nc_path, engine=self.engine)

        limit = opt.get("limit_samples", None)
        n = int(self.ds_in.sizes[self.sample_dim])
        if limit is not None:
            n = min(n, int(limit))
        self._n = n

        if self.normalize:
            if any(v is None for v in (self.input_mean, self.input_std, self.output_mean, self.output_std)):
                raise ValueError("normalize=True needs input_mean/input_std/output_mean/output_std")
            self.im = np.asarray(self.input_mean, dtype=np.float32).reshape(-1, 1, 1)
            self.istd = np.asarray(self.input_std, dtype=np.float32).reshape(-1, 1, 1)
            self.om = np.asarray(self.output_mean, dtype=np.float32).reshape(-1, 1, 1)
            self.ostd = np.asarray(self.output_std, dtype=np.float32).reshape(-1, 1, 1)
        else:
            self.im = self.istd = self.om = self.ostd = None

        self.has_coord = self.coord_var in self.ds_meta

    def __len__(self):
        return self._n

    def _stack_one(self, ds, vars, idx):
        a = ds[vars].isel({self.sample_dim: idx}).to_array(dim="channel").values.astype(np.float32)
        if self.nan_to_num is not None:
            np.nan_to_num(a, copy=False, nan=self.nan_to_num, posinf=None, neginf=None)
        return a

    def __getitem__(self, index):
        L = self._stack_one(self.ds_in, self.input_vars, index)
        H = self._stack_one(self.ds_out, self.output_vars, index)

        if self.normalize:
            L = (L - self.im) / (self.istd + 1e-8)
            H = (H - self.om) / (self.ostd + 1e-8)

        sample = {
            "L": torch.from_numpy(L),
            "H": torch.from_numpy(H),
            "L_path": self.nc_path,
            "H_path": self.nc_path,
        }

        if self.has_coord:
            c = self.ds_meta[self.coord_var].isel({self.sample_dim: index}).values
            sample["coord"] = torch.as_tensor(c)

        return sample
