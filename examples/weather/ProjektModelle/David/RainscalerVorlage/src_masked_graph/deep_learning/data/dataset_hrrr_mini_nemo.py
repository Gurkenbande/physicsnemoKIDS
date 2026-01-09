import torch
import torch.utils.data as data


class DatasetHRRRMiniNeMo(data.Dataset):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt

        data_path = opt["dataroot_nc"]
        stats_path = opt["stats_path"]

        input_vars = opt.get("input_vars", None)
        output_vars = opt.get("output_vars", None)
        invariant_vars = opt.get("invariant_vars", ("elev_mean", "lsm_mean"))

        from examples.weather.corrdiff.datasets.hrrrmini import HRRRMiniDataset

        self.ds = HRRRMiniDataset(
            data_path=data_path,
            stats_path=stats_path,
            input_variables=input_vars,
            output_variables=output_vars,
            invariant_variables=invariant_vars,
        )

        self.data_path = data_path

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, index):
        y, x = self.ds[index]

        sample = {
            "L": torch.from_numpy(x).float(),
            "H": torch.from_numpy(y).float(),
            "L_path": self.data_path,
            "H_path": self.data_path,
        }

        return sample
