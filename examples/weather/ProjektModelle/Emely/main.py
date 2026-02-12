import torch
import torch.multiprocessing as mp
from pathlib import Path
import sys
from torch.utils.data import Subset
import torch.nn as nn
from torch_geometric.loader import DataLoader

notebook_dir = Path(__file__).resolve().parent if "__file__" in globals() else Path().resolve()
sys.path.insert(0, "/home/s448562/physicsnemoKIDS/")

from gnn4cd_model import GNN4CD_Model
from examples.weather.ProjektModelle.Emely.graphBuilder import BipartiteGraph
from examples.weather.ProjektModelle.Emely.train_test import Trainer, Tester
from examples.weather.corrdiff.datasets.cwb import get_zarr_dataset
from examples.weather.corrdiff.datasets.hrrrmini import HRRRMiniDataset

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device, flush=True)

    d = "subset"  # "mini", "subset", "full"

    if d == "mini":
        stats_path = "/home/s448562/LAB4/data_corrdiff_mini/stats.json"
        data_path = "/home/s448562/LAB4/data_corrdiff_mini/hrrr_mini_train.nc"
        ds = HRRRMiniDataset(data_path=data_path, stats_path=stats_path)
        fine_shape = (64, 64)
        coarse_shape = (8, 8)
        n_input_features = 28

    elif d == "subset":
        data_path = Path("/tmp/data_corrdiff_3months.zarr").expanduser()
        ds = get_zarr_dataset(data_path=data_path)
        fine_shape = (448, 448)
        coarse_shape = (56, 56)
        n_input_features = 20

    elif d == "full":
        data_path = Path("/tmp/data_corrdiff.zarr").expanduser()
        ds = get_zarr_dataset(data_path=data_path)
        fine_shape = (448, 448)
        coarse_shape = (56, 56)
        n_input_features = 20

    dataset = BipartiteGraph(
        ds=ds,
        device="cpu",
        coarse_shape=coarse_shape,
        fine_shape=fine_shape,
        neighbors=6,
        seq_len=None,
    )

    model = GNN4CD_Model(
        encoding_dim=128,
        n_input_features=n_input_features,
        h_hid=64,
        n_layers=3,
        high_in=4,
        low2high_out=64,
        high_out=64,
        n_output_vars=4,
    ).to(device)

    args = type("", (), {})()
    args.epochs = 40                 
    args.output_path = str(notebook_dir / "output")
    args.model_type = "Rall"

    N = len(dataset)
    train_ds = Subset(dataset, range(0, int(0.8 * N)))
    val_ds   = Subset(dataset, range(int(0.8 * N), int(0.9 * N)))
    test_ds  = Subset(dataset, range(int(0.9 * N), N))

    train_loader = DataLoader(train_ds, batch_size=4, shuffle=True,num_workers=32, persistent_workers=True)
    val_loader   = DataLoader(val_ds, batch_size=4, shuffle=False,num_workers=32, persistent_workers=True)
    test_loader  = DataLoader(test_ds, batch_size=4, shuffle=False,num_workers=32, persistent_workers=True)

    trainer = Trainer()
    tester = Tester(dataset=dataset)

    optimizer = torch.optim.Adam(model.parameters(), lr=5e-3)
    loss_fn = nn.MSELoss()
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=4)

    checkpoint_path = Path(args.output_path) / "checkpoint.pt"
    final_model_path = Path(args.output_path) / "final_model.pt"
    start_epoch = 0

    do_train = True

    if final_model_path.exists():
        print("testing only", flush=True)
        ckpt = torch.load(final_model_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        do_train = False

    elif checkpoint_path.exists():
        print("going on with training", flush=True)
        ckpt = torch.load(checkpoint_path, map_location=device)

        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        lr_scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        start_epoch = ckpt["epoch"]


    if do_train:
        trainer.train_R_Rall(
            model=model,
            dataloader_train=train_loader,
            dataloader_val=val_loader,
            optimizer=optimizer,
            loss_fn=loss_fn,
            lr_scheduler=lr_scheduler,
            args=args,
            epoch_start=start_epoch,
        )

        final_model_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {"model_state_dict": model.state_dict()},
            final_model_path
        )

    lon = ds.longitude()
    lat = ds.latitude()

    if lon.ndim == 2:
        lon = lon[: fine_shape[0], : fine_shape[1]]
        lat = lat[: fine_shape[0], : fine_shape[1]]
    else:
        lon = lon[: fine_shape[1]]
        lat = lat[: fine_shape[0]]

    tester.test(
        model=model,
        dataloader=test_loader,
        loss_fn=loss_fn,
        output_path=args.output_path,
        lon=lon,
        lat=lat,
    )

    print("finittttooo", flush=True)
