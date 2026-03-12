import os
import torch
import torch.multiprocessing as mp
from pathlib import Path
import sys
from torch.utils.data import Subset
import torch.nn as nn
from torch_geometric.loader import DataLoader
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

notebook_dir = Path(__file__).resolve().parent if "__file__" in globals() else Path().resolve()
sys.path.insert(0, "/home/s448562/physicsnemoKIDS/")

from gnn4cd_model import GNN4CD_Model
from examples.weather.ProjektModelle.Emely.graphBuilder import BipartiteGraph
from examples.weather.ProjektModelle.Emely.train_test2 import Trainer, Tester
from examples.weather.corrdiff.datasets.cwb import get_zarr_dataset
from examples.weather.corrdiff.datasets.hrrrmini import HRRRMiniDataset

if __name__ == "__main__":
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    is_main = (local_rank == 0)

    if not is_main:
        os.environ["WANDB_MODE"] = "disabled"
    
    print(f"Device: {device}", flush=True)

    d = "subset"

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
        neighbors=15, 
        seq_len=None,
        n_neighbors=8,  
    )

    model = GNN4CD_Model(
        encoding_dim=128,
        n_input_features=n_input_features,
        h_hid=64,
        n_layers=4,
        high_in=4,
        low2high_out=64,
        high_out=64,
        n_output_vars=4,
    ).to(device)

    model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)

    args = type("", (), {})()
    args.epochs = 60
    args.output_path = str(notebook_dir / "output")
    args.model_type = "Rall"

    N = len(dataset)
    train_ds = Subset(dataset, range(0, int(0.8 * N)))
    val_ds   = Subset(dataset, range(int(0.8 * N), int(0.9 * N)))
    test_ds  = Subset(dataset, range(int(0.9 * N), N))

    train_sampler = DistributedSampler(train_ds, shuffle=True)
    val_sampler   = DistributedSampler(val_ds, shuffle=False)

    train_loader = DataLoader(train_ds, batch_size=6, sampler=train_sampler, num_workers=16, persistent_workers=True)
    val_loader   = DataLoader(val_ds, batch_size=6, sampler=val_sampler, num_workers=16, persistent_workers=True)
    test_loader  = DataLoader(test_ds, batch_size=6, shuffle=False, num_workers=16, persistent_workers=True)

    trainer = Trainer()
    tester = Tester(dataset=ds)

    optimizer = torch.optim.Adam(model.parameters(), lr=5e-3)
    loss_fn = nn.MSELoss()
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=4)

    checkpoint_path = Path(args.output_path) / "checkpoint.pt"
    final_model_path = Path(args.output_path) / "final_model.pt"
    start_epoch = 0
    do_train = True

    if final_model_path.exists():
        if is_main:
            print("testing only", flush=True)
        ckpt = torch.load(final_model_path, map_location=device)
        model.module.load_state_dict(ckpt["model_state_dict"])
        do_train = False

    elif checkpoint_path.exists():
        if is_main:
            print("going on with training", flush=True)
        ckpt = torch.load(checkpoint_path, map_location=device)
        model.module.load_state_dict(ckpt["model_state_dict"])
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

        if is_main:
            final_model_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {"model_state_dict": model.module.state_dict()},
                final_model_path,
            )

    dist.barrier()

    if is_main:
        lon = ds.longitude()
        lat = ds.latitude()

        if lon.ndim == 2:
            lon = lon[: fine_shape[0], : fine_shape[1]]
            lat = lat[: fine_shape[0], : fine_shape[1]]
        else:
            lon = lon[: fine_shape[1]]
            lat = lat[: fine_shape[0]]

        tester.test(
            model=model.module,   
            dataloader=test_loader,
            loss_fn=loss_fn,
            output_path=args.output_path,
            lon=lon,
            lat=lat,
        )

        print("finittttooo", flush=True)

    dist.destroy_process_group()

#scp -r s448562@julia2.hpc.uni-wuerzburg.de:/home/s448562/physicsnemoKIDS/examples/weather/ProjektModelle/Emely/output .
#rsync  -r ../physicsnemoKIDS julia:~/. 

#cp -r \
#/home/s448562/LAB4/data_corrdiff_3months.zarr \
#/tmp/data_corrdiff_3months.zarr
#SBATCH --qos=normal