import torch
import torch.multiprocessing as mp
from pathlib import Path
import sys
import numpy as np
from torch.utils.data import Subset

notebook_dir = Path(__file__).resolve().parent if "__file__" in globals() else Path().resolve()
ROOT = notebook_dir.parent
sys.path.insert(0, "/home/s448562/physicsnemoKIDS/")  


from gnn4cd_model import GNN4CD_Model
from examples.weather.corrdiff.datasets.hrrrmini import HRRRMiniDataset
from graphBuilder2 import BipartiteGraph
from train_test3 import Trainer, Tester
from torch_geometric.loader import DataLoader
import torch.nn as nn
from examples.weather.corrdiff.datasets.cwb import get_zarr_dataset


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "mps")
    print("Device: ", device, flush=True)

    d = "subset"  # "mini", "subset", "full"

    if d == "mini":
        stats_path = "/home/s448562/LAB4/data_corrdiff_mini/stats.json"
        data_path = "/home/s448562/LAB4/data_corrdiff_mini/hrrr_mini_train.nc"

        ds = HRRRMiniDataset(data_path=data_path, stats_path=stats_path)

    elif d == "subset":
       # data_path = Path("/home/s448562/LAB4/data_corrdiff_3months.zarr").expanduser()
        data_path = Path("/tmp/data_corrdiff_3months.zarr").expanduser()    
        ds = get_zarr_dataset(data_path=data_path)
     
    elif d == "full":
        data_path = Path("/tmp/data_corrdiff.zarr").expanduser()
        ds = get_zarr_dataset(data_path=data_path)
    
    print("length: ", len(ds), flush=True)
    print("Dataset: ", d, flush=True)

    if d == "mini":
        fine_shape = (64,64)
        n_input_features = 28   
        coarse_shape = (8,8)
    else:
        fine_shape = (448,448)
        n_input_features = 20
        coarse_shape = (56,56)

    dataset = BipartiteGraph(
        ds = ds,
        device = device,
        coarse_shape=coarse_shape,
        fine_shape=fine_shape,
        neighbors=4,
        seq_len=None  ##########!!!!!!!!!
    )

    model = GNN4CD_Model(
        encoding_dim=64,
        n_input_features=n_input_features,
        h_hid=32,
        n_layers=2,
        high_in=4,
        low2high_out=32,
        high_out=32,
        n_output_vars=4
    ).to(device)

    args = type('', (), {})()
    args.epochs = 5
    args.alpha = 0.75
    BASE_DIR = Path(__file__).resolve().parent
    args.output_path = str(BASE_DIR / "output")
    args.loss_fn = "MSE"
    args.model_type = "Rall"

    N = len(dataset)
    train_dataset = Subset(dataset, range(0, int(0.8*N)))
    val_dataset   = Subset(dataset, range(int(0.8*N), int(0.9*N)))
    test_dataset  = Subset(dataset, range(int(0.9*N), N))

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=3, pin_memory=False,persistent_workers=True)
    val_loader   = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=2, pin_memory=False,persistent_workers=True)
    test_loader  = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=2, pin_memory=False,persistent_workers=True)
    # vlt batchsize erhöhen

    trainer = Trainer()
    tester = Tester()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
    loss_fn = nn.MSELoss()

    args.lr_scheduler = "ReduceLROnPlateau"
    print("lr_scheduler:", args.lr_scheduler, flush=True)
    args.step_size = 2 

    if args.lr_scheduler == "StepLR":
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=0.1)
    elif args.lr_scheduler == "ReduceLROnPlateau":
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    elif args.lr_scheduler == "CosineAnnealingLR":
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    else:
        lr_scheduler = None


    trainer.train_R_Rall(
        model=model,
        dataloader_train=train_loader,
        dataloader_val=val_loader,
        optimizer=optimizer,
        loss_fn=loss_fn,
        lr_scheduler=lr_scheduler,
        args=args
    )

    lon = ds.longitude()
    lat = ds.latitude()

    preds, targets, test_loss = tester.test(
        model=model,
        dataloader=test_loader,
        loss_fn=loss_fn,
        output_path=args.output_path,
        lon=lon,
        lat=lat
    )

    print("finittoooo", flush=True)

# download plots: scp -r s448562@julia2.hpc.uni-wuerzburg.de:/home/s448562/physicsnemoKIDS/examples/weather/ProjektModelle/Emely/output .

