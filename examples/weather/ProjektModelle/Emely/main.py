"""
Main training script using Graph Neural Networks.
This script handles distributed training, dataset loading, model training, and evaluation.
"""

import os
import hydra
from omegaconf import DictConfig
import torch
from pathlib import Path
import sys
from torch.utils.data import Subset
import torch.nn as nn
from torch_geometric.loader import DataLoader
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import numpy as np
from examples.weather.ProjektModelle.Emely.losse import MSEGradientLoss, QuantizedLoss


possible_paths = [
    "/home/s448562/physicsnemoKIDS/",
    "/LAB4/physicsnemoKIDS/",
    "/Users/emely/Uni/Lab4/physicsnemoKIDS/",
    str(Path(__file__).parent / "physicsnemoKIDS")
]

for p in possible_paths:
    if os.path.exists(p):
        sys.path.insert(0, p)
        print(f"Using physicsnemoKIDS at: {p}")
        break
else:
    raise FileNotFoundError("physicsnemoKIDS/ nicht gefunden!")

from gnn4cd_model import GNN4CD_Model
from examples.weather.ProjektModelle.Emely.graphBuilder import BipartiteGraph
from examples.weather.ProjektModelle.Emely.train_test2 import Trainer, Tester
from examples.weather.corrdiff.datasets.cwb import get_zarr_dataset
from examples.weather.corrdiff.datasets.hrrrmini import HRRRMiniDataset


@hydra.main(version_base=None, config_path=".", config_name="config")
def main(cfg: DictConfig):
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    n_gpus = int(os.environ.get("WORLD_SIZE", 1))
    
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cpu")
        local_rank = 0
        n_gpus = 1
    
    is_main = (local_rank == 0)
    
    if n_gpus > 1 and torch.cuda.is_available():
        dist.init_process_group(backend=cfg.distributed.backend)
    if not is_main:
        os.environ["WANDB_MODE"] = cfg.wandb.mode

    # Load dataset
    if cfg.dataset.type == "mini":
        ds = HRRRMiniDataset(data_path=cfg.dataset.data_path, stats_path=cfg.dataset.stats_path)
        fine_shape = tuple(cfg.dataset.fine_shape)
        coarse_shape = tuple(cfg.dataset.coarse_shape)
        n_input_features = cfg.dataset.n_input_features
        
    elif cfg.dataset.type in ["subset", "full"]:
        ds = get_zarr_dataset(data_path=Path(cfg.dataset.data_path))
        fine_shape = tuple(cfg.dataset.fine_shape)
        coarse_shape = tuple(cfg.dataset.coarse_shape)
        n_input_features = cfg.dataset.n_input_features
       
    print("Dataset: ", cfg.dataset.type, flush=True)
    
    # Create bipartite graph dataset 
    dataset = BipartiteGraph(ds=ds, device="cpu", coarse_shape=coarse_shape, fine_shape=fine_shape, neighbors=cfg.dataset.neighbors, seq_len=cfg.dataset.seq_len, n_neighbors=cfg.dataset.n_neighbors)

    # GNN model
    model = GNN4CD_Model(
        encoding_dim=cfg.model.encoding_dim,
        n_input_features=n_input_features,
        h_hid=cfg.model.h_hid,
        n_layers=cfg.model.n_layers,
        high_in=cfg.model.high_in,
        low2high_out=cfg.model.low2high_out,
        high_out=cfg.model.high_out,
        n_output_vars=cfg.model.n_output_vars
    ).to(device)

    if n_gpus > 1:
        model = DDP(
            model, 
            device_ids=[local_rank], 
            find_unused_parameters=cfg.distributed.find_unused_parameters
        )

    SEED = cfg.seed
    torch.manual_seed(SEED)
    
    # Split into train/val/test
    N = len(dataset)
    train_end = int(cfg.trainer.splits.train * N)
    val_end = int((cfg.trainer.splits.train + cfg.trainer.splits.val) * N)

    train_idx = np.arange(0, train_end)
    val_idx   = np.arange(train_end, val_end)
    test_idx  = np.arange(val_end, N)

    train_ds = Subset(dataset, train_idx)
    val_ds = Subset(dataset, val_idx)
    test_ds = Subset(dataset, test_idx)

    train_sampler = (DistributedSampler(train_ds, shuffle=True, seed=SEED) if n_gpus > 1 else None)
    val_sampler = (DistributedSampler(val_ds, shuffle=False) if n_gpus > 1 else None)

    train_loader = DataLoader(train_ds, batch_size=cfg.trainer.batch_size, sampler=train_sampler,num_workers=cfg.dataloader.num_workers,persistent_workers=cfg.dataloader.persistent_workers,pin_memory=cfg.dataloader.pin_memory)
    val_loader = DataLoader(val_ds, batch_size=cfg.trainer.batch_size, sampler=val_sampler,num_workers=cfg.dataloader.num_workers,persistent_workers=cfg.dataloader.persistent_workers,pin_memory=cfg.dataloader.pin_memory)
    test_loader = DataLoader(test_ds, batch_size=cfg.trainer.batch_size, shuffle=False,num_workers=cfg.dataloader.num_workers,persistent_workers=cfg.dataloader.persistent_workers,pin_memory=cfg.dataloader.pin_memory)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.trainer.lr)
    
    # alternative loss fcts
    loss_fn = MSEGradientLoss(fine_shape=fine_shape,alpha=1.5)
    #loss_fn= QuantizedLoss(n_bins=10)
    #loss_fn = nn.MSELoss()
    
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=cfg.trainer.scheduler.factor,patience=cfg.trainer.scheduler.patience)

    trainer_obj = Trainer(dataset=ds)
    tester = Tester(dataset=ds)

    # output paths and checkpoint 
    output_path = Path(cfg.trainer.output_path).resolve()
    checkpoint_path = output_path / "checkpoint.pt"
    final_model_path = output_path / "final_model.pt"
    start_epoch = 0
    do_train = True

    # Load final model if it exists (skip training)
    if final_model_path.exists() and is_main:
        print("Testing only", flush=True)
        ckpt = torch.load(final_model_path, map_location=device)

        state_dict = ckpt["model_state_dict"]
        if any(k.startswith("module.") for k in state_dict.keys()):
            state_dict = {k.removeprefix("module."): v for k, v in state_dict.items()}
        if hasattr(model, 'module'):
            model.module.load_state_dict(state_dict)
        else:
            model.load_state_dict(state_dict)
        do_train = False

    # Load checkpoint if it exists (resume training)
    elif checkpoint_path.exists() and is_main:
        print("Resuming training", flush=True)
        ckpt = torch.load(checkpoint_path, map_location=device)
        state_dict = ckpt["model_state_dict"]
        if any(k.startswith("module.") for k in state_dict.keys()):
            state_dict = {k.removeprefix("module."): v for k, v in state_dict.items()}
        if hasattr(model, 'module'):
            model.module.load_state_dict(state_dict)
        else:
            model.load_state_dict(state_dict)
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])     
        lr_scheduler.load_state_dict(ckpt["scheduler_state_dict"])   
        start_epoch = ckpt.get("epoch", 0) 

    target_channels = [0] # which output channels to predict

    # Training phase
    if do_train:
        trainer_obj.train_R_Rall(
            model=model,
            dataloader_train=train_loader,
            dataloader_val=val_loader,
            optimizer=optimizer,
            loss_fn=loss_fn,
            lr_scheduler=lr_scheduler,
            args=cfg.trainer,  
            epoch_start=start_epoch,
            target_channels=target_channels
        )
        
        # Save final model
        if is_main:
            output_path.mkdir(parents=True, exist_ok=True)
            
            state_dict = model.module.state_dict() if hasattr(model, 'module') else model.state_dict()
            torch.save({"model_state_dict": state_dict, "epoch": cfg.trainer.epochs}, output_path / "final_model.pt") 

    if n_gpus > 1:
        dist.barrier()

    # Testing 
    if is_main:
        lon = ds.longitude()
        lat = ds.latitude()
        
        if lon.ndim == 2:
            lon = lon[:fine_shape[0], :fine_shape[1]]
            lat = lat[:fine_shape[0], :fine_shape[1]]
        else:
            lon = lon[:fine_shape[1]]
            lat = lat[:fine_shape[0]]
        
        tester.test(
            model=model.module if hasattr(model, 'module') else model,
            dataloader=test_loader,
            loss_fn=loss_fn,
            output_path=str(output_path),
            lon=lon,
            lat=lat,
            coarse_shape=coarse_shape,
            fine_shape=fine_shape,
            target_channels=target_channels
        )
        print("finittttooo", flush=True)

    if n_gpus > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
