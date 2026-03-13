import os
import argparse
import torch
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.data import DataLoader, random_split
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from modelupdate import Consistency
from basis import ConsistencyLoss
from dataset import WeatherDownscalingDataset
from configurationupdate import Config
from loss import LPIPSLoss

import wandb

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import Callback, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

def masked_mse(pred, target):
    mask = torch.isfinite(target)
    
    if mask.sum() == 0:
        return None  # falls kompletter Batch ungültig
    
    diff = (pred - target) ** 2
    return diff[mask].mean()


def split_dataset(ds, train_ratio=0.9, val_ratio=0.05):
    n = len(ds)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    n_test = n - n_train - n_val
    return random_split(ds, [n_train, n_val, n_test])


def unpack_batch(batch, device):
    if isinstance(batch, (tuple, list)) and len(batch) == 2:
        x, y = batch
        return x.to(device), y.to(device)

    if isinstance(batch, dict):
        if "x" in batch and "y" in batch:
            return batch["x"].to(device), batch["y"].to(device)
        if "x_low" in batch and "y" in batch:
            return batch["x_low"].to(device), batch["y"].to(device)

    if hasattr(batch, "__dict__"):
        for x_key, y_key in [("x_low", "y"), ("x", "y"), ("x_low", "y_high"), ("x", "y_high")]:
            if hasattr(batch, x_key) and hasattr(batch, y_key):
                x = getattr(batch, x_key)
                y = getattr(batch, y_key)
                return x.to(device), y.to(device)

    raise TypeError(f"Unbekanntes Batch-Format: {type(batch)}")


class PlotCallback(Callback):
    def __init__(self, train_loader, val_loader, device, plot_every_n_epochs=3):
        self.train_dataset = train_loader.dataset
        self.val_dataset = val_loader.dataset
        self.batch_size = train_loader.batch_size
        self.device = device
        self.plot_every_n_epochs = plot_every_n_epochs

    def on_validation_epoch_end(self, trainer, pl_module):
        epoch = trainer.current_epoch
        if epoch % self.plot_every_n_epochs != 0:
            return

        pl_module.eval()

        # Hole normalisierungs-stats
        y_mean = getattr(pl_module, 'y_mean', 0.0)
        y_std = getattr(pl_module, 'y_std', 1.0)

        def get_channel_stats(channel_idx: int):
            if torch.is_tensor(y_mean):
                mean_val = float(y_mean[channel_idx].item()) if y_mean.ndim > 0 else float(y_mean.item())
            else:
                mean_val = float(y_mean)

            if torch.is_tensor(y_std):
                std_val = float(y_std[channel_idx].item()) if y_std.ndim > 0 else float(y_std.item())
            else:
                std_val = float(y_std)

            return mean_val, std_val
    
        plot_train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size,
                                       shuffle=True, num_workers=0)
        plot_val_loader   = DataLoader(self.val_dataset,   batch_size=self.batch_size,
                                       shuffle=False, num_workers=0)

        def build_channel_figure(y_true_batch, y_pred_batch, split_name: str):
            n_channels = y_true_batch.shape[1]
            fig, axes = plt.subplots(n_channels, 3, figsize=(12, 4 * n_channels), squeeze=False)

            for ch in range(n_channels):
                y_mean_ch, y_std_ch = get_channel_stats(ch)

                y_true_np = (y_true_batch[0, ch].detach().cpu().float() * y_std_ch + y_mean_ch).numpy()
                y_pred_np = (y_pred_batch[0, ch].detach().cpu().float() * y_std_ch + y_mean_ch).numpy()
                y_diff_np = y_pred_np - y_true_np

                axes[ch, 0].set_title(f"{split_name} GT ch{ch}")
                im0 = axes[ch, 0].imshow(y_true_np)
                fig.colorbar(im0, ax=axes[ch, 0], fraction=0.046, pad=0.04)

                axes[ch, 1].set_title(f"{split_name} Pred ch{ch}")
                im1 = axes[ch, 1].imshow(y_pred_np)
                fig.colorbar(im1, ax=axes[ch, 1], fraction=0.046, pad=0.04)

                axes[ch, 2].set_title(f"{split_name} Diff ch{ch}")
                im2 = axes[ch, 2].imshow(y_diff_np)
                fig.colorbar(im2, ax=axes[ch, 2], fraction=0.046, pad=0.04)

            fig.tight_layout()
            return fig

        with torch.no_grad():
            try:
                x_tv, y_tv = unpack_batch(next(iter(plot_train_loader)), self.device)
                y_pred_tv, _ = pl_module.sample_conditional(
                    conditioning=x_tv,
                    x_image_size=448,
                    y_image_size=448,
                    steps=20,
                    use_ema=True,
                )

                fig_train = build_channel_figure(y_tv, y_pred_tv, "Train")
                trainer.logger.experiment.log({"Train Prediction vs GT (all channels)": wandb.Image(fig_train), "epoch": epoch})
                plt.close(fig_train)
            except Exception as e:
                print(f"Train plot failed (epoch {epoch}):", e)

            try:
                x_vis, y_vis = unpack_batch(next(iter(plot_val_loader)), self.device)
                y_pred_vis, _ = pl_module.sample_conditional(
                    conditioning=x_vis,
                    x_image_size=448,
                    y_image_size=448,
                    steps=20,
                    use_ema=True,
                )

                fig_val = build_channel_figure(y_vis, y_pred_vis, "Val")
                trainer.logger.experiment.log({"Val Prediction vs GT (all channels)": wandb.Image(fig_val), "epoch": epoch})
                plt.close(fig_val)
            except Exception as e:
                print(f"Val plot failed (epoch {epoch}):", e)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["sub", "full"], default="sub")
    parser.add_argument("--output_dir", default="/home/s458614/climate_project/outputs_corrdiff")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--model_channels", type=int, default=64)
    parser.add_argument("--num_steps", type=int, default=5)
    parser.add_argument("--train_ratio", type=float, default=0.9)
    parser.add_argument("--val_ratio", type=float, default=0.05)
    args = parser.parse_args()

    wandb.init(
        
        project="climate-project",
        entity="riyad-waez-universit-t-w-rzburg",
        config=vars(args)
    )
    


    wandb.define_metric("epoch")
    wandb.define_metric("train_loss", step_metric="epoch")
    wandb.define_metric("val_loss", step_metric="epoch")
    wandb.define_metric("test_loss", step_metric="epoch")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "checkpoints"), exist_ok=True)

    wd = WeatherDownscalingDataset(
        mode=args.mode,
        device=device,
    )

    ds = wd.get_dataset()
    info = wd.get_feature_info()

    train_ds, val_ds, test_ds = split_dataset(ds, args.train_ratio, args.val_ratio)

    
    workers = args.num_workers

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=workers,
        pin_memory=True,
        persistent_workers=False,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=True,
        persistent_workers=False,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=True,
        persistent_workers=False,
    )
    

    config = Config(
    in_channels=info["n_input_features"] + info["n_output_vars"],
    out_channels=info["n_output_vars"],
    sample_dimension=info["fine_shape"],
    channels=(64,128,256,512),
    down_block_types=("DownBlock2D","DownBlock2D","AttnDownBlock2D","DownBlock2D"),
    up_block_types=("UpBlock2D","AttnUpBlock2D","UpBlock2D","UpBlock2D"),
    lr=args.lr,
    )


    model = Consistency(
        config=config,
        loss_func="MSE",
    ).to(device)
    
    # Speichere normalisierungs-stats im Modell für Denormalisierung
    #
    model.y_mean = wd.y_mean
    model.y_std = wd.y_std
    
    print(f"\n✓ Normalisierungs-Stats:")
    if torch.is_tensor(wd.y_mean):
        y_mean_preview = [round(v, 4) for v in wd.y_mean[: min(4, wd.y_mean.numel())].tolist()]
        y_std_preview = [round(v, 4) for v in wd.y_std[: min(4, wd.y_std.numel())].tolist()]
        print(f"  Y mean (first channels): {y_mean_preview}")
        print(f"  Y std  (first channels): {y_std_preview}")
    else:
        print(f"  Y mean: {wd.y_mean:.4f}, std: {wd.y_std:.4f}")
    
    wandb_logger = WandbLogger(
        project="climate-project",
        entity="riyad-waez-universit-t-w-rzburg",
        config=vars(args),
    )

    plot_callback = PlotCallback(
        train_loader, val_loader, device
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(args.output_dir, "checkpoints"),
        filename="epoch{epoch:02d}-val{val_loss:.4f}",
        monitor="val_loss",
        mode="min",
        save_top_k=3,
        save_last=True,
    )

    trainer = Trainer(
        max_epochs=args.epochs,
        accelerator="gpu",
        devices=1,
        logger=wandb_logger,
        default_root_dir=args.output_dir,
        precision="bf16-mixed",
        gradient_clip_val=1.0,
        callbacks=[plot_callback, checkpoint_callback],
    )

    trainer.fit(model, train_loader, val_loader)


    
    
           

    model.eval()
    model = model.to(device)
    test_loss = 0.0
    valid_test_batches = 0

    print("\n📊 Evaluating on test set...")

    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            x, y = unpack_batch(batch, device)
            
            if batch_idx == 0:
                print(f"  Batch 0 - Y min: {y.min().item():.4f}, max: {y.max().item():.4f}")
                print(f"  X shape: {x.shape}, Y shape: {y.shape}")
            
            y_pred, _ = model.sample_conditional(
                conditioning=x,
                x_image_size=448,
                y_image_size=448,
                steps=1,
                use_ema=True,
            )

            loss = masked_mse(y_pred, y)

            if loss is None or torch.isnan(loss):
                continue

            test_loss += loss.item()
            valid_test_batches += 1

    if valid_test_batches > 0:
        avg_test = test_loss / valid_test_batches
    else:
        avg_test = float("nan")

    wandb.log({
        "epoch": args.epochs,
        "test_loss": avg_test
    })

    print(f"\n✓ Training completed!")
    print(f"  Test Loss: {avg_test:.6f}")
    print(f"  Valid test batches: {valid_test_batches}/{len(test_loader)}")

    wandb.finish()


if __name__ == "__main__":
    main()



