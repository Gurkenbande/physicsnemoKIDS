import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import netCDF4 as nc
from metrics import AverageMeter
from tqdm import tqdm
from examples.weather.corrdiff.inference.plot_single_sample import (
    pattern_correlation
)
from pathlib import Path
import wandb

class Trainer:
    def __init__(self, device=None):
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def _create_plots_R_Rall(self, y_pred, y, epoch, args):
        os.makedirs(args.output_path, exist_ok=True)
        y_pred_mean = y_pred.mean(dim=0).cpu().numpy()
        y_mean = y.mean(dim=0).cpu().numpy()
        x = np.arange(y_mean.shape[0])

        plt.figure(figsize=(8, 4))
        plt.bar(x - 0.2, y_mean, width=0.4, label="Ground Truth")
        plt.bar(x + 0.2, y_pred_mean, width=0.4, label="Prediction")
        plt.xlabel("Output variable index")
        plt.ylabel("Mean value")
        plt.title(f"Validation mean prediction (epoch {epoch})")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(args.output_path, f"val_mean_epoch_{epoch}.png"))
        plt.close()

    def _plot_loss_curves(self, train_losses, val_losses, args):
        os.makedirs(args.output_path, exist_ok=True)
        epochs = np.arange(1, len(train_losses) + 1)

        plt.figure(figsize=(8, 5))
        plt.plot(epochs, train_losses, label="Train Loss", marker='o')
        plt.plot(epochs, val_losses, label="Val Loss", marker='o')
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Train vs Validation Loss")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(args.output_path, "loss_curve.png"))
        plt.close()

    def train_R_Rall(self, model, dataloader_train, dataloader_val, optimizer, loss_fn, lr_scheduler, args, epoch_start=0, log_freq=5):
        wandb.init(project="downscaling")
        model.to(self.device)
        train_losses, val_losses = [], []
        global_step = 0

        for epoch in range(epoch_start, args.epochs):
            model.train()
            train_loss = AverageMeter()
            train_bar = tqdm(dataloader_train, desc=f"Epoch {epoch+1} | Train", leave=False)

            for graph in train_bar:
                graph = graph.to(self.device)
                optimizer.zero_grad(set_to_none=True)

                y_pred = model(graph)
                y = graph['high'].y
                mask = graph['high'].train_mask

                loss = loss_fn(y_pred[mask], y[mask])
                loss.backward()
                optimizer.step()

                train_loss.update(loss.item(), n=mask.sum().item())
                wandb.log(
                    {
                        "train_loss": loss.item(),
                    },
                    step=global_step)

                train_bar.set_postfix(train=f"{train_loss.avg:.4f}")
                global_step += 1

            train_losses.append(train_loss.avg)

            # Validation
            model.eval()
            val_loss = AverageMeter()
            preds, targets = [], []
            val_bar = tqdm(dataloader_val, desc=f"Epoch {epoch+1} | Val", leave=False)

            with torch.no_grad():
                for graph in val_bar:
                    graph = graph.to(self.device)
                    y_pred = model(graph)
                    y = graph['high'].y
                    mask = graph['high'].train_mask

                    loss = loss_fn(y_pred[mask], y[mask])
                    val_loss.update(loss.item(), n=mask.sum().item())
                    val_bar.set_postfix(val=f"{val_loss.avg:.4f}")

                    preds.append(y_pred[mask].detach())   
                    targets.append(y[mask].detach())

            all_preds = torch.cat(preds, dim=0)
            all_targets = torch.cat(targets, dim=0)
            rmse_per_var = torch.sqrt(((all_preds - all_targets) ** 2).mean(dim=0))

            wandb.log({"val_loss": val_loss.avg, **{f"val_rmse_var{i}": r.item() 
                    for i, r in enumerate(rmse_per_var)}}, step=global_step)

            val_losses.append(val_loss.avg)

            tqdm.write(f"Epoch {epoch+1:03d} | Train: {train_loss.avg:.4f} | Val: {val_loss.avg:.4f}")

            if epoch % log_freq == 0:
                self._create_plots_R_Rall(all_preds, all_targets, epoch, args)
                
            if lr_scheduler is not None:
                if isinstance(lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    lr_scheduler.step(val_loss.avg)
                else:
                    lr_scheduler.step()

            checkpoint_path = Path(args.output_path) / "checkpoint.pt"
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": (
                    lr_scheduler.state_dict() if lr_scheduler is not None else None
                ),
                "train_losses": train_losses,
                "val_losses": val_losses,
            }, checkpoint_path)

        self._plot_loss_curves(train_losses, val_losses, args)


class Tester:
    def __init__(self, device=None, dataset=None):
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.dataset = dataset

    def save_to_netcdf(self, preds, targets, lon, lat, output_path, times=None):
        netcdf_file = os.path.join(output_path, "test_results.nc")
        if os.path.exists(netcdf_file):
            os.remove(netcdf_file)

        n_total_points = preds.shape[0]
        n_vars = preds.shape[1]

        if lon.ndim == 2:
            n_lat, n_lon = lon.shape
        elif lon.ndim == 1:
            n_lon = len(lon)
            n_lat = len(lat)
        else:
            raise ValueError(f"Unexpected lon dimensions: {lon.shape}")

        n_spatial = n_lat * n_lon

        if n_total_points % n_spatial != 0:
            n_samples = n_total_points // n_spatial
            remainder = n_total_points % n_spatial
            if remainder > 0:
                n_complete_samples = n_samples
                n_points_to_use = n_complete_samples * n_spatial
                print(f"Truncating to {n_complete_samples} complete samples ({n_points_to_use} points)")
                preds = preds[:n_points_to_use]
                targets = targets[:n_points_to_use]
                n_total_points = n_points_to_use
                n_samples = n_complete_samples
        else:
            n_samples = n_total_points // n_spatial

        with nc.Dataset(netcdf_file, 'w', format='NETCDF4') as f:
            f.createDimension('sample', n_samples)
            f.createDimension('lat', n_lat)
            f.createDimension('lon', n_lon)
            f.createDimension('variable', n_vars)

            if lon.ndim == 2:
                lon_var = f.createVariable('lon', 'f4', ('lat', 'lon'))
                lat_var = f.createVariable('lat', 'f4', ('lat', 'lon'))
                lon_var[:] = lon
                lat_var[:] = lat
            else:
                lon_var = f.createVariable('lon', 'f4', ('lon',))
                lat_var = f.createVariable('lat', 'f4', ('lat',))
                lon_var[:] = lon
                lat_var[:] = lat

            lon_var.units = 'degrees_east'
            lat_var.units = 'degrees_north'

            preds_reshaped = preds.reshape(n_samples, n_lat, n_lon, n_vars)
            targets_reshaped = targets.reshape(n_samples, n_lat, n_lon, n_vars)

            var_names = ['maximum_radar_reflectivity', 'temperature_2m', 'eastward_wind_10m', 'northward_wind_10m']

            for var_idx in range(n_vars):
                var_name = var_names[var_idx] if var_idx < len(var_names) else f'var_{var_idx}'

                pred_var = f.createVariable(f'pred_{var_name}', 'f4', ('sample', 'lat', 'lon'))
                pred_var[:] = preds_reshaped[:, :, :, var_idx]
                pred_var.long_name = f'Predicted {var_name}'

                truth_var = f.createVariable(f'truth_{var_name}', 'f4', ('sample', 'lat', 'lon'))
                truth_var[:] = targets_reshaped[:, :, :, var_idx]
                truth_var.long_name = f'True {var_name}'

            f.description = 'Weather downscaling model predictions'
            f.source = 'GNN4CD downscaling model'

        return netcdf_file

    def test(self, model, dataloader, loss_fn, output_path="./output", lon=None, lat=None):
        model.to(self.device)
        model.eval()
        os.makedirs(output_path, exist_ok=True)

        loss_meter = AverageMeter()
        preds = []
        targets = []

        with torch.no_grad():
            for graph in tqdm(dataloader, desc="Test", leave=True):
                graph = graph.to(self.device)
                y_pred = model(graph)
                y = graph['high'].y
                mask = graph['high'].train_mask

                loss = loss_fn(y_pred[mask], y[mask])
                loss_meter.update(loss.item(), n=mask.sum().item())

                preds.append(y_pred[mask].cpu().numpy())
                targets.append(y[mask].cpu().numpy())

        preds = np.concatenate(preds, axis=0)
        targets = np.concatenate(targets, axis=0)

        # denormalizing
        info = self.dataset.info()
        centers, scales = info['target_normalization']
        preds_denorm = preds * scales[np.newaxis, :] + centers[np.newaxis, :]
        targets_denorm = targets * scales[np.newaxis, :] + centers[np.newaxis, :]

        rmse_per_var = np.sqrt(((preds_denorm - targets_denorm) ** 2).mean(axis=0))
        var_names = ['reflectivity', 'temperature_2m', 'wind_u', 'wind_v']
        for i, (name, rmse) in enumerate(zip(var_names, rmse_per_var)):
            print(f"RMSE {name}: {rmse:.4f}")

        for var_idx in range(preds_denorm.shape[1]):
            corr = pattern_correlation(preds_denorm[:, var_idx], targets_denorm[:, var_idx])
            print(f"Pattern Correlation (var {var_idx}): {corr:.4f}")

        self.save_to_netcdf(preds_denorm, targets_denorm, lon, lat, output_path)

        print(f"Final Test Loss: {loss_meter.avg:.6f}")
        return preds_denorm, targets_denorm, loss_meter.avg