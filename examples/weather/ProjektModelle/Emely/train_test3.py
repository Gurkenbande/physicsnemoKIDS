import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import netCDF4 as nc
from metrics import AverageMeter
from tqdm import tqdm
from examples.weather.corrdiff.inference.plot_single_sample import (
    main as plot_main,
    pattern_correlation
)

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
        for epoch in range(epoch_start, epoch_start + args.epochs):

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
                            "train_loss":train_loss.avg,
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

                    if epoch % log_freq == 0:
                        preds.append(y_pred[mask].detach())
                        targets.append(y[mask].detach())
            wandb.log(
                        {
                            "val_loss":val_loss.avg,
                        }, 
                        step=global_step)
                    
            val_losses.append(val_loss.avg)
            tqdm.write(f"Epoch {epoch+1:03d} | Train: {train_loss.avg:.4f} | Val: {val_loss.avg:.4f}")
            

            if epoch % log_freq == 0 and preds:
                self._create_plots_R_Rall(torch.cat(preds, dim=0), torch.cat(targets, dim=0), epoch, args)

            if lr_scheduler is not None:
                if isinstance(lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    lr_scheduler.step(val_loss.avg)
                else:
                    lr_scheduler.step()

        self._plot_loss_curves(train_losses, val_losses, args)




class Tester:
    def __init__(self, device=None):
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def save_to_netcdf(self, preds, targets, lon, lat, output_path, times=None):
        netcdf_file = os.path.join(output_path, "test_results.nc")
        
        with nc.Dataset(netcdf_file, 'w', format='NETCDF4') as f:
            n_samples, n_times = preds.shape[0], preds.shape[1] if preds.ndim > 2 else 1
            n_lat, n_lon = lon.shape[0], lon.shape[1] if lon.ndim > 1 else lon.shape[0]
            
            f.createDimension('sample', n_samples)
            f.createDimension('time', n_times)
            f.createDimension('lat', n_lat)
            f.createDimension('lon', n_lon)
            
            lon_var = f.createVariable('lon', 'f4', ('lon',))
            lat_var = f.createVariable('lat', 'f4', ('lat',))
            lon_var[:] = lon.flatten() if lon.ndim > 1 else lon
            lat_var[:] = lat.flatten() if lat.ndim > 1 else lat
           
            time_var = f.createVariable('time', 'f8', ('time',))
            if times is not None:
                time_var[:] = times
            else:
                time_var[:] = np.arange(n_times)
            time_var.units = 'hours since 2000-01-01 00:00:00'
            time_var.calendar = 'standard'
            pred_group = f.createGroup('prediction')
            truth_group = f.createGroup('truth')
            
            pred_var = pred_group.createVariable('output', 'f4', ('sample', 'time', 'lat', 'lon'))
            truth_var = truth_group.createVariable('output', 'f4', ('time', 'lat', 'lon'))
            
            if preds.ndim == 2:  
                preds_reshaped = preds.reshape(n_samples, n_times, n_lat, n_lon)
                targets_reshaped = targets.reshape(n_times, n_lat, n_lon)
            else:
                preds_reshaped = preds
                targets_reshaped = targets
            
            pred_var[:] = preds_reshaped
            truth_var[:] = targets_reshaped
        
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
        
        corr = pattern_correlation(preds.flatten(), targets.flatten())
        print(f"Pattern Correlation: {corr:.4f}")

        np.savez(os.path.join(output_path, "test_outputs.npz"),prediction=preds,truth=targets)

        netcdf_file = self.save_to_netcdf(preds, targets, lon, lat, output_path)
       
        print(f"Final Test Loss: {loss_meter.avg:.6f}")
        return preds, targets, loss_meter.avg

