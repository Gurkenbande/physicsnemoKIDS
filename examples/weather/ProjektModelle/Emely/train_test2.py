"""
Training and testing utilities 
Includes Trainer class for model training with validation and Tester class for evaluation.
"""

import os
import torch
import numpy as np
from metrics import AverageMeter
from tqdm import tqdm
from examples.weather.corrdiff.inference.plot_single_sample import (pattern_correlation)
from pathlib import Path
import wandb
from plotting import plot_histograms, plot_power_spectra,save_to_netcdf

class Trainer:
    def __init__(self, device=None, dataset=None):
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.dataset =  dataset

    def train_R_Rall(self, model, dataloader_train, dataloader_val, optimizer, loss_fn, lr_scheduler, args, target_channels=None,epoch_start=0):
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

                # Select target channels if specified
                if target_channels is not None:
                    y = y[:, target_channels]
                  
                loss = loss_fn(y_pred, y)

                loss.backward()
                optimizer.step()

                train_loss.update(loss.item(), n=y.shape[0])
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

                    if target_channels is not None:
                        y = y[:, target_channels]
                       
                    loss = loss_fn(y_pred, y)

                    val_loss.update(loss.item(),n=y.shape[0])
                    val_bar.set_postfix(val=f"{val_loss.avg:.4f}")

                    preds.append(y_pred.detach())   
                    targets.append(y.detach())

            # Compute RMSE per variable for logging
            all_preds = torch.cat(preds, dim=0)
            all_targets = torch.cat(targets, dim=0)
            rmse_per_var = torch.sqrt(((all_preds - all_targets) ** 2).mean(dim=0))
            rmse_per_var = torch.atleast_1d(rmse_per_var) 

            # Map channel indices to variable names
            all_var_names = ['maximum_radar_reflectivity', 'temperature_2m', 'eastward_wind_10m', 'northward_wind_10m']
            active_channels = target_channels if target_channels is not None else list(range(len(all_var_names)))
            channel_names = {i: all_var_names[c] for i, c in enumerate(active_channels)}

            wandb.log({
                "val_loss": val_loss.avg,
                **{f"val_rmse_{channel_names[i]}": r.item() for i, r in enumerate(rmse_per_var)}
            }, step=global_step)
            
            val_losses.append(val_loss.avg)

            tqdm.write(f"Epoch {epoch+1:03d} | Train Loss: {train_loss.avg:.4f} | Val Loss: {val_loss.avg:.4f}")

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












class Tester:
    def __init__(self, device=None, dataset=None):
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.dataset = dataset


    def test(self, model, dataloader, loss_fn, output_path="./output", lon=None, lat=None, coarse_shape=None, fine_shape=None,target_channels=None):
        model.to(self.device)
        model.eval()
        os.makedirs(output_path, exist_ok=True)

        loss_meter = AverageMeter()
        preds = []
        targets = []
        inputs= []

        with torch.no_grad():
            for graph in tqdm(dataloader, desc="Test", leave=True):
                graph = graph.to(self.device)
                y_pred = model(graph)
                y = graph['high'].y

                if target_channels is not None:
                    y = y[:, target_channels]
                    
                loss = loss_fn(y_pred, y)

                loss_meter.update(loss.item(), n=y.shape[0])

                preds.append(y_pred.cpu().numpy())
                targets.append(y.cpu().numpy())
                inputs.append(graph['low'].x.cpu().numpy())

        preds = np.concatenate(preds, axis=0)
        targets = np.concatenate(targets, axis=0)

        preds_denorm = denormalize_data(preds, self.dataset, mode='target',target_channels=target_channels)
        targets_denorm = denormalize_data(targets, self.dataset, mode='target',target_channels=target_channels)

        # for single channel training
        var_names = ['maximum_radar_reflectivity', 'temperature_2m', 'eastward_wind_10m', 'northward_wind_10m']  
        if target_channels is not None:
            var_names = [var_names[i] for i in target_channels]
        else:
            var_names = var_names 

        # evaluation metrics
        mse_per_var = np.atleast_1d(((preds_denorm - targets_denorm) ** 2).mean(axis=0))
        rmse_per_var = np.atleast_1d(np.sqrt(mse_per_var))
        mae_per_var = np.atleast_1d(np.abs(preds_denorm - targets_denorm).mean(axis=0))
       
        for i, (name, mse) in enumerate(zip(var_names, mse_per_var)):
            print(f"MSE {name}: {mse:.4f}")

        for i, (name, rmse) in enumerate(zip(var_names, rmse_per_var)):
            print(f"RMSE {name}: {rmse:.4f}")

        for i, (name, mae) in enumerate(zip(var_names, mae_per_var)):
            print(f"MAE {name}: {mae:.4f}")

        for var_idx in range(preds_denorm.shape[1]):
            corr = pattern_correlation(preds_denorm[:, var_idx], targets_denorm[:, var_idx])
            print(f"Pattern Correlation (var {var_idx}): {corr:.4f}")

        inputs = np.concatenate(inputs, axis=0)          
        inputs_denorm = denormalize_data(inputs, self.dataset, mode='input')  
        
        plot_histograms(preds_denorm, targets_denorm, var_names, output_path)
        plot_power_spectra(preds_denorm, targets_denorm, var_names, output_path, fine_shape=fine_shape)
        save_to_netcdf(preds_denorm, targets_denorm, inputs_denorm, lon, lat, output_path, coarse_shape, var_names=var_names)
      
        print(f"Final Test Loss: {loss_meter.avg:.6f}")
        return preds_denorm, targets_denorm, loss_meter.avg


def denormalize_data(data, dataset, mode='target', target_channels=None):
    """
        data: Normalized data array
        mode: 'target' or 'input' normalization
        target_channels: Specific channels to denormalize (for target mode)

    """
    info = dataset.info()
    key = 'target_normalization' if mode == 'target' else 'input_normalization'
    centers, scales = info[key]

    if target_channels is not None:
        centers = np.array(centers)[target_channels]
        scales = np.array(scales)[target_channels]

    data_np = np.array(data)
    return data_np * scales[np.newaxis, :] + centers[np.newaxis, :]
