import numpy as np
import matplotlib.pyplot as plt
import netCDF4 as nc
from pathlib import Path

class WeatherDownscalingVisualizer:
    def __init__(self, netcdf_path):
        self.nc_file = nc.Dataset(netcdf_path, 'r')
        self.var_names = self._get_variable_names()
        print(f"Found variables: {self.var_names}")
        
    def _get_variable_names(self):
        var_names = []
        for var in self.nc_file.variables.keys():
            if var.startswith('pred_'):
                var_names.append(var.replace('pred_', ''))
        return var_names
    
    def plot_single_sample(self, sample_idx=0, figsize=(20, 12), save_path=None):
        n_vars = len(self.var_names)
        fig, axes = plt.subplots(n_vars, 3, figsize=figsize)
        
        if n_vars == 1:
            axes = axes.reshape(1, -1)
        
        for i, var_name in enumerate(self.var_names):
            pred = self.nc_file.variables[f'pred_{var_name}'][sample_idx, :, :]
            truth = self.nc_file.variables[f'truth_{var_name}'][sample_idx, :, :]
            diff = pred - truth
            
            vmin = min(pred.min(), truth.min())
            vmax = max(pred.max(), truth.max())
            
            # Prediction
            im1 = axes[i, 0].imshow(pred, cmap='RdBu_r', vmin=vmin, vmax=vmax, origin='lower')
            axes[i, 0].set_title(f'{var_name.upper()} - Prediction')
            axes[i, 0].set_ylabel('Latitude')
            plt.colorbar(im1, ax=axes[i, 0], fraction=0.046)
            
            # Truth
            im2 = axes[i, 1].imshow(truth, cmap='RdBu_r', vmin=vmin, vmax=vmax, origin='lower')
            axes[i, 1].set_title(f'{var_name.upper()} - Ground Truth')
            plt.colorbar(im2, ax=axes[i, 1], fraction=0.046)
            
            # Difference
            diff_max = max(abs(diff.min()), abs(diff.max()))
            im3 = axes[i, 2].imshow(diff, cmap='seismic', vmin=-diff_max, vmax=diff_max, origin='lower')
            axes[i, 2].set_title(f'{var_name.upper()} - Difference (Pred - Truth)')
            plt.colorbar(im3, ax=axes[i, 2], fraction=0.046)
            
            mse = np.mean(diff**2)
            mae = np.mean(np.abs(diff))
            axes[i, 2].text(0.02, 0.98, f'MSE: {mse:.4f}\nMAE: {mae:.4f}', 
                          transform=axes[i, 2].transAxes, 
                          verticalalignment='top',
                          bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        for j in range(3):
            axes[-1, j].set_xlabel('Longitude')
        
        plt.suptitle(f'Weather Downscaling Results - Sample {sample_idx}', fontsize=16, y=0.995)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved to {save_path}")
        
        plt.show()
        
    def plot_variable_comparison(self, var_name, n_samples=4, figsize=(18, 12), save_path=None):
       
        fig, axes = plt.subplots(n_samples, 3, figsize=figsize)
        
        if n_samples == 1:
            axes = axes.reshape(1, -1)
        
        for i in range(n_samples):
            pred = self.nc_file.variables[f'pred_{var_name}'][i, :, :]
            truth = self.nc_file.variables[f'truth_{var_name}'][i, :, :]
            diff = pred - truth
            
            vmin = min(pred.min(), truth.min())
            vmax = max(pred.max(), truth.max())
            
            # Prediction
            im1 = axes[i, 0].imshow(pred, cmap='RdBu_r', vmin=vmin, vmax=vmax, origin='lower')
            axes[i, 0].set_ylabel(f'Sample {i}')
            if i == 0:
                axes[i, 0].set_title('Prediction')
            plt.colorbar(im1, ax=axes[i, 0], fraction=0.046)
            
            # Truth
            im2 = axes[i, 1].imshow(truth, cmap='RdBu_r', vmin=vmin, vmax=vmax, origin='lower')
            if i == 0:
                axes[i, 1].set_title('Ground Truth')
            plt.colorbar(im2, ax=axes[i, 1], fraction=0.046)
            
            # Difference
            diff_max = max(abs(diff.min()), abs(diff.max()))
            im3 = axes[i, 2].imshow(diff, cmap='seismic', vmin=-diff_max, vmax=diff_max, origin='lower')
            if i == 0:
                axes[i, 2].set_title('Difference')
            plt.colorbar(im3, ax=axes[i, 2], fraction=0.046)
        
        plt.suptitle(f'{var_name.upper()} - Multiple Samples', fontsize=16)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved to {save_path}")
        
        plt.show()
    
   

    
   
 
    
    def create_summary_report(self, output_dir, n_samples_stats=None):
      
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)
        
        print("Creating visualization report...")
        
        # 1. Single sample overview
        print("1. Plotting single sample overview...")
        self.plot_single_sample(
            sample_idx=0, 
            save_path=output_dir / "01_single_sample.png"
        )
        
        # 2. Variable comparisons
        print("2. Plotting variable comparisons...")
        for var_name in self.var_names:
            self.plot_variable_comparison(
                var_name=var_name,
                n_samples=4,
                save_path=output_dir / f"02_comparison_{var_name}.png"
            )
        
       