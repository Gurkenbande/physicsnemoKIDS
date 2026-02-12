import numpy as np
import matplotlib.pyplot as plt
import netCDF4 as nc
from pathlib import Path

class WeatherDownscalingVisualizer:
    def __init__(self, netcdf_path):
        self.nc_file = nc.Dataset(netcdf_path, 'r')
        self.var_names = self._get_variable_names()
        
    def _get_variable_names(self):
        var_names = []
        for var in self.nc_file.variables.keys():
            if var.startswith('pred_'):
                var_names.append(var.replace('pred_', ''))
        return var_names
    
    def _get_sample_indices(self, n_samples):
        total_samples = self.nc_file.variables[f'pred_{self.var_names[0]}'].shape[0]
        if n_samples >= total_samples:
            return list(range(total_samples))
        
        indices = np.linspace(0, total_samples - 1, n_samples, dtype=int)
        return indices
    
    def plot_single_sample(self, sample_idx=0, figsize=(20, 12), save_path=None):
        n_vars = len(self.var_names)
        fig, axes = plt.subplots(n_vars, 2, figsize=figsize)
        
        if n_vars == 1:
            axes = axes.reshape(1, -1)
        
        for i, var_name in enumerate(self.var_names):
            pred = self.nc_file.variables[f'pred_{var_name}'][sample_idx, :, :]
            truth = self.nc_file.variables[f'truth_{var_name}'][sample_idx, :, :]
            
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
        
        for j in range(2):
            axes[-1, j].set_xlabel('Longitude')
        
        plt.suptitle(f'Sample {sample_idx}', fontsize=16, y=0.995)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()
    
    def plot_variable_comparison(self, var_name, n_samples=4, figsize=(18, 12), save_path=None):
        sample_indices = self._get_sample_indices(n_samples)
        
        fig, axes = plt.subplots(n_samples, 2, figsize=figsize)
        
        if n_samples == 1:
            axes = axes.reshape(1, -1)
        
        for i, sample_idx in enumerate(sample_indices):
            pred = self.nc_file.variables[f'pred_{var_name}'][sample_idx, :, :]
            truth = self.nc_file.variables[f'truth_{var_name}'][sample_idx, :, :]
            
            vmin = min(pred.min(), truth.min())
            vmax = max(pred.max(), truth.max())
            
            # Prediction
            im1 = axes[i, 0].imshow(pred, cmap='RdBu_r', vmin=vmin, vmax=vmax, origin='lower')
            axes[i, 0].set_ylabel(f'Sample {sample_idx}')
            if i == 0:
                axes[i, 0].set_title('Prediction')
            plt.colorbar(im1, ax=axes[i, 0], fraction=0.046)
            
            # Truth
            im2 = axes[i, 1].imshow(truth, cmap='RdBu_r', vmin=vmin, vmax=vmax, origin='lower')
            if i == 0:
                axes[i, 1].set_title('Ground Truth')
            plt.colorbar(im2, ax=axes[i, 1], fraction=0.046)
        
        plt.suptitle(f'{var_name.upper()} - Multiple Samples', fontsize=16)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()
    
    def create_summary_report(self, output_dir, n_samples_stats=None):
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)
        
        self.plot_single_sample(
            sample_idx=0, 
            save_path=output_dir / "01_single_sample.png"
        )
        
        for var_name in self.var_names:
            self.plot_variable_comparison(
                var_name=var_name,
                n_samples=4,
                save_path=output_dir / f"02_comparison_{var_name}.png"
            )