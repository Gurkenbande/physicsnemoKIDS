"""
Visualization utilities
Creates plots comparing predictions, ground truth, and low-resolution inputs.
"""

import numpy as np
import matplotlib.pyplot as plt
import netCDF4 as nc
from pathlib import Path

class WeatherDownscalingVisualizer:
    def __init__(self, netcdf_path):
        self.nc_file = nc.Dataset(netcdf_path, 'r')
        self.var_names = self._get_variable_names()
        
    def _get_variable_names(self):
        """Extract variable names from NetCDF file by finding prediction variables."""
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
        """
        Plot a single sample showing LR input, prediction, and ground truth
        
        Args:
            sample_idx: Index of the sample to plot
            save_path: Path to save the plot 
        """
        n_vars = len(self.var_names)
        fig, axes = plt.subplots(n_vars, 3, figsize=figsize)
        if n_vars == 1:
            axes = axes.reshape(1, -1)

        input_var_names = ['temperature_2m', 'eastward_wind_10m', 'northward_wind_10m']
        vars_with_input = set(input_var_names)

        for i, var_name in enumerate(self.var_names):
            pred  = self.nc_file.variables[f'pred_{var_name}'][sample_idx, :, :]
            truth = self.nc_file.variables[f'truth_{var_name}'][sample_idx, :, :]

            has_input = var_name in vars_with_input
            if has_input:
                lr_input = self.nc_file.variables[f'input_{var_name}'][sample_idx, :, :]
                vmin = min(pred.min(), truth.min(), lr_input.min())
                vmax = max(pred.max(), truth.max(), lr_input.max())
            else:
                vmin = min(pred.min(), truth.min())
                vmax = max(pred.max(), truth.max())

            cmap = 'RdBu_r'
            
            # Column 0: LR Input (if available)
            if has_input:
                im0 = axes[i, 0].imshow(lr_input, cmap=cmap, vmin=vmin, vmax=vmax, origin='lower')
                axes[i, 0].set_title(f'{var_name.upper()} - LR Input')
                plt.colorbar(im0, ax=axes[i, 0], fraction=0.046)
            else:
                axes[i, 0].axis('off')
                axes[i, 0].set_title(f'{var_name.upper()} - No LR Input')

            # Column 1: Prediction
            im1 = axes[i, 1].imshow(pred, cmap=cmap, vmin=vmin, vmax=vmax, origin='lower')
            axes[i, 1].set_title(f'{var_name.upper()} - Prediction')
            plt.colorbar(im1, ax=axes[i, 1], fraction=0.046)

            # Column 2: Ground Truth
            im2 = axes[i, 2].imshow(truth, cmap=cmap, vmin=vmin, vmax=vmax, origin='lower')
            axes[i, 2].set_title(f'{var_name.upper()} - Ground Truth')
            plt.colorbar(im2, ax=axes[i, 2], fraction=0.046)

            axes[i, 0].set_ylabel(var_name)

        for j in range(3):
            axes[-1, j].set_xlabel('Longitude')

        plt.suptitle(f'Sample {sample_idx}', fontsize=16, y=0.995)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()
    
    def plot_variable_comparison(self, var_name, n_samples=4, figsize=(18, 12), save_path=None):
        """
        Plot comparison for a specific variable across multiple samples.
        
        Args:
            var_name: Name of the variable to plot
            n_samples: Number of samples to show
            figsize: Figure size (width, height)
            save_path: Path to save the plot (optional)
        """
        sample_indices = self._get_sample_indices(n_samples)
        
        input_var_names = {'temperature_2m', 'eastward_wind_10m', 'northward_wind_10m'}
        has_input = var_name in input_var_names
        n_cols = 3 if has_input else 2

        fig, axes = plt.subplots(n_samples, n_cols, figsize=figsize)
        if n_samples == 1:
            axes = axes.reshape(1, -1)

        for i, sample_idx in enumerate(sample_indices):
            pred  = self.nc_file.variables[f'pred_{var_name}'][sample_idx, :, :]
            truth = self.nc_file.variables[f'truth_{var_name}'][sample_idx, :, :]

            if has_input:
                lr_input = self.nc_file.variables[f'input_{var_name}'][sample_idx, :, :]
                vmin = min(pred.min(), truth.min(), lr_input.min())
                vmax = max(pred.max(), truth.max(), lr_input.max())

                # LR Input column
                im0 = axes[i, 0].imshow(lr_input, cmap='RdBu_r', vmin=vmin, vmax=vmax, origin='lower')
                axes[i, 0].set_ylabel(f'Sample {sample_idx}')
                if i == 0:
                    axes[i, 0].set_title('LR Input')
                plt.colorbar(im0, ax=axes[i, 0], fraction=0.046)

                # Prediction column
                im1 = axes[i, 1].imshow(pred, cmap='RdBu_r', vmin=vmin, vmax=vmax, origin='lower')
                if i == 0:
                    axes[i, 1].set_title('Prediction')
                plt.colorbar(im1, ax=axes[i, 1], fraction=0.046)

                # Ground truth column
                im2 = axes[i, 2].imshow(truth, cmap='RdBu_r', vmin=vmin, vmax=vmax, origin='lower')
                if i == 0:
                    axes[i, 2].set_title('Ground Truth')
                plt.colorbar(im2, ax=axes[i, 2], fraction=0.046)

            else:  # radar — kein LR Input
                vmin = min(pred.min(), truth.min())
                vmax = max(pred.max(), truth.max())

                # Prediction column
                im1 = axes[i, 0].imshow(pred, cmap='RdBu_r', vmin=vmin, vmax=vmax, origin='lower')
                axes[i, 0].set_ylabel(f'Sample {sample_idx}')
                if i == 0:
                    axes[i, 0].set_title('Prediction')
                plt.colorbar(im1, ax=axes[i, 0], fraction=0.046)

                # Ground truth column
                im2 = axes[i, 1].imshow(truth, cmap='RdBu_r', vmin=vmin, vmax=vmax, origin='lower')
                if i == 0:
                    axes[i, 1].set_title('Ground Truth')
                plt.colorbar(im2, ax=axes[i, 1], fraction=0.046)

        plt.suptitle(f'{var_name.upper()} - Multiple Samples', fontsize=16)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()
    
    def combined_plot(self, output_dir):
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)
        
        # Create single sample plot
        self.plot_single_sample(
            sample_idx=0, 
            save_path=output_dir / "01_single_sample.png"
        )
        
        # Create comparison plots for each variable
        for var_name in self.var_names:
            self.plot_variable_comparison(
                var_name=var_name,
                n_samples=4,
                save_path=output_dir / f"02_comparison_{var_name}.png"
            )