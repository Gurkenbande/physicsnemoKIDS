import os
import numpy as np
import matplotlib.pyplot as plt
import netCDF4 as nc

units = {
    'maximum_radar_reflectivity': 'dBZ',
    'temperature_2m':             'K',
    'eastward_wind_10m':          'm/s',
    'northward_wind_10m':         'm/s',
}
LOG_VARS = {'maximum_radar_reflectivity'}


def _compute_radial_psd(fields):
    """
    Compute radial power spectral density from 2D fields using FFT.
    Args:
        fields: Array of shape (N, H, W) containing N spatial fields
    Returns radial PSD averaged over all fields
    """
    N, H, W = fields.shape
    psds = []

    for n in range(N):
        f = fields[n]
        fft2 = np.fft.fft2(f)
        fft2_shifted = np.fft.fftshift(fft2)
        psd2d = np.abs(fft2_shifted) ** 2 / (H * W)

        cy, cx = H // 2, W // 2
        y, x = np.ogrid[:H, :W]
        r = np.sqrt((x - cx)**2 + (y - cy)**2).astype(int)

        max_r = min(cx, cy)
        radial_psd = np.array([psd2d[r == radius].mean() for radius in range(1, max_r + 1)])
        psds.append(radial_psd)

    return np.mean(psds, axis=0)

def plot_histograms(preds_denorm, targets_denorm, var_names, output_path):
    """
    Plot histograms comparing predicted vs true values for each variable.
    
    Args:
        preds_denorm: Denormalized predictions array
        targets_denorm: Denormalized target values array  
        var_names: List of variable names
        output_path: Directory to save plots
    """
    n_vars = len(var_names)
    fig, axes = plt.subplots(1, n_vars, figsize=(6 * n_vars, 5))
    axes = np.atleast_1d(axes)

    for i, name in enumerate(var_names):
        p = preds_denorm[:, i]
        t = targets_denorm[:, i]
        unit = units.get(name, '')

        if name in LOG_VARS and np.all(p > 0) and np.all(t > 0):
            p = np.log(p)
            t = np.log(t)
            xlabel = f'log({name}) [log({unit})]'
        else:
            xlabel = f'{name} [{unit}]'

        axes[i].hist(t, bins=50, alpha=0.5, label='Ground Truth', color='blue', density=True)
        axes[i].hist(p, bins=50, alpha=0.5, label='Predictions', color='red', density=True,
                     histtype='step', linewidth=1.5)
        axes[i].set_title(f'{name}: Pred vs Truth')
        axes[i].set_xlabel(xlabel)
        axes[i].set_ylabel('Probability Density')
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)
        axes[i].set_yscale('log')

    plt.tight_layout()
    plt.savefig(os.path.join(output_path, 'histograms_pred_vs_truth.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {os.path.join(output_path, 'histograms_pred_vs_truth.png')}")


def plot_power_spectra(preds_denorm, targets_denorm, var_names, output_path, fine_shape):
    """
    Plot radial power spectral density comparison between predictions and targets.
    
    Args:
        preds_denorm: Denormalized predictions
        targets_denorm: Denormalized targets
        var_names: List of variable names
        output_path: Directory to save plots
        fine_shape: (H, W) dimensions of fine resolution
    """
    H, W = fine_shape
    n_vars = len(var_names)
    N = preds_denorm.shape[0] // (H * W)

    preds_4d   = preds_denorm[:N*H*W].reshape(N, H, W, n_vars)
    targets_4d = targets_denorm[:N*H*W].reshape(N, H, W, n_vars)

    fig, axes = plt.subplots(1, n_vars, figsize=(6 * n_vars, 5))
    axes = np.atleast_1d(axes)

    for i, name in enumerate(var_names):
        pred_psd   = _compute_radial_psd(preds_4d[:, :, :, i])
        target_psd = _compute_radial_psd(targets_4d[:, :, :, i])

        freqs = np.arange(1, len(pred_psd) + 1)

        axes[i].loglog(freqs, target_psd, color='blue', label='Ground Truth', linewidth=2)
        axes[i].loglog(freqs, pred_psd,   color='red',  label='Predictions',  linewidth=2)
        axes[i].set_title(f'{name} - Power Spectral Density')
        axes[i].set_xlabel('Wavenumber')
        axes[i].set_ylabel(f'PSD [{units.get(name, "")}² / wavenumber]')
        axes[i].legend()
        axes[i].grid(True, which='both', alpha=0.3)

    plt.suptitle('Radial Power Spectral Density', fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, 'power_spectra.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {os.path.join(output_path, 'power_spectra.png')}")

def save_to_netcdf(preds, targets, inputs_lr, lon, lat, output_path, coarse_shape, var_names=None):
    """
    Save predictions, targets, and inputs to NetCDF format for further analysis.
    
    Args:
        preds: Prediction array
        targets: Target array  
        inputs_lr: Low-resolution input array
        lon, lat: Longitude and latitude coordinates
        output_path: Directory to save NetCDF file
        coarse_shape: (H, W) dimensions of coarse resolution
        var_names: List of variable names (optional)
        
    Returns:
        Path to created NetCDF file
    """
    all_var_names = ['maximum_radar_reflectivity', 'temperature_2m', 'eastward_wind_10m', 'northward_wind_10m']
    if var_names is None:
        var_names = all_var_names
        
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
    n_coarse_lat, n_coarse_lon = coarse_shape 

    if n_total_points % n_spatial != 0:
        n_samples = n_total_points // n_spatial
        remainder = n_total_points % n_spatial
        if remainder > 0:
            n_points_to_use = n_samples * n_spatial
            print(f"Truncating to {n_samples} complete samples ({n_points_to_use} points)")
            preds = preds[:n_points_to_use]
            targets = targets[:n_points_to_use]
            n_total_points = n_points_to_use
    else:
        n_samples = n_total_points // n_spatial

    n_coarse = n_coarse_lat * n_coarse_lon
    inputs_lr = inputs_lr[:n_samples * n_coarse]

    with nc.Dataset(netcdf_file, 'w', format='NETCDF4') as f:
        f.createDimension('sample', n_samples)
        f.createDimension('lat', n_lat)
        f.createDimension('lon', n_lon)
        f.createDimension('lat_lr', n_coarse_lat)
        f.createDimension('lon_lr', n_coarse_lon)

        if lon.ndim == 2:
            lon_var = f.createVariable('lon', 'f4', ('lat', 'lon'))
            lat_var = f.createVariable('lat', 'f4', ('lat', 'lon'))
        else:
            lon_var = f.createVariable('lon', 'f4', ('lon',))
            lat_var = f.createVariable('lat', 'f4', ('lat',))
        lon_var[:] = lon
        lat_var[:] = lat
        lon_var.units = 'degrees_east'
        lat_var.units = 'degrees_north'

        preds_reshaped   = preds.reshape(n_samples, n_lat, n_lon, n_vars)
        targets_reshaped = targets.reshape(n_samples, n_lat, n_lon, n_vars)
        inputs_reshaped  = inputs_lr.reshape(n_samples, n_coarse_lat, n_coarse_lon, -1)

        input_var_names = ['temperature_2m', 'eastward_wind_10m', 'northward_wind_10m']

        for var_idx in range(n_vars):
            var_name = var_names[var_idx] if var_idx < len(var_names) else f'var_{var_idx}'

            pred_var = f.createVariable(f'pred_{var_name}', 'f4', ('sample', 'lat', 'lon'))
            pred_var[:] = preds_reshaped[:, :, :, var_idx]
            pred_var.long_name = f'Predicted {var_name}'

            truth_var = f.createVariable(f'truth_{var_name}', 'f4', ('sample', 'lat', 'lon'))
            truth_var[:] = targets_reshaped[:, :, :, var_idx]
            truth_var.long_name = f'True {var_name}'

        input_var_indices = [17, 18, 19] 

        for inp_idx, inp_name in enumerate(input_var_names):
            channel_idx = input_var_indices[inp_idx]
            inp_var = f.createVariable(f'input_{inp_name}', 'f4', ('sample', 'lat_lr', 'lon_lr'))
            inp_var[:] = inputs_reshaped[:, :, :, channel_idx] 
            inp_var.long_name = f'LR Input {inp_name}'

    return netcdf_file