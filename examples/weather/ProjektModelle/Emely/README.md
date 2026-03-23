# GNN4CD — Graph Neural Network for Climate Downscaling

A GNN-based weather downscaling model that learns to map low-resolution 
atmospheric variables to high-resolution predictions using a bipartite graph structure.

## Project Structure
```
├── config.yaml          # Hydra configuration
├── gnn4cd_model.py      # Model architecture (GNN4CD)
├── graphBuilder.py      # Bipartite graph dataset construction
├── losses.py            # Custom loss functions (MSEGradientLoss, QuantizedLoss)
├── main.py              # Training and Test main file
├── main.sh              # SLURM job script
├── metrics.py           # AverageMeter 
├── plotting.py          # Histograms, power spectra, NetCDF saving
├── train_test2.py       # Trainer and Tester classes
├── vis.py               # Visualization from NetCDF results
└── visualize.ipynb      # Jupyter notebook for interactive visualization
```

## Configuration
See `config.yaml`. Key parameters:

| Parameter | Description |
|---|---|
| `dataset.type` | `mini` / `subset` / `full` |
| `model.n_output_vars` | 1 (single channel) or 4 (all variables) |
| `dataset.seq_len` | `null` (Linear encoder) or int (GRU) |

> **Note:** Set `dataset.data_path` in `config.yaml` to your data path before running:
> ```yaml
> dataset:
>   data_path: "/path/to/your/data.zarr"
> ```

> **Note:** Set the loss function in `main.py` before running:
> ```python
> # loss_fn = MSEGradientLoss(fine_shape=fine_shape, alpha=1.5)  # currently used
> # loss_fn = QuantizedLoss(n_bins=10)
> # loss_fn = nn.MSELoss()
> ```

## Single-Channel Training
Set in `main.py`:
```python
target_channels = [1]  # 0=radar, 1=t2m, 2=u10m, 3=v10m
```
And in `config.yaml`: `n_output_vars: 1`

## Usage
```bash
# Single GPU
python main.py

# Multi-GPU
torchrun --nproc_per_node=2 --nnodes=1  main.py

# SLURM
sbatch main.sh
```

## Target Variables
| Index | Variable | Unit |
|---|---|---|
| 0 | maximum_radar_reflectivity | dBZ |
| 1 | temperature_2m | K |
| 2 | eastward_wind_10m | m/s |
| 3 | northward_wind_10m | m/s |


## Output

Results are saved to `output/`:
- `checkpoint.pt` — latest checkpoint (for resuming)
- `final_model.pt` — final trained model
- `test_results.nc` — predictions and targets as NetCDF
- `histograms_pred_vs_truth.png` — value distribution comparison
- `power_spectra.png` — power spectral density



## References

Blasone et al. (2025). *Graph neural networks for hourly precipitation projections 
at the convection permitting scale with a novel hybrid imperfect framework*. 
Environmental Data Science. https://doi.org/10.1017/eds.2025.10022

