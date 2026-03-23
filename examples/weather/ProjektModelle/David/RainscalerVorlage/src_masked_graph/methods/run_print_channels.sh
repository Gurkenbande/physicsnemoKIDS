#!/bin/bash
#SBATCH -J print_cwa_channels
#SBATCH -c 32
#SBATCH --mem=128G
#SBATCH -p standard
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --tmp=500G
#SBATCH -e /home/s460479/ProjektRainscale/Cluster/physicsnemoKIDS/examples/weather/ProjektModelle/David/RainscalerVorlage/src_masked_graph/methods/logs/slurm-%j.out
#SBATCH -o /home/s460479/ProjektRainscale/Cluster/physicsnemoKIDS/examples/weather/ProjektModelle/David/RainscalerVorlage/src_masked_graph/methods/logs/slurm-%j.err

cd /home/s460479/ProjektRainscale/Cluster/physicsnemoKIDS/examples/weather/ProjektModelle/David/RainscalerVorlage/src_masked_graph/methods

source /home/s460479/ProjektRainscale/Cluster/venv/bin/activate

cp -r \
  /home/s460479/ProjektRainscale/Cluster/physicsnemoKIDS/examples/weather/ProjektModelle/David/Data/cwa_dataset_3months.zarr \
  /tmp/cwa_dataset_3months.zarr

srun python -u print_dataset_channels.py --data_path /tmp/cwa_dataset_3months.zarr
