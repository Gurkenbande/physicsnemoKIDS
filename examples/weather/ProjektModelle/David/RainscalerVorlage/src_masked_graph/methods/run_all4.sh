#!/bin/bash
#SBATCH -J rainscale_all4_ch0
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

srun python -u rainscaler_rain_all4.py --opt ../deep_learning/options/rainscaler_config_cwb_all4.json
