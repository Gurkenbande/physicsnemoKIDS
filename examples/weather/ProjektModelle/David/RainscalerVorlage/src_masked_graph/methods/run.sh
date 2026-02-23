#!/bin/bash
#SBATCH -J rainscale
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH -p standard
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --tmp=5G
#SBATCH -e /home/s460479/ProjektRainscale/Cluster/physicsnemoKIDS/examples/weather/ProjektModelle/David/RainscalerVorlage/src_masked_graph/methods/logs/slurm-%j.out
#SBATCH -o /home/s460479/ProjektRainscale/Cluster/physicsnemoKIDS/examples/weather/ProjektModelle/David/RainscalerVorlage/src_masked_graph/methods/logs/slurm-%j.err

cd /home/s460479/ProjektRainscale/Cluster/physicsnemoKIDS/examples/weather/ProjektModelle/David/RainscalerVorlage/src_masked_graph/methods

source /home/s460479/ProjektRainscale/Cluster/venv/bin/activate

DATA_SRC=/home/s460479/ProjektRainscale/Cluster/physicsnemoKIDS/examples/weather/ProjektModelle/David/Data/cwa_dataset_3months.zarr
DATA_DST=${SLURM_TMPDIR:-/tmp}/cwa_dataset_3months.zarr

if [[ ! -d "$DATA_DST" ]]; then
  echo "Staging dataset to $DATA_DST"
  rsync -a "$DATA_SRC/" "$DATA_DST/"
fi

export CWA_DATA_PATH="$DATA_DST"
echo "CWA_DATA_PATH=$CWA_DATA_PATH"

srun python -u rainscaler.py --opt ../deep_learning/options/rainscaler_config_cwb.json
