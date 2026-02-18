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

srun python -u rainscaler.py --opt ../deep_learning/options/rainscaler_config_cwb.json