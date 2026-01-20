#!/bin/bash
#SBATCH -J copy_cwa_dataset
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH -p standard
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --tmp=5G
#SBATCH -e /home/s460479/ProjektRainscale/Cluster/physicsnemoKIDS/examples/weather/ProjektModelle/David/RainscalerVorlage/src_masked_graph/methods/logs/slurm-%j.out

source /home/s460479/ProjektRainscale/Cluster/venv/bin/activate

cp -r \
  /data/42-julia-hpc-rz-lsx/sih25nq/downscaling/CorrDiff/cwa_dataset/cwa_dataset_3months.zarr \
  /home/s460479/ProjektRainscale/Cluster/physicsnemoKIDS/examples/weather/ProjektModelle/David/Data/cwa_dataset_3months.zarr
