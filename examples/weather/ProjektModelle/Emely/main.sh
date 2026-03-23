#!/bin/bash
#SBATCH -J lab4
#SBATCH --output=output/job-%j.txt
#SBATCH -p standard
#SBATCH --gres=gpu:2
#SBATCH --ntasks=1             
#SBATCH --cpus-per-task=32
#SBATCH --tmp=500g
#SBATCH --mem=128G


source /home/s448562/lab4/bin/activate  

cp -r \
/home/s448562/LAB4/data_corrdiff_3months.zarr \
/tmp/data_corrdiff_3months.zarr

torchrun \
--nproc_per_node=2 \
--nnodes=1 \
--master_port=$((29500 + SLURM_JOB_ID % 1000)) \
main.py