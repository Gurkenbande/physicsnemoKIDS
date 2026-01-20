#!/bin/bash
#SBATCH -J lab4
#SBATCH --output=output/job-%j.txt

#SBATCH -p standard
#SBATCH --gres=gpu:1
#SBATCH --tmp=500g
#SBATCH --mem=32G  
#SBATCH --cpus-per-task=8



source /home/s448562/lab4/bin/activate  

cp -r \
/home/s448562/LAB4/data_corrdiff_3months.zarr \
/tmp/data_corrdiff_3months.zarr


srun python main.py
