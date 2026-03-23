# Climate Downscaling with Consistency Models

This project implements a **consistency model for climate data downscaling** using PyTorch Lightning.  
The model learns to map low-resolution weather data to high-resolution outputs.

## Overview

- Model: Consistency Model 
- Architecture: UNet (from Hugging Face diffusers)
- Framework: PyTorch Lightning
- Logging: Weights & Biases (wandb)

---

## Project Structure
├── modelupdate.py 
├── train_corrdiff_update.py 
├── dataset.py 
├── loss.py 
├── configurationupdate.py 
├── requirements.txt
└── README.md



## how to start:
1) clone the project
2) connect to the cluster
   ssh s458614@julia2.hpc.uni-wuerzburg.de
   
3) find the path
   cd ~/climate_project/physicsnemoKIDS
4) GPU
   srun -J rainscale      -p standard      --gres=gpu:1      -c 16      --mem=64G      --time=24:00:00      --pty bash

5) Environment 
6) cd ~/climate_project/physicsnemoKIDS/examples/weather/ProjektModelle/Ryad/MaxConsistency
7) copy to tmp:
   cp -r /data/42-julia-hpc-rz-lsx/sih25nq/downscaling/CorrDiff/cwa_dataset/cwa_dataset_3months.zarr /tmp/
8) start the training:
   python train_corrdiff_update.py --mode sub --batch_size 8 --epochs x --num_workers 16
   
 
