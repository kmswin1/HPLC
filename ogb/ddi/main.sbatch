#!/bin/bash
#SBATCH --job-name=ddi
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --output=logs/%j.%x.info.log
#SBATCH --error=logs/%j.%x.error.log

python ddi/main.py