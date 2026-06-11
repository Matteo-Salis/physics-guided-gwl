#!/bin/bash

#SBATCH --nodes=1                    # 1 node
#SBATCH --ntasks-per-node=1         # 32 tasks per node
#SBATCH --gpus-per-node=1
#SBATCH --mem=80000          # memory per node out of 80000MB (80GB)
#SBATCH --time=05:00:00               # time limits: 1 hour
#SBATCH --error=/leonardo_scratch/fast/IscrC_DL4STP/physics-guided-gwl/results/logs/predictions_%j.err            # standard error file
#SBATCH --output=/leonardo_scratch/fast/IscrC_DL4STP/physics-guided-gwl/results/logs/predictions_%j.out           # standard output file
#SBATCH --account=IscrC_DL4STP       # account name
#SBATCH --partition=boost_usr_prod   # partition name
#SBATCH --qos=normal             # quality of service

export PYTHONUNBUFFERED=TRUE
module purge

source /leonardo_scratch/fast/IscrC_DL4STP/.venv/bin/activate
export WANDB_MODE="offline"
cd /leonardo_scratch/fast/IscrC_DL4STP/physics-guided-gwl/src

python compute_predictions.py --config=/leonardo_scratch/fast/IscrC_DL4STP/physics-guided-gwl/config/predictions_no_film.json