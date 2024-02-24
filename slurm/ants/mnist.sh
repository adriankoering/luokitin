#!/bin/bash
#SBATCH --job-name=mnist
#SBATCH --partition=gpu
#SBATCH --time=2-00:00:00

#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-cpu=4G

#SBATCH -o logs/slogs/%A_%a.out
#SBATCH -e logs/slogs/%A_%a.err

##SBATCH --array=1-3

export WANDB_PROJECT=mnist

# Set these values per environment variable rather than config
# s.t. we are reminded to update them before launching a run
export WANDB_NAME=devel
export WANDB_JOB_TYPE=devel
export WANDB_JOB_NAME=devel
export WANDB_TAGS="devel,softmax,regression"
export WANDB_NOTES="Simple Softmax Regression"

srun python train.py