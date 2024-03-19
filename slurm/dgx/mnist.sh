#!/bin/bash
#SBATCH --job-name=baseline
#SBATCH --partition=p2
#SBATCH --time=1:00:00


# DGX features 10 threads and 62 GB memory per GPU (6.25 GB per CPU)
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --mem-per-cpu=6G

#SBATCH -o logs/slogs/%A_%a.out
#SBATCH -e logs/slogs/%A_%a.err

##SBATCH --array=1-42

export WANDB_PROJECT=mnist

# Set these values per environment variable rather than config
# s.t. we are reminded to update them before launching a run
export WANDB_NAME=${SLURM_JOB_NAME}
export WANDB_JOB_TYPE=devel
export WANDB_JOB_NAME=${WANDB_NAME}
export WANDB_TAGS="devel,softmax,regression"
export WANDB_NOTES="Simple Softmax Regression"

export HYDRA_FULL_ERROR=1 
srun python train.py trainer=cpu