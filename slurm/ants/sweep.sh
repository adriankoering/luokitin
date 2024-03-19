#!/bin/bash
#SBATCH --job-name=sgd_sweep
#SBATCH --partition=ci
#SBATCH --time=02:00:00

#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=4G

#SBATCH -o logs/slogs/%A_%a.out
#SBATCH -e logs/slogs/%A_%a.err

#SBATCH --array=1-100

source ${HOME}/.bash_profile
conda activate luo

export WANDB_PROJECT=mnist

# Set these values per environment variable rather than config
# s.t. we are reminded to update them before launching a run
export WANDB_NAME=${SLURM_JOB_NAME}
export WANDB_JOB_TYPE=sweep
export WANDB_JOB_NAME=${WANDB_NAME}
export WANDB_TAGS="softmax,regression,sgd_sweep"
export WANDB_NOTES="Simple Softmax Regression"

export HYDRA_FULL_ERROR=1
wandb agent --count 10 adriank/luokitin/ij2cvojw