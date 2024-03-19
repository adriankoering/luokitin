#!/bin/bash
#SBATCH --job-name=mnist
#SBATCH --partition=ci
#SBATCH --time=2-00:00:00

#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=4G

#SBATCH -o logs/slogs/%A_%a.out
#SBATCH -e logs/slogs/%A_%a.err

##SBATCH --array=1-3

source ${HOME}/.bash_profile
conda activate luo

export WANDB_PROJECT=mnist

# Set these values per environment variable rather than config
# s.t. we are reminded to update them before launching a run
export WANDB_NAME=${SLURM_JOB_NAME}
export WANDB_JOB_TYPE=baseline
export WANDB_JOB_NAME=${WANDB_NAME}
export WANDB_TAGS="baseline,softmax,regression"
export WANDB_NOTES="Simple Softmax Regression"

srun python train.py trainer=cpu experiment=mnist/baseline