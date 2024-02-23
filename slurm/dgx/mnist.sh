#!/bin/bash
#SBATCH --job-name=mnist
#SBATCH --partition=p2
#SBATCH --time=1:00:00


# DGX features 10 threads and 62 GB memory per GPU (6.25 GB per CPU)
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --mem-per-cpu=6G

#SBATCH -o logs/slogs/%A_%a.out
#SBATCH -e logs/slogs/%A_%a.err

##SBATCH --array=1-42

export WANDB_NOTES="Softmax Regression Type Model"
export WANDB_PROJECT=mnist

export HYDRA_FULL_ERROR=1 
srun python train.py trainer=cpu
# +experiment=reference
#  +experiment=loss/cross_entropy
# srun python train.py --config-path config/city +experiment=loss/inverse_iou