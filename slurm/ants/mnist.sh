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

# source ${HOME}/.bashrc

# conda activate nighly
# cd ${HOME}/certaimage

export WANDB_PROJECT=mnist

srun python train.py 
# +experiment=loss/cross_entropy