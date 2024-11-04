#!/bin/bash
#SBATCH --job-name=day_rgbd_rn34
#SBATCH --partition=gpu
#SBATCH --time=3:00:00

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem-per-cpu=2G

# mkdir -p logs/slogs/${SLURM_SLURM_ARRAY_JOB_ID}
#SBATCH -o logs/slogs/%A_%a.out
#SBATCH -e logs/slogs/%A_%a.err

##SBATCH --export ALL # exports env-varialbes from current shell to job?

#SBATCH --nice=1000
#SBATCH --array=1-8%4

export WANDB_PROJECT=sofja

# Set these values per environment variable rather than config
# s.t. we are reminded to update them before launching a run
export WANDB_NAME=${SLURM_JOB_NAME}
export WANDB_JOB_TYPE=devel
export WANDB_JOB_NAME=${WANDB_NAME}
export WANDB_TAGS="devel,rgbd,day,resnet34"
export WANDB_MODE=offline

export HYDRA_FULL_ERROR=1 
srun python train.py experiment=invvis/rgbd dataset.data_dir=/home/koering/data/invvis/webds/daysplit model/encoder=resnet34 model.encoder.pretrained=False