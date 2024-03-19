export WANDB_PROJECT=imagenet

export WANDB_NAME=frozen_resnet18
export WANDB_JOB_TYPE=initial
export WANDB_JOB_NAME=${WANDB_NAME}
export WANDB_TAGS="initial,imagenet,resnet18,frozen"
export WANDB_NOTES="Early Devel Run"

export HYDRA_FULL_ERROR=1 
python train.py trainer=explore experiment=imagenet/baseline