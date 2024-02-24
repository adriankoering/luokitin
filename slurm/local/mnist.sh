export WANDB_PROJECT=mnist

# Set these values per environment variable rather than config
# s.t. we are reminded to update them before launching a run
export WANDB_NAME=devel
export WANDB_JOB_TYPE=devel
export WANDB_JOB_NAME=devel
export WANDB_TAGS="devel,softmax,regression"
export WANDB_NOTES="Simple Softmax Regression"

export HYDRA_FULL_ERROR=1 
python train.py trainer=cpu experiment=mnist/baseline