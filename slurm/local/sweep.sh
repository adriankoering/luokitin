export WANDB_PROJECT=mnist

# Set these values per environment variable rather than config
# s.t. we are reminded to update them before launching a run
export WANDB_NAME=sgd_sweep
export WANDB_JOB_TYPE=sweep
export WANDB_JOB_NAME=${WANDB_NAME}
export WANDB_TAGS="softmax,regression,sgd_sweep"
export WANDB_NOTES="Simple Softmax Regression"

export HYDRA_FULL_ERROR=1
wandb agent --count 10 adriank/luokitin/ij2cvojw