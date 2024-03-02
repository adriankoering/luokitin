export WANDB_PROJECT=mnist

# Set these values per environment variable rather than config
# s.t. we are reminded to update them before launching a run
export WANDB_NAME=devel
export WANDB_JOB_TYPE=devel
export WANDB_JOB_NAME=${WANDB_NAME}
export WANDB_TAGS="softmax,regression,devel"
export WANDB_NOTES="Simple Softmax Regression"

export HYDRA_FULL_ERROR=1 
# python train.py --multirun trainer=cpu experiment=mnist/baseline model.optimizer.lr=0.9,0.8,0.7,0.6,0.5,0.4,0.3,0.2,0.1,0.05,0.01,0.005,0.001 '+repeat_run=range(5)'
python train.py trainer=cpu experiment=mnist/baseline