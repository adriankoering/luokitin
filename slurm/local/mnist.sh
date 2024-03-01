export WANDB_PROJECT=mnist

# Set these values per environment variable rather than config
# s.t. we are reminded to update them before launching a run
export WANDB_NAME=devel
export WANDB_JOB_TYPE=devel
export WANDB_JOB_NAME=${WANDB_NAME}
export WANDB_TAGS="devel,softmax,regression"
export WANDB_NOTES="Simple Softmax Regression"

export HYDRA_FULL_ERROR=1 
python train.py --multirun trainer=cpu experiment=mnist/baseline model.optimizer.lr=0.01,0.001,0.0001
# python train.py trainer=cpu experiment=mnist/baseline model.optimizer.lr=0.01 #  -c hydra