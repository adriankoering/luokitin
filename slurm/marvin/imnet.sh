export WANDB_PROJECT=imagenet

export WANDB_NAME=short_fresh_random_crop_conv_atto
export WANDB_JOB_TYPE=initial
export WANDB_JOB_NAME=${WANDB_NAME}
export WANDB_TAGS="initial,imagenet,conv_atto,random_crop,nopretraining"
export WANDB_NOTES="Early Devel Run"

export HYDRA_FULL_ERROR=1
python train.py experiment=imagenet/baseline model/encoder=convatto model.encoder.pretrained=False trainer.max_steps=100000
