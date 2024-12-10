import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate

import wandb
import torch
from lightning.pytorch.loggers import WandbLogger

OmegaConf.register_new_resolver("eval", eval)
OmegaConf.register_new_resolver("len", lambda x: len(x))


@hydra.main(config_path="config", config_name="train", version_base="1.3")
def main(cfg: DictConfig):
    dcfg = OmegaConf.to_container(cfg, resolve=True)

    dm = instantiate(cfg.dataset)
    model = instantiate(cfg.model)
    # Channels Last is a couple of % faster and costs nothing,
    #   but a image.to(channels_last) in the forward pass
    model = model.to(memory_format=torch.channels_last)

    # Compared to WandbLogger(config=hcfg),
    # these two steps also work with wandb sweeps
    ## logger = WandbLogger(reinit=True)
    logger = instantiate(cfg.wandb, reinit=True)
    logger.experiment.config.setdefaults(dcfg)

    trainer = instantiate(cfg.trainer, logger=logger)
    trainer.fit(model, dm)
    # Since we specify num_steps instead of epochs,
    # training doesnt necessarily end with a validation run
    #  NOTE: recalculate val and test metrics at the end
    trainer.validate(model, dm)
    trainer.test(model, dm)

    wandb.finish()  # required for multi-runs


if __name__ == "__main__":
    main()
