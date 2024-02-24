import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate

import wandb
import torch
from lightning.pytorch.loggers import WandbLogger

import os

@hydra.main(config_path="config", config_name="train", version_base="1.3")
def main(cfg: DictConfig):
    hcfg = HydraConfig.get()
    dcfg = OmegaConf.to_container(cfg, resolve=True) if hcfg.mode == hydra.types.RunMode.RUN else None

    dm = instantiate(cfg.dataset)
    model = instantiate(cfg.model, datamodule=dm)
    model = model.to(memory_format=torch.channels_last)

    loggers = [WandbLogger(name=cfg.name, config=dcfg)]
    # TODO: check Rich-Progress-Bar on slurm?
    callbacks = [instantiate(cb_cfg) for _, cb_cfg in cfg.callbacks.items()]

    trainer = instantiate(cfg.trainer, logger=loggers, callbacks=callbacks)
    trainer.fit(model, dm)
    trainer.test(model, dm)

    wandb.finish() # required for multi-runs

if __name__ == "__main__":
    main()