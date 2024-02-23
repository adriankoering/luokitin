import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.core.hydra_config import HydraConfig

import wandb
import torch
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor

import os

@hydra.main(config_path="config", config_name="train", version_base="1.3")
def main(cfg: DictConfig):
    hcfg = HydraConfig.get()
    dcfg = OmegaConf.to_container(cfg, resolve=True) if hcfg.mode == hydra.types.RunMode.RUN else None

    dm = hydra.utils.instantiate(cfg.dataset)
    model = hydra.utils.instantiate(cfg.model, datamodule=dm)
    model = model.to(memory_format=torch.channels_last)

    loggers = [WandbLogger(name=cfg.name, config=dcfg)]
    callbacks = [ModelCheckpoint("ckpt"), LearningRateMonitor()]

    trainer = hydra.utils.instantiate(cfg.trainer, logger=loggers, callbacks=callbacks)
    trainer.fit(model, dm)
    trainer.test(model, dm)

    wandb.finish() # required for multi-runs

if __name__ == "__main__":
    main()