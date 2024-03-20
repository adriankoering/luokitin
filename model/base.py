import hydra
from omegaconf import DictConfig

import torch

from lightning.pytorch import LightningModule, LightningDataModule


class BaseModel(LightningModule):
    def __init__(self, optimizer, loss, learning_rate_scheduler=None):
        super().__init__()

        self.loss_fn = hydra.utils.instantiate(loss)
        # store the remaining configs to instantiate later
        self.opt_cfg = optimizer
        self.lrs_cfg = learning_rate_scheduler


    def step(self, batch):
        images, labels = batch
        images = images.to(memory_format=torch.channels_last)
        logits = self(images)
        return self.loss_fn(logits, labels), logits, labels

    def configure_optimizers(self):
        trainable = filter(lambda x: x.requires_grad, self.parameters())
        opt = hydra.utils.instantiate(self.opt_cfg, trainable)
        scheduler = hydra.utils.instantiate(self.lrs_cfg, opt)
        schedule = {
            "scheduler": scheduler,
            "interval": "step",  # or 'epoch'
            "frequency": 1,
        }

        return [[opt], [schedule]] if scheduler else opt
