import hydra
from omegaconf import DictConfig

import torch

from lightning.pytorch import LightningModule, LightningDataModule

class BaseModel(LightningModule):
    def __init__(self, datamodule, optimizer, loss, learning_rate_scheduler = None):
        super().__init__()

        self.num_classes = datamodule.num_classes
        self.ignore_index = datamodule.ignore_index

        self.loss_fn = hydra.utils.instantiate(
            loss,
            # num_classes=self.num_classes,
            ignore_index=self.ignore_index,
        )

        # store the config to instantiate later
        self.opt_cfg = optimizer
        self.lrs_cfg = learning_rate_scheduler

        # TODO: save some hparams?
        # self.save_hyperparameters()

    def step(self, batch):
        images, labels = batch
        images = images.to(memory_format=torch.channels_last)
        logits = self(images)
        return self.loss_fn(logits, labels), logits, labels

    def configure_optimizers(self):
        opt = hydra.utils.instantiate(self.opt_cfg, self.parameters())
        scheduler = hydra.utils.instantiate(self.lrs_cfg, opt)
        schedule = {
            "scheduler": scheduler,
            "interval": "step",  # or 'epoch'
            "frequency": 1,
        }

        return [[opt], [schedule]] if scheduler else opt
