from hydra.utils import instantiate
from omegaconf import OmegaConf

import torch
import torchmetrics as tm
import torchmetrics.classification as tmc

from .base import BaseModel


class ClassificationModel(BaseModel):
    def __init__(self, encoder, decoder=None, metrics=None, compile: bool = True, freeze_encoder: bool = False, **kwargs):
        super().__init__(**kwargs)
        self.save_hyperparameters(logger=False)

        self.model = instantiate(encoder)
        if freeze_encoder:
            self.freeze_encoder()
        if compile:
            self.model = torch.compile(self.model)

        collection = tm.MetricCollection(OmegaConf.to_container(instantiate(metrics)))
        self.train_metrics = collection.clone(prefix="train/")
        self.validation_metrics = collection.clone(prefix="validation/")
        self.test_metrics = collection.clone(prefix="test/")

        self.log_cfg = dict(on_step=False, on_epoch=True, sync_dist=True)

    def forward(self, images):
        return self.model(images)

    def training_step(self, batch, batch_idx):
        loss, logits, labels = self.step(batch)
        self.log("train/loss", loss, on_step=True, prog_bar=True)
        self.log_dict(self.train_metrics(logits, labels), **self.log_cfg)
        return dict(loss=loss, logits=logits)

    def validation_step(self, batch, batch_idx):
        loss, logits, labels = self.step(batch)
        self.log("validation/loss", loss, **self.log_cfg)
        self.log_dict(
            self.validation_metrics(logits, labels),
            **self.log_cfg,
            prog_bar=True,
        )
        return dict(loss=loss, logits=logits)

    def test_step(self, batch, batch_idx):
        loss, logits, labels = self.step(batch)
        self.log("test/loss", loss, **self.log_cfg)
        self.log_dict(self.test_metrics(logits, labels), **self.log_cfg)
        return dict(loss=loss, logits=logits)

    def freeze_encoder(self):
        for param in self.model.parameters():
            param.requires_grad = False
        for param in self.model.fc.parameters():
            param.requires_grad = True
