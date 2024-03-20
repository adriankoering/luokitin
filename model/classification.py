import hydra

import torch
import torchmetrics as tm
import torchmetrics.classification as tmc

from .base import BaseModel


class ClassificationModel(BaseModel):
    def __init__(
        self,
        encoder,
        decoder=None,
        compile: bool = True,
        freeze_encoder: bool = False,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.save_hyperparameters(ignore="datamodule", logger=False)

        self.model = hydra.utils.instantiate(encoder, num_classes=self.num_classes)

        if freeze_encoder:
            self.freeze_encoder()
        if compile:
            self.model = torch.compile(self.model)

        acc = tmc.MulticlassAccuracy(
            self.num_classes, ignore_index=self.ignore_index, average="micro"
        )
        top5 = tmc.MulticlassAccuracy(
            self.num_classes, ignore_index=self.ignore_index, average="micro", top_k=5
        )
        collection = tm.MetricCollection({"acc": acc, "top5": top5})
        self.train_metrics = collection.clone(prefix="train/")

        # TODO: need less memory-intenstive implementation for segmentation
        # only apply ece on validation / test, because it OOM during training
        # ece = tmc.MulticlassCalibrationError(
        #     num_classes = self.num_classes,
        #     ignore_index = self.ignore_index,
        # )
        # collection.add_metrics({"ece": ece})
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
