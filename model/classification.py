import hydra

import torch
import torchmetrics as tm
import torchmetrics.classification as tmc

from .base import BaseModel

class ClassificationModel(BaseModel):
    def __init__(self, encoder, decoder = None, compile: bool = True, **kwargs):
        super().__init__(**kwargs)

        self.model = hydra.utils.instantiate(encoder, in_chans=1, num_classes=self.num_classes)
        self.model = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(28*28, self.num_classes)
        )
        if compile:
            self.model = torch.compile(self.model)

        acc = tmc.MulticlassAccuracy(self.num_classes, ignore_index=self.ignore_index)
        collection = tm.MetricCollection({"acc": acc})
        self.train_metrics = collection.clone(prefix="train/")

        # TODO: need less memory-intenstive implementation for segmentation
        # only apply ece on validation / test, because it OOM during training
        ece = tmc.MulticlassCalibrationError(
            num_classes = self.num_classes,
            ignore_index = self.ignore_index,
        )
        collection.add_metrics({"ece": ece})
        self.validation_metrics = collection.clone(prefix="validation/")
        self.test_metrics = collection.clone(prefix="test/")

        self.log_cfg = dict(on_step=False, on_epoch=True, sync_dist=True)

    def forward(self, images):
        return self.model(images)
    
    def training_step(self, batch, batch_idx):
        loss, logits, labels = self.step(batch)
        self.log("train/loss", loss, **self.log_cfg)
        self.log_dict(self.train_metrics(logits, labels), **self.log_cfg)
        return dict(loss=loss, logits=logits)

    def on_train_epoch_end(self) -> None:
        self.train_metrics.reset()  # reset these explicitly
        return super().on_train_epoch_end()

    def validation_step(self, batch, batch_idx):
        loss, logits, labels = self.step(batch)
        self.log("validation/loss", loss, **self.log_cfg)
        self.log_dict(
            self.validation_metrics(logits, labels),
            **self.log_cfg,
            prog_bar=True,
        )
        return dict(loss=loss, logits=logits)

    def on_validation_epoch_end(self) -> None:
        self.validation_metrics.reset()  # reset these explicitly
        return super().on_train_epoch_end()

    def test_step(self, batch, batch_idx):
        loss, logits, labels = self.step(batch)
        self.log("test/loss", loss, **self.log_cfg)
        self.log_dict(self.test_metrics(logits, labels), **self.log_cfg)
        return dict(loss=loss, logits=logits)

    def on_test_epoch_end(self) -> None:
        self.test_metrics.reset()  # reset these explicitly
        return super().on_train_epoch_end()


    def forward(self, images):
        return self.model(images)