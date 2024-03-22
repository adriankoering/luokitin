
from pathlib import Path
from typing import List

from hydra.utils import to_absolute_path
from omegaconf import OmegaConf

import torch
from torch import nn
from torchinfo import summary

from lightning.pytorch import LightningModule, LightningDataModule

import importlib

def import_cls(target: str):
    module, cls = target.rsplit(".", maxsplit=1)
    return getattr(importlib.import_module(module), cls)


class EnsembleModel(LightningModule):
    def __init__(self, run_dirs: List[str], config_suffix: str=".hydra/config.yaml", ckpt_suffix: str="ckpt/last.ckpt"):
        super().__init__()

        assert run_dirs, "Ensembled didnt receive any member models"
        abs_dirs = [Path(to_absolute_path(Path(run_dir).expanduser())) for run_dir in run_dirs]
        assert all([run_dir.exists() for run_dir in abs_dirs]), "Resolving paths to absolute paths failed"
        cfg_files = [run_dir / config_suffix for run_dir in abs_dirs]
        assert all([cfg_file.exists() for cfg_file in cfg_files]), "Ensemble Member is missing its config"
        ckpt_files = [run_dir / ckpt_suffix for run_dir in abs_dirs]
        assert all([ckpt_file.exists() for ckpt_file in ckpt_files]), "Ensemble Member is missing its ckpt"
        model_cfgs = [OmegaConf.load(cfg_file).model for cfg_file in cfg_files]

        self.models = nn.ModuleList([])
        for model_cfg, ckpt_file in zip(model_cfgs, ckpt_files):
            Model = import_cls(model_cfg._target_)
            model = Model.load_from_checkpoint(ckpt_file)
            self.models.append(model)
            # TODO: memory format?

        self.loss_fn = self.models[0].loss_fn
        self.validation_metrics = self.models[0].validation_metrics
        self.test_metrics = self.models[0].test_metrics

    # def mysummary(self, model):
    #     s = summary(
    #         model,
    #         input_data=torch.randn(1, 3, 1024, 2048).cuda(),
    #         col_names=["num_params", "mult_adds"],
    #         verbose=0,
    #     )
    #     return np.array((s.total_params, s.total_mult_adds))

    # def stats(self):
    #     num_params, mult_adds = sum([self.mysummary(model) for model in self.models])
    #     return dict(num_params=num_params, mult_adds=mult_adds)

    def forward(self, image):
        dim = (0, 2, 3)
        # print(image.amin(dim), image.mean(dim), image.amax(dim))
        out = torch.stack([model(image) for model in self.models])
        return out.mean(dim=0)

    def step(self, batch):
        images, labels = batch
        logits = self(images)
        return self.loss_fn(logits, labels), logits, labels
    
    def validation_step(self, batch, batch_idx):
        loss, logits, labels = self.step(batch)
        log_cfg = dict(on_step=False, on_epoch=True)
        self.log("validation/loss", loss, **log_cfg)
        self.log_dict(self.validation_metrics(logits, labels), **log_cfg)
        return dict(loss=loss, logits=logits)

    def test_step(self, batch, batch_idx):
        loss, logits, labels = self.step(batch)
        log_cfg = dict(on_step=False, on_epoch=True)
        self.log("test/loss", loss, **log_cfg)
        self.log_dict(self.test_metrics(logits, labels), **log_cfg)
        return dict(loss=loss, logits=logits)

    def predict_step(self, batch, batch_idx):
        images, labels = batch
        return self(images)
