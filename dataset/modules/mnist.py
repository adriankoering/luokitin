from typing import Any, Tuple, Callable, List, Optional
from pathlib import Path

import torch
import torchvision as tv

from torchvision.datasets import MNIST
from torch.utils.data import DataLoader, random_split
from lightning.pytorch import LightningDataModule

import torchvision.transforms.v2 as transform_lib

from .gpu_dataset import ImageTensorDataset, TrivialDataLoader

class MnistBase(LightningDataModule):
    def __init__(self, root, batch_size, num_workers, mean, std, classes, **kwargs):
        self.root = Path(root).expanduser()
        self.batch_size = batch_size
        self.num_workers = num_workers
        # mean/std when given by config are actually omegaconf-list
        self.mean, self.std = tuple(mean), tuple(std)
        self._classes = [str(c) for c in classes]
        # Some attributes lightning requires
        self.prepare_data_per_node=True
        self.allow_zero_length_dataloader_with_multiple_devices=False
        self._log_hyperparams = False

    @property
    def classes(self) -> List[str]:
        return self._classes

    @property
    def num_classes(self) -> int:
        return len(self.classes)
    
    @property
    def ignore_index(self) -> int:
        return -1
    
    def _default_augmentations(self) -> Callable:
        return transform_lib.Compose(
            [
                transform_lib.ToImage(),
                transform_lib.ToDtype(torch.float32, scale=True),
                transform_lib.RandomResizedCrop(28, scale=(0.8, 1.0), antialias=True),
                transform_lib.Normalize(mean=self.mean, std=self.std),
            ]
        )

    def _default_preprocessing(self) -> Callable:
        return transform_lib.Compose(
            [
                transform_lib.ToImage(),
                transform_lib.ToDtype(torch.float32, scale=True),
                transform_lib.Normalize(mean=self.mean, std=self.std),
            ]
        )


class MnistDataModule(MnistBase):
    def prepare_data(self):
        MNIST(self.root, train=True, download=True)
        MNIST(self.root, train=False, download=True)

    def setup(self, stage: str):
        rng = torch.Generator().manual_seed(42)

        # Assign train/val datasets for use in dataloaders
        if stage == "fit":
            mnist = MNIST(self.root, train=True, transform=self._default_preprocessing())
            train_count, val_count = 55_000, 5_000
            self.train_ds, self.val_ds = random_split(mnist, (train_count, val_count), rng)
            self.train_ds.transform = self._default_augmentations()
            self.val_ds.transform = self._default_preprocessing()

        # Assign test dataset for use in dataloader(s)
        if stage == "test":
            self.test_ds = MNIST(self.root, train=False, transform=self._default_preprocessing())

        if stage == "predict":
            self.pred_ds = MNIST(self.root, train=False, transform=self._default_preprocessing())

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=True,
            drop_last=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_ds, 
            batch_size=2 * self.batch_size, 
            num_workers=self.num_workers
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_ds, 
            batch_size=2 * self.batch_size, 
            num_workers=self.num_workers
        )


    def predict_dataloader(self):
        return DataLoader(self.mnist_predict, batch_size=self.batch_size)


class GPUDataModule(MnistBase):
    def prepare_data(self):
        MNIST(self.root, train=True, download=True)
        MNIST(self.root, train=False, download=True)

    def setup(self, stage: str, device: str = "cuda"):
        rng = torch.Generator().manual_seed(42)

        # Assign train/val datasets for use in dataloaders
        if stage == "fit":
            mnist = MNIST(self.root, train=True)

            train_count, val_count = 55_000, 5_000
            train_ds, val_ds = random_split(mnist, (train_count, val_count), rng)

            # Select training samples, add batch dimension and move to device
            train_imgs = mnist.data[train_ds.indices].unsqueeze(1).cuda()
            train_targets = mnist.targets[train_ds.indices].cuda()

            # Select validation samples, add batch dimension and move to device
            val_imgs = mnist.data[val_ds.indices].unsqueeze(1).cuda()
            val_targets = mnist.targets[val_ds.indices].cuda()

            # Accelerate train and validation datasets
            self.train_ds = ImageTensorDataset(train_imgs, train_targets,
                                               transforms=self._default_augmentations())
            self.val_ds = ImageTensorDataset(val_imgs, val_targets,
                                             transforms=self._default_preprocessing())

        # Keep test dataset on CPU for bitwise compatibility
        if stage == "test":
            self.test_ds = MNIST(self.root, train=False, transform=self._default_preprocessing())

        if stage == "predict":
            self.pred_ds = MNIST(self.root, train=False, transform=self._default_preprocessing())

    def train_dataloader(self) -> DataLoader:
        return TrivialDataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            drop_last=True,
        )

    def val_dataloader(self) -> DataLoader:
        return TrivialDataLoader(
            self.val_ds, 
            batch_size=2 * self.batch_size, 
            num_workers=self.num_workers
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_ds, 
            batch_size=2 * self.batch_size, 
            num_workers=self.num_workers
        )

    def predict_dataloader(self):
        return DataLoader(
            self.pred_ds, 
            batch_size=2 * self.batch_size,
            num_workers=self.num_workers
        )


class WebDSDataModule(MnistBase):
    pass