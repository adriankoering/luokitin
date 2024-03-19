from pathlib import Path
from typing import Any, List, Optional, Tuple
from io import BytesIO
from PIL import Image

import torch
from torch.utils.data import DataLoader
import torchvision.transforms.v2 as transform_lib

from lightning.pytorch import LightningDataModule
import webdataset as wds


__all__ = ["WebDatasetClassificationModule"]


def find_files(root, pattern):
    return sorted(Path(root).expanduser().glob(pattern))


def decode_image(self, encoded_image):
    return Image.open(BytesIO(encoded_image)).convert("RGB")


def decode_label(self, encoded_label):
    return Image.open(BytesIO(encoded_label)).convert("L")


def decode_class(self, label):
    return label


class WebDatasetClassificationModule(LightningDataModule):
    def __init__(
        self,
        data_dir: str,
        num_workers: int = 8,
        batch_size: int = 32,
        image_size: int = 224,
        mean: Optional[tuple] = (0.0, 0.0, 0.0),
        std: Optional[tuple] = (1.0, 1.0, 1.0),
        classes: Optional[Tuple[str]] = None,
        ignore_index: Optional[int] = None,
        extensions: Tuple[str] = ("img", "lbl"),
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """
        Args:
            root: location of data containing leftImg8bit and gt{Fine, Coarse}
            num_workers: how many workers to use for loading data
            batch_size: number of examples per training/eval step
            image_size: image resolution for training/eval

            extensions: fields for image and label in the webdataset
        """
        super().__init__()

        self.train_shards = find_files(data_dir, "*train*.tar")
        assert self.train_shards, "No training data found"

        self.val_shards = find_files(data_dir, "*val*.tar")
        assert self.val_shards, "No validation data found"

        self.test_shards = find_files(data_dir, "*test*.tar")
        if not self.test_shards:
            print("No Test-Split available - re-using Validation Split")
            self.test_shards = self.val_shards

        self.extensions = tuple(extensions)

        self.num_workers = num_workers
        self.batch_size = batch_size
        self.image_size = image_size
        self.mean = torch.as_tensor(mean)
        self.std = torch.as_tensor(std)
        self._classes = classes
        self._ignore = ignore_index

    @property
    def classes(self) -> List[str]:
        """Return: the classes contained in the dataset"""
        return self._classes

    @property
    def num_classes(self) -> int:
        """Return: number of classes in the dataset"""
        return len(self._classes) if self._classes else 0

    @property
    def ignore_index(self) -> Optional[int]:
        return self._ignore

    def setup(self, stage: Optional[str] = None):
        self.train_ds = (
            wds.WebDataset(self.train_shards)
            .shuffle(1000)
            .map(self.decode_sample)
            .to_tuple(*self.extensions)
        )
        self.val_ds = (
            wds.WebDataset(self.val_shards)
            .map(self.decode_sample)
            .to_tuple(*self.extensions)
        )
        self.test_ds = (
            wds.WebDataset(self.test_shards)
            .map(self.decode_sample)
            .to_tuple(*self.extensions)
        )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_ds.batched(self.batch_size),
            batch_size=None,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_ds.batched(self.batch_size),
            batch_size=None,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def test_dataloader(self) -> DataLoader:

        return DataLoader(
            self.test_ds.batched(self.batch_size),
            batch_size=None,
            num_workers=self.num_workers,
            pin_memory=True,
        )
