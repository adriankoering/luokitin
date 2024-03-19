from pathlib import Path
from typing import Any, List, Optional, Tuple

import torch
from torch.utils.data import DataLoader as DataLoaderLike
from lightning.pytorch import LightningDataModule

import nvidia.dali as dali
from nvidia.dali import pipeline_def
import nvidia.dali.fn as fn
import nvidia.dali.types as types
from nvidia.dali.plugin.pytorch import DALIClassificationIterator, LastBatchPolicy

__all__ = ["WebdalisetDataModule"]


def tar_to_index_files(tarfiles):
    return [tar.with_suffix(".idx") for tar in tarfiles]


def find_files(root, pattern):
    return sorted(Path(root).expanduser().glob(pattern))


class LightningWrapper(DALIClassificationIterator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __next__(self):
        out = super().__next__()
        # DDP is used so only one pipeline per process also we need to transform dict
        # returned by DALIClassificationIterator to iterable and squeeze the lables
        out = out[0]
        return [out[k] for k in self.output_map]


class WebdalisetDataModule(LightningDataModule):
    def __init__(
        self,
        data_dir: str,
        num_workers: int = 8,
        batch_size: int = 32,
        image_size: int = 1024,
        mean: Optional[tuple] = (0.0, 0.0, 0.0),
        std: Optional[tuple] = (1.0, 1.0, 1.0),
        classes: Optional[Tuple[str]] = None,
        ignore_index: Optional[int] = None,
        extensions: Tuple[str] = ("img", "lbl"),
        crop: Optional[Tuple[int]] = None,
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
        # Stats measured from [0, 1] but applied on [0, 255]
        self.mean = 255 * torch.as_tensor(mean)
        self.std = 255 * torch.as_tensor(std)
        self._classes = classes
        self._ignore = ignore_index
        self.crop = tuple(crop)

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

    @pipeline_def
    def train_pipeline(self, webdatasets):
        mirror = fn.random.coin_flip()

        index_files = tar_to_index_files(webdatasets)

        jpeg, label = dali.fn.readers.webdataset(
            paths=webdatasets,
            index_paths=index_files,
            ext=self.extensions,
            dtypes=[types.UINT8, types.INT32],
            shard_id=self.trainer.local_rank,
            num_shards=self.trainer.world_size,
            dont_use_mmap=True,
            missing_component_behavior="error",
            prefetch_queue_depth=2,
            read_ahead=True,
            name="Reader",
            # TODO: shuflfe once training longer
            # random_shuffle=True,
            # initial_fill=1000,
        )

        image = fn.decoders.image(jpeg, device="mixed", output_type=types.RGB)
        if self.crop is not None:
            image = fn.resize(image, size=self.crop, mode="not_smaller")
        image = fn.crop_mirror_normalize(
            image,
            dtype=types.FLOAT,
            mean=self.mean,
            std=self.std,
            output_layout="CHW",
            mirror=mirror,
            crop=self.crop,
        )

        label = fn.cast(label, dtype=types.INT64)
        label = fn.squeeze(label, axes=0)
        label = label.gpu()

        return image, label

    @pipeline_def
    def evaluation_pipeline(self, webdatasets):
        index_files = tar_to_index_files(webdatasets)

        jpeg, label = dali.fn.readers.webdataset(
            paths=webdatasets,
            index_paths=index_files,
            ext=self.extensions,
            dtypes=[types.UINT8, types.INT32],
            shard_id=self.trainer.local_rank,
            num_shards=self.trainer.world_size,
            dont_use_mmap=True,
            missing_component_behavior="error",
            prefetch_queue_depth=2,
            read_ahead=True,
            name="Reader",
        )

        image = fn.decoders.image(jpeg, device="mixed", output_type=types.RGB)
        if self.crop is not None:
            image = fn.resize(image, size=self.crop, mode="not_smaller")
        image = fn.crop_mirror_normalize(
            image,
            dtype=types.FLOAT,
            mean=self.mean,
            std=self.std,
            output_layout="CHW",
            mirror=False,
            crop=self.crop,
        )

        label = fn.cast(label, dtype=types.INT64)
        label = fn.squeeze(label, axes=0)
        label = label.gpu()

        return image, label

    def train_dataloader(self) -> DataLoaderLike:
        B, N, R = self.batch_size, self.num_workers, self.trainer.local_rank
        train_pipe = self.train_pipeline(
            self.train_shards, batch_size=B, num_threads=N, device_id=R
        )

        return LightningWrapper(
            train_pipe,
            reader_name="Reader",
            last_batch_policy=LastBatchPolicy.PARTIAL,  # TODO: DROP for small batch sizes
        )

    def val_dataloader(self) -> DataLoaderLike:
        B, N, R = 2 * self.batch_size, self.num_workers, self.trainer.local_rank
        val_pipe = self.evaluation_pipeline(
            self.val_shards, batch_size=B, num_threads=N, device_id=R
        )

        return LightningWrapper(
            val_pipe,
            reader_name="Reader",
            last_batch_policy=LastBatchPolicy.PARTIAL,
        )

    def test_dataloader(self) -> DataLoaderLike:
        B, N, R = 2 * self.batch_size, self.num_workers, self.trainer.local_rank
        test_pipe = self.evaluation_pipeline(
            self.test_shards, batch_size=B, num_threads=N, device_id=R
        )

        return LightningWrapper(
            test_pipe,
            reader_name="Reader",
            last_batch_policy=LastBatchPolicy.PARTIAL,
        )
