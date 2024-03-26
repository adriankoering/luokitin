from pathlib import Path
from typing import Any, List, Optional, Tuple

import torch
from torch.utils.data import DataLoader as DataLoaderLike
from lightning.pytorch import LightningDataModule

from nvidia.dali import fn
from nvidia.dali import types
from nvidia.dali import pipeline_def
from nvidia.dali.plugin.pytorch import DALIClassificationIterator, LastBatchPolicy

__all__ = ["WebdalisetDataModule"]


def tar_to_index_files(tarfiles):
    return [tar.with_suffix(".idx") for tar in tarfiles]


def find_files(root, pattern):
    return sorted(Path(root).expanduser().glob(pattern))


def to_dtype(extension):
    return {"img": types.UINT8, "dep": types.UINT8, "lbl": types.INT32}[extension]


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
        extensions: Tuple[str] = None,  # ("img", "lbl"),
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

        self.extensions = tuple(extensions) if extensions else None

        self.num_workers = num_workers
        self.batch_size = batch_size
        self.image_size = image_size
        # Stats measured from [0, 1] but applied on [0, 255]
        self.mean = 255 * torch.as_tensor(mean)
        self.std = 255 * torch.as_tensor(std)
        self._classes = classes
        self._ignore = ignore_index
        self.crop = tuple(crop) if crop else None

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
        rnd_mirror = fn.random.coin_flip()
        # rnd_crop = fn.random.uniform(dtype=types.INT16, range=[0, 65536])

        index_files = tar_to_index_files(webdatasets)
        color, *depth, label = fn.readers.webdataset(
            paths=webdatasets,
            index_paths=index_files,
            ext=self.extensions,
            dtypes=list(map(to_dtype, self.extensions)),
            shard_id=self.trainer.local_rank,
            num_shards=self.trainer.world_size,
            dont_use_mmap=True,
            missing_component_behavior="error",
            prefetch_queue_depth=2,
            read_ahead=True,
            name="Reader",
            random_shuffle=True,
            initial_fill=1000,
        )

        # image = fn.decoders.image(color, device="mixed", output_type=types.RGB)
        image = fn.decoders.image_random_crop(
            color,
            device="mixed",
            output_type=types.RGB,
            num_attempts=10,
            random_area=[0.2, 1.0],
            random_aspect_ratio=[0.75, 1.333333],
            # seed=rnd_crop,
        )
        if depth:
            assert (
                len(depth) == 1
            ), f"Expected only a single depth image. Found {len(depth)=}"
            # Load depth image (and concatenate to color)
            depth = fn.decoders.image(depth[0], device="mixed", output_type=types.GRAY)
            image = fn.cat(image, depth, axis=-1)

        # if self.crop is not None:
        #     image = fn.resize(image, size=self.crop, mode="not_smaller")
        image = fn.resize(image, size=self.crop)
        image = fn.crop_mirror_normalize(
            image,
            dtype=types.FLOAT,
            mean=self.mean,
            std=self.std,
            output_layout="CHW",
            mirror=rnd_mirror,
            # crop=self.crop,
        )

        label = fn.cast(label, dtype=types.INT64)
        label = fn.squeeze(label, axes=0)
        label = label.gpu()

        return image, label

    @pipeline_def
    def evaluation_pipeline(self, webdatasets):
        index_files = tar_to_index_files(webdatasets)

        color, *depth, label = fn.readers.webdataset(
            paths=webdatasets,
            index_paths=index_files,
            ext=self.extensions,
            dtypes=list(map(to_dtype, self.extensions)),
            shard_id=self.trainer.local_rank,
            num_shards=self.trainer.world_size,
            dont_use_mmap=True,
            missing_component_behavior="error",
            prefetch_queue_depth=2,
            read_ahead=True,
            name="Reader",
        )

        image = fn.decoders.image(color, device="mixed", output_type=types.RGB)
        # if depth:
        #     assert len(depth) == 1, f"Expected only a single depth image. Found {len(depth)=}"
        #     # Load depth image (and concatenate to color)
        #     depth = fn.decoders.image(depth[0], device="mixed", output_type=types.GRAY)
        #     image = fn.cat(image, depth, axis=-1)

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
