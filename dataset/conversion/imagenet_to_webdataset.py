from pathlib import Path
from argparse import ArgumentParser
from math import ceil

from omegaconf import OmegaConf

from torchvision.datasets import ImageFolder, wrap_dataset_for_transforms_v2
from torchvision.transforms import transforms as transform_lib

from utils import (
    convert_to_webdataset,
    create_classification_example,
    create_webdataset_index,
    maybe_shuffle,
)


def parse_arguments():
    args = ArgumentParser()
    args.add_argument(
        "--outdir", type=Path, default=Path("~/data/imagenet/webds/jpg").expanduser()
    )
    args.add_argument("--pattern", type=str, default="imagenet-{split}-%03d.tar")
    args.add_argument("--num-shards", type=int, default=4)
    args.add_argument("--image-size", type=int, default=256)

    return args.parse_args()


def main(args):
    cfg = OmegaConf.load("config/dataset/imagenet.yaml")

    data_dir = Path(cfg.data_dir).expanduser()

    splits = ["train", "val"]
    train_tfs = val_tfs = transform_lib.Resize(args.image_size)
    transforms = [train_tfs, val_tfs]

    for split, tf in zip(splits, transforms):
        ds = ImageFolder(data_dir / split, transform=tf)
        ds = maybe_shuffle(ds, split)

        maxcount = ceil(len(ds) / args.num_shards)
        convert_to_webdataset(
            ds,
            create_classification_example,
            args.outdir,
            args.pattern.format(split=split),
            maxcount,
        )

    create_webdataset_index(args.outdir)


if __name__ == "__main__":
    main(args=parse_arguments())
