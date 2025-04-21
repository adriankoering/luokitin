from pathlib import Path
from argparse import ArgumentParser
from math import ceil

from torchvision.datasets import ImageFolder
from torchvision.transforms import transforms as transform_lib
from torch.nn import Identity

from utils import (
    convert_to_webdataset,
    create_classification_example,
    create_webdataset_index,
    maybe_shuffle,
)


def parse_arguments():
    args = ArgumentParser()
    # Input Parameters
    args.add_argument("data_dir", type=Path)
    args.add_argument("--splits", type=str, nargs="+", default=["train", "val", "test"])
    # Output Parameters
    args.add_argument(
        "--outdir", type=Path, default=Path("~/data/cumulus/webds/jpg").expanduser()
    )
    args.add_argument("--pattern", type=str, default="cumulus-{split}-%03d.tar")
    args.add_argument("--num-shards", type=int, default=4)
    # Processing Parameters
    args.add_argument("--image-size", type=int)

    return args.parse_args()


def main(args):
    data_dir = Path(args.data_dir).expanduser()

    if args.image_size is None:
        train_tfs = val_tfs = Identity()
    else:
        train_tfs = val_tfs = transform_lib.Resize(args.image_size)
    
    transforms = [train_tfs, val_tfs]
    for split, tf in zip(args.splits, transforms):
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
