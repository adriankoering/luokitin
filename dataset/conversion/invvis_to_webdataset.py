from typing import Callable, Optional
from pathlib import Path
from argparse import ArgumentParser
from functools import partial
from math import ceil

from omegaconf import OmegaConf

from torch.utils.data import Dataset
from torchvision.datasets import DatasetFolder, ImageFolder
from torchvision.transforms import transforms as transform_lib

from utils import (
    convert_to_webdataset,
    create_rgbd_classification_example,
    create_webdataset_index,
    maybe_shuffle,
)



def is_color_image(ipath: str, suffix: str):
    return Path(ipath).name.endswith(suffix)

def color_to_depth_image(ipath: str, suffix: str = None):
    ipath = Path(ipath)
    [fraction, inv, mode, ext] = ipath.suffixes
    # stem removes '.ext', now replace .mode with suffix
    return ipath.parent / Path(ipath.stem).with_suffix(suffix)


class InvvisDataset(ImageFolder):
    def __init__(self, root: str, transform: Optional[Callable]=None, color_suffix: str = ".color.jpg", depth_suffix: str = ".depth.png"):
        super().__init__(root, is_valid_file=partial(is_color_image, suffix=color_suffix), transform=transform)
        self.to_depth = partial(color_to_depth_image, suffix=depth_suffix)

    def __getitem__(self, idx):
        color, label = self.samples[idx]
        depth = self.to_depth(color)

        # Color images are gather by globbing valid image files - therefore only the depth image should be missing
        assert Path(color).exists() and Path(depth).exists(), f"Didnt find {depth} image (derived from {color})"

        color = self.loader(color)
        depth = self.loader(depth)

        if self.transform is not None:
            color = self.transform(color)
            depth = self.transform(depth)
        
        if self.target_transform is not None:
            label = self.target_transform(label)

        return (color, depth, label)     


def parse_arguments():
    args = ArgumentParser()
    args.add_argument(
        "--outdir", type=Path, default=Path("~/data/invvis/webds/rgbd").expanduser()
    )
    args.add_argument("--pattern", type=str, default="invvis-{split}-%03d.tar")
    args.add_argument("--num-shards", type=int, default=4)
    return args.parse_args()


def main(args):
    cfg = OmegaConf.load("config/dataset/invvis.yaml")

    data_dir = Path(cfg.data_dir).expanduser()

    splits = ["train", "val", "test"]
    transforms = [None, None, None]

    for split, tf in zip(splits, transforms):
        ds = InvvisDataset(data_dir / split, transform=tf)
        ds = maybe_shuffle(ds, split)

        maxcount = ceil(len(ds) / args.num_shards)
        convert_to_webdataset(
            ds,
            create_rgbd_classification_example,
            args.outdir,
            args.pattern.format(split=split),
            maxcount,
        )

    create_webdataset_index(args.outdir)


if __name__ == "__main__":
    main(args=parse_arguments())
