#!/usr/bin/python3
from argparse import ArgumentParser
from pathlib import Path
from rich.progress import track

import torch

import nvidia.dali as dali
from nvidia.dali import pipeline_def
import nvidia.dali.fn as fn
import nvidia.dali.types as types
from nvidia.dali.plugin.pytorch import DALIClassificationIterator, LastBatchPolicy


# TODO: compute all statistics all at once
#  0. image shapes
#  1. mean / std
#  2. label frequencies

def parse_arguments():
    parser = ArgumentParser()
    parser.add_argument(
        "webds",
        nargs="+",
        type=str,
    )
    parser.add_argument(
        "--extensions",
        default=("img", "lbl"),
        nargs="+",
        type=str,
    )
    parser.add_argument(
        "--in-memory",
        action="store_true"
    )
    return parser.parse_args()


def tar_to_index_files(tarfiles):
    return [Path(tar).with_suffix(".idx") for tar in tarfiles]



class LightningWrapper(DALIClassificationIterator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __next__(self):
        out = super().__next__()
        # DDP is used so only one pipeline per process also we need to transform dict
        # returned by DALIClassificationIterator to iterable and squeeze the lables
        out = out[0]
        return [out[k] for k in self.output_map]



@pipeline_def
def pipeline(webdatasets, extensions):

    index_files = tar_to_index_files(webdatasets)

    jpeg, label = dali.fn.readers.webdataset(
        paths=webdatasets,
        index_paths=index_files,
        ext=extensions,
        missing_component_behavior="error",
        name="Reader",
    )

    image = fn.decoders.image(jpeg, device="mixed", output_type=types.RGB)
    image = fn.cast(image, dtype=types.FLOAT) / 255.
    return image, label.gpu()



# Reference (in-memory) implementation
def stacked_std_mean(webdatasets, extensions, dim=(1, 2)):
    pipe = pipeline(
        webdatasets, extensions, batch_size=1, num_threads=10, device_id=0
    )

    for img, lbl in track(LightningWrapper(pipe, reader_name="Reader")):
        # user provides reduction dims per sample
        # we need to account for the additional dim
        dim = [0] + [d + 1 for d in dim]
        images = torch.stack([img for img, lbl in ds], dim=0)
        return torch.std_mean(images, dim)


# Referenc-ish online implementation
def summing_std_mean(webdatasets, extensions, dim=(0, 1, 2)):

    pipe = pipeline(
        webdatasets, extensions, batch_size=1, num_threads=10, device_id=0
    )

    # works, but might become numerically unstable for lots of entries
    c = 0
    s = s2 = None

    for img, lbl in track(LightningWrapper(pipe, reader_name="Reader")):
        
        B, H, W, C = img.shape
        c += H * W
        if s is None:
            s = img.sum(dim).double()
            s2 = img.pow(2).sum(dim).double()
        else:
            s += img.sum(dim)
            s2 += img.pow(2).sum(dim)

    mean = s / c
    var = (s2 / (c - 1)) - mean**2

    return var.sqrt(), mean


def repr(x):
    r, g, b = x
    return f"[{r:2.4f}, {g:2.4f}, {b:2.4f}]"


def pprint(std, mean):
    print("mean: ", repr(mean))
    print("std:  ", repr(std))


def main(args):

    std_mean = stacked_std_mean if args.in_memory else summing_std_mean
    pprint(*std_mean(args.webds, args.extensions))


if __name__ == "__main__":
    main(args=parse_arguments())
