#!/usr/bin/python3
from argparse import ArgumentParser
from pathlib import Path
from rich.progress import track

from PIL import Image

from nvidia.dali import fn
from nvidia.dali import types
from nvidia.dali import pipeline_def
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
    # NOTE: provide arguments delimited by space only:
    #   eg. --extensions img dep lbl
    parser.add_argument(
        "--extensions",
        default=("img", "lbl"),
        nargs="+",
        type=str,
    )
    parser.add_argument("--out-dir", type=Path, default=Path("vis"))
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

    color, *depth, label = fn.readers.webdataset(
        paths=webdatasets,
        index_paths=index_files,
        ext=extensions,
        dtypes=[types.RGB, types.INT32],
        missing_component_behavior="error",
        name="Reader",
    )

    # rnd_crop = fn.random.uniform(dtype=types.INT16, range=[0, 65536])

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

    crop = (112, 112)

    # image = fn.cast(image, dtype=types.FLOAT) / 255.0
    image = fn.crop_mirror_normalize(
        image,
        dtype=types.FLOAT,
        mean=[0, 0, 0],
        std=[255, 255, 255],
        output_layout="CHW",
        mirror=0,
        # crop=(112, 112),
    )

    image = fn.resize(image, size=crop)
    image = fn.cast(255 * image, dtype=types.UINT8)

    label = fn.cast(label, dtype=types.INT64)
    label = fn.squeeze(label, axes=0)
    label = label.gpu()

    return image, label.gpu()


def write_images(webdatasets, extensions, out_dir):
    out_dir.mkdir(exist_ok=True, parents=True)

    pipe = pipeline(webdatasets, extensions, batch_size=1, num_threads=10, device_id=0)
    for n, (img, lbl) in enumerate(track(LightningWrapper(pipe, reader_name="Reader"))):
        img, lbl = img[0], lbl[0]
        print(img.shape, img.dtype)

        # perm = (0, 1, 2)
        perm = (1, 2, 0)
        ipath = out_dir / f"{n:05d}_{lbl:03d}.jpg"
        Image.fromarray(img.permute(*perm).cpu().numpy()).save(ipath)

        if n == 10:
            break


def main(args):
    write_images(args.webds, args.extensions, args.out_dir)


if __name__ == "__main__":
    main(args=parse_arguments())
