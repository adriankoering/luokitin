from rich.progress import track
from io import BytesIO
from PIL import Image
import subprocess
import struct

import torch
from torchvision.transforms.v2.functional import to_pil_image

import webdataset as wds

__all__ = [
    "convert_to_webdataset",
    "create_classification_example",
    "create_segmentation_example",
    "create_webdataset_index",
    "maybe_shuffle",
]


def maybe_shuffle(ds, split=None):
    split = split or ds.split  # Try to Read the split from dataset if none provided
    if split == "train":
        rng = torch.Generator().manual_seed(42)
        idx = torch.randperm(len(ds), generator=rng)
        # Not actually selecting a subset, but the
        # same number of elements in random order
        return torch.utils.data.Subset(ds, idx)

    return ds


def create_segmentation_example(key, image, label):
    image = to_pil_image(image) if not isinstance(image, Image.Image) else image
    label = to_pil_image(label) if not isinstance(label, Image.Image) else label

    jpg = BytesIO()
    image.save(jpg, format="jpeg", quality=95)

    png = BytesIO()
    label.save(png, format="png")

    return {"__key__": str(int(key)), "img": jpg.getvalue(), "lbl": png.getvalue()}


def create_classification_example(key, image, label):
    image = to_pil_image(image) if not isinstance(image, Image.Image) else image

    jpg = BytesIO()
    image.save(jpg, format="jpeg", quality=95)

    return {
        "__key__": str(int(key)),
        "img": jpg.getvalue(),
        "lbl": struct.pack("i", label),  # binary encoded (for dali)
        "cls": label,  # ascii encoded (for webdataset)
    }

def create_rgbd_classification_example(key, image, depth, label):
    image = to_pil_image(image) if not isinstance(image, Image.Image) else image

    jpg = BytesIO()
    image.save(jpg, format="jpeg", quality=95)

    png = BytesIO()
    depth.save(png, format="png")

    return {
        "__key__": str(int(key)),
        "img": jpg.getvalue(),
        "dep": png.getvalue(),
        "lbl": struct.pack("i", label),  # binary encoded (for dali)
        "cls": label,  # ascii encoded (for webdataset)
    }


# # Test-Case of preprocessing without writing
# def convert_to_webdataset(
#     ds, create_sample, outdir, pattern, maxcount=None, maxsize=None
# ):
#     for e in ds:
#         print(type(e), len(e))

#         if len(e) == 2:
#             image, label = e
#         elif len(e) == 3:
#             image, depth, label = e

#         print(type(image), type(label))
#         try:
#             print(image.shape, label.shape)
#             print(image.dtype, label.dtype)
#         except:
#             pass
#         break


def convert_to_webdataset(
    ds, create_example, outdir, pattern, maxcount=None, maxsize=None
):
    outdir.mkdir(parents=True, exist_ok=True)

    kwargs = dict()
    if maxcount:
        kwargs["maxcount"] = maxcount
    if maxsize:
        kwargs["maxsize"] = maxsize

    fpath = outdir / pattern
    with wds.ShardWriter(str(fpath), **kwargs) as writer:
        for key, sample in enumerate(track(ds)):
            writer.write(create_example(key, *sample))


def create_webdataset_index(outdir):
    print("Creating WebDataset Indices ...")
    for ds in outdir.glob("*.tar"):
        subprocess.run(["wds2idx", str(ds)])
