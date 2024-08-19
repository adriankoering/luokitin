from typing import Any, Callable, Optional, List, Tuple
from pathlib import Path
from functools import partial
from PIL import Image

import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.transforms import v2 as transform_lib
from lightning.pytorch import LightningDataModule

__all__ = ["InvvisDataset", "RGBInvvisDataset", "DepthInvvisDataset", "RGBDInvvisDataset", "InvvisDataModule"]

def is_color_image(ipath: str, suffix: str):
    return Path(ipath).name.endswith(suffix)

def color_to_depth_image(ipath: str, suffix: str = None):
    ipath = Path(ipath)
    [fraction, inv, mode, ext] = ipath.suffixes
    # stem removes '.ext', now replace .mode with suffix
    return ipath.parent / Path(ipath.stem).with_suffix(suffix)

def pil_loader_asis(ipath):
    # based on torchvision.datasets.folder without enforcing RGB
    with open(ipath, "rb") as f:
        img = Image.open(f)
        img.load()
        return img



class InvvisDataset(ImageFolder):
    def __init__(self, root: str, color_suffix: str = ".color.jpg", depth_suffix: str = ".depth.png"):
        super().__init__(root, loader=pil_loader_asis, is_valid_file=partial(is_color_image, suffix=color_suffix))
        self.to_depth = partial(color_to_depth_image, suffix=depth_suffix)

        # my_classes = sorted(root.iterdir())
        # self.my_images = sorted([sorted(c.glob(f"*{color_suffix}")) for c in my_classes])


    def __getitem__(self, idx):
        color, label = self.samples[idx]
        depth = self.to_depth(color)

        # Color images are gather by globbing valid image files - therefore only the depth image should be missing
        assert Path(color).exists() and Path(depth).exists(), f"Didnt find {depth} image (derived from {color})"

        # store fpath in order
        paths = f"{str(color)}\n{str(depth)}"
        (Path("sorted") / f"{idx:06d}.path").write_text(paths)

        # print(color)
        # print(self.my_images[0][idx])
        # assert color == str(self.my_images[0][idx]), f"{color} at {idx} doesnt match mine"

        color = self.loader(color)
        depth = self.loader(depth)

        return (color, depth, label)

class RGBInvvisDataset(InvvisDataset):
    def __init__(self, root: str,  transform: Optional[Callable]=None, *args, **kwargs):
        super().__init__(root, *args, **kwargs)
        self.transform = transform

    def __getitem__(self, idx):
        color, depth, label = super().__getitem__(idx)

        if self.transform is not None:
            color = self.transform(color)
        
        if self.target_transform is not None:
            label = self.target_transform(label)

        return color, label
    
class DepthInvvisDataset(InvvisDataset):
    def __init__(self, root: str, transform: Optional[Callable]=None, *args, **kwargs):
        super().__init__(root, *args, **kwargs)
        self.transform = transform
    
    def __getitem__(self, idx):
        color, depth, label = super().__getitem__(idx)

        if self.transform is not None:
            depth = self.transform(depth)
        
        if self.target_transform is not None:
            label = self.target_transform(label)
            
        return depth, label

class RGBDInvvisDataset(InvvisDataset):
    def __init__(self, root: str, transform: Optional[Callable]=None, *args, **kwargs):
        super().__init__(root, *args, **kwargs)
        self.transform = transform

    def __getitem__(self, idx):
        color, depth, label = super().__getitem__(idx)

        # TODO: check how dali loads images to be equivalent here
        # print(color.mode, depth.mode)
        color = transform_lib.functional.pil_to_tensor(color).to(torch.uint8)
        depth = transform_lib.functional.pil_to_tensor(depth).to(torch.uint8)
        image = torch.concat((color, depth), dim=0) # silently upgraded to depth's int32 

        # print(color.min(), color.max(), color.dtype)
        # print(depth.min(), depth.max(), depth.dtype)
        # print(image.min(), image.max(), image.dtype)

        if self.transform is not None:
            image = self.transform(image)
        
        if self.target_transform is not None:
            label = self.target_transform(label)

        return image, label


class InvvisDataModule(LightningDataModule):
    MODES = ("RGB", "Depth", "RGBD")
    
    def __init__(
            self,
            data_dir: str,
            mode: str,
            num_workers: int = 8,
            batch_size: int = 32,
            mean: Optional[tuple] = (0.0, 0.0, 0.0),
            std: Optional[tuple] = (1.0, 1.0, 1.0),
            classes: Optional[Tuple[str]] = None,
            ignore_index: Optional[int] = None,
            *args: Any,
            **kwargs: Any,
        ) -> None:
        super().__init__()

        self.data_dir = Path(data_dir).expanduser()
        
        assert mode in self.MODES, f"Provided {mode=} not in {self.MODES=}"
        self.DataSet = globals()[mode + "InvvisDataset"]

        self.num_workers = num_workers
        self.batch_size = batch_size
        self.mean = mean
        self.std = std

        self._classes = classes
        self._ignore = ignore_index

        self.preprocess = transform_lib.Compose([
            transform_lib.PILToTensor(),
            transform_lib.ToDtype(torch.float32, scale=True),
            transform_lib.Normalize(self.mean, self.std)
        ])
        # TODO: Augmentations?

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

    def train_dataloader(self) -> DataLoader:
        self.train_ds = self.DataSet(self.data_dir / "train", transform=self.preprocess)
        return DataLoader(
            self.train_ds, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers,
        )

    def val_dataloader(self) -> DataLoader:
        self.val_ds = self.DataSet(self.data_dir / "val", transform=self.preprocess)
        return DataLoader(
            self.val_ds, batch_size=2 * self.batch_size, num_workers=self.num_workers
        )

    def test_dataloader(self) -> DataLoader:
        self.test_ds = self.DataSet(self.data_dir / "test", transform=self.preprocess)
        return DataLoader(
            self.test_ds, batch_size=2 * self.batch_size, num_workers=self.num_workers
        )
    
    def predict_dataloader(self) -> DataLoader:
        self.pred_ds = self.DataSet(self.data_dir / "val", transform=self.preprocess)
        return DataLoader(
            self.pred_ds, batch_size=2 * self.batch_size, num_workers=self.num_workers
        )


if __name__ == "__main__":
    from torchvision.transforms.v2 import PILToTensor
    for image, label in RGBDInvvisDataset("/home/koering/data/invvis/extra", transform=PILToTensor()):
        print(image.shape, label)
        break

    dm = InvvisDataModule("/home/koering/data/invvis", mode="RGBD")

    for image, label in dm.predict_dataloader():
        print(image.shape, label.shape)
        break


