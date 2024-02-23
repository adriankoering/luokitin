import torch
import torchvision as tv
from torchvision import tv_tensors

from torch.utils.data import DataLoader
from lightning.pytorch import LightningDataModule

# tv.disable_beta_transforms_warning()
import torchvision.transforms.v2 as transform_lib

class CifarBase(LightningDataModule):
    def __init__(self, root, batch_size, num_workers, mean, std, classes, **kwargs):
        self.batch_size = batch_size
        self.num_workers = num_workers
        # mean/std when given by config are actually omegaconf-list
        self.mean, self.std = tuple(mean), tuple(std)
        self._classes = classes
        print("Loading Cifar Default")

    @property
    def classes(self):
        return self._classes

    @property
    def num_classes(self):
        return len(self.classes)

    @property
    def ignore_index(self):
        return None


class CifarDataModule(CifarBase):
    pass

class GPUDataModule(CifarBase):
    pass

class WebDSDataModule(CifarBase):
    pass
