from typing import Any, Tuple, Callable, List, Optional

import torch
from torch.utils.data import Dataset, SequentialSampler, RandomSampler, BatchSampler

class ImageTensorDataset(Dataset):
    def __init__(self, images, targets, transforms: Callable = None, target_transforms: Callable = None) -> None:
        assert len(images) == len(targets), "Size mismatch between tensors"
        self.images = images
        self.targets = targets

        self.image_transforms = transforms or torch.nn.Identity()
        self.target_transforms = target_transforms or torch.nn.Identity()

    def __getitem__(self, index):
        return (self.image_transforms(self.images[index]),
                self.target_transforms(self.targets[index]))
    
    def __len__(self):
        return len(self.images)


class TrivialDataLoader:
    """ For the ImageTensorDataset (with images and targets on device):
        - calling the default DataLoader num_workers=0 works, but is slow
        - calling the default DataLoader with num_workers=N yields a 
            RuntimeError: CUDA error: initialization error. Probably, because
            it tries to move the images onto the device again?
        Hence, here we create a simple DataLoader ourselves that just samples
        batch_indices and returns tensor slices from the dataset.
    """
    def __init__(self, dataset, batch_size=1, shuffle=False, sampler=None,
                 batch_sampler=None, num_workers=0, collate_fn=None,
                 pin_memory=False, drop_last=False, timeout=0,
                 worker_init_fn=None, *, prefetch_factor=2,
                 persistent_workers=False):
        
        self.dataset = dataset
        # batch_size used in batch_sampler
        if sampler is not None and shuffle:
            raise ValueError("sampler option is mutually exclusive with shuffle")
        self.sampler = sampler or (RandomSampler(dataset) if shuffle else SequentialSampler(dataset))
        self.batch_sampler = batch_sampler or BatchSampler(self.sampler, 
                                                           batch_size=batch_size, 
                                                           drop_last=drop_last)

        # assume num_workers == 0
        # ignore collate_fn, because we slice from a single (image or label) tensor 
        # ignore pin_memory
        # drop_last already used in batch_sampler
        # ignore timeout
        # ignore worker_init_fn -> no worker
        # ignore prefetch_factor -> compute's done on device anyway
        # ignore persistent_workers
        # torch.set_vital('Dataloader', 'enabled', 'True')  # type: ignore[attr-defined]
        
    def __iter__(self):
        for batch_indices in self.batch_sampler:
            yield self.dataset[batch_indices]
