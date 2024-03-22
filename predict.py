import hydra
from lightning import LightningModule, Trainer
from omegaconf import DictConfig
from hydra.utils import instantiate


import torch
import numpy as np
from lightning.pytorch.callbacks import BasePredictionWriter
from pathlib import Path
class ImageSorter(BasePredictionWriter):
    def __init__(self, output_dir):
        super().__init__(write_interval="batch_and_epoch")
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)

        self.mean= torch.as_tensor([0.1605, 0.2303, 0.2896, 0.8826]).view(-1, 1, 1)
        self.std=torch.as_tensor([0.2156, 0.2101, 0.2017, 0.2881]).view(-1, 1, 1)


    def write_on_batch_end(self, trainer, pl_module, prediction, batch_indices, batch, batch_idx, dataloader_idx):
        images, _ = batch
        for n, (image, label) in enumerate(zip(images, prediction)):
            # denormalize and write image with prediction as fname
            self.mean = self.mean.to(image.device)
            self.std = self.std.to(image.device)
            # print(image.shape, label.shape)
            # print(image.amin(dim=(1, 2)), image.amax(dim=(1, 2)))
            image = image.mul(self.std).add(self.mean)
            # print(image.amin(dim=(1, 2)), image.amax(dim=(1, 2)))
            break

    
    def write_on_epoch_end(self, trainer, pl_module, predictions, batch_indices):
        predictions = torch.concat(predictions).argmax(dim=1).cpu().numpy()
        np.savetxt(self.output_dir / "preds.csv", predictions, fmt="%i")



@hydra.main(config_path="config", config_name="predict", version_base="1.3")
def main(cfg: DictConfig):
    dm = instantiate(cfg.dataset)
    ensemble = instantiate(cfg.model)

    trainer = instantiate(cfg.trainer, callbacks=[ImageSorter("sorted")], logger=None)
    trainer.predict(ensemble, dm)


if __name__ == "__main__":
    main()