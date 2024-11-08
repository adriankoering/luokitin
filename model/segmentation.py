import hydra

import torch
torch.set_float32_matmul_precision('high')

from .base import BaseModel

class SegmentationModel(BaseModel):
    def __init__(self, encoder, decoder = None, compile=True, **kwargs):
        super().__init__(**kwargs)
        self.save_hyperparameters(ignore="datamodule", logger=False)

        self.encoder = hydra.utils.instantiate(encoder, features_only=True, output_stride=16)
        self.decoder = hydra.utils.instantiate(decoder) # self.num_classes)

        if compile:
            self.encoder = torch.compile(self.encoder)
            self.decoder = torch.compile(self.decoder)


    def forward(self, images):
        embeddings = self.encoder(images)
        return self.decoder(embeddings)
