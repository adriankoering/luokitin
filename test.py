import hydra
from omegaconf import DictConfig
from hydra.utils import instantiate

from lightning.pytorch.loggers import CSVLogger

@hydra.main(config_path="config", config_name="test", version_base="1.3")
def main(cfg: DictConfig):
    dm = instantiate(cfg.dataset)
    ensemble = instantiate(cfg.model, datamodule=dm)

    loggers = [CSVLogger(save_dir="csv", name="", version="")]
    trainer = instantiate(cfg.trainer, logger=loggers)
    trainer.test(ensemble, dm)


if __name__ == "__main__":
    main()