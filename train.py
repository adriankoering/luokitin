import hydra
from omegaconf import DictConfig, OmegaConf

@hydra.main(config_path="config", config_name="train", version_base="1.3")
def main(cfg: DictConfig):

    # dm = hydra.utils.instantiate(cfg.dataset)
    # model = hydra.utils.instantiate(cfg.model, datamodule=dm)
    # model = model.to(memory_format=torch.channels_last)


    trainer = hydra.utils.instantiate(cfg.trainer, logger=[], callbacks=[])
    # trainer.fit(model, dm)
    # trainer.test(model, dm)

if __name__ == "__main__":
    main()