"""Usage:
    $ export WANDB_PROJECT=yourproject
    $ python power.py
    # or override config
    $ python power.py project=yourproject
"""
from tqdm import tqdm

import hydra
import wandb
from omegaconf import DictConfig

import pandas as pd


@hydra.main(config_path="config", config_name="power", version_base="1.3")
def main(cfg: DictConfig):

    api = wandb.Api(overrides=dict(project=cfg.project))

    total_kwh = 0
    for run in tqdm(api.runs()):
        sys = run.history(stream="events")

        try:
            # Compute the average power consumption. (Fill nans with previous value)
            watts = sys["system.gpu.0.powerWatts"].ffill().mean()

            # Compute the duration of the experiment
            stamps = pd.to_datetime(sys["_timestamp"], unit="s")
            first, last = stamps.iloc[0], stamps.iloc[-1]
            duration = last - first

            # Convert watts into kilowatts and duration into hours
            kilowatts = watts / 1_000
            hours = duration.total_seconds() / (60 * 60)

            total_kwh += kilowatts * hours

        except KeyError:
            continue

    kg_per_kwh = cfg.grams_per_kwh / 1000
    co2 = total_kwh * kg_per_kwh
    euros_per_kwh = cfg.cents_per_kwh / 100
    euros = total_kwh * euros_per_kwh

    print(f"WandB tracked experiments used {total_kwh:.02f} kWh.")
    print(f"At an average of {cfg.grams_per_kwh} grams/kWh this emitted {co2:.02f} kg CO₂")
    print(f"At {cfg.cents_per_kwh} cents/kWh this costs {euros:.02f}€")


if __name__ == "__main__":
    main()
