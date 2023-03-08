import hydra
from omegaconf import DictConfig

from oml.lightning.entrypoints.validate import pl_val

from oml.registry.postprocessors import POSTPROCESSORS_REGISTRY
from postprocessing.top_freq import TopFrequencyPostprocessor

POSTPROCESSORS_REGISTRY['top_freq'] = TopFrequencyPostprocessor


@hydra.main(config_path="configs", config_name="val_cars.yaml")
def main_hydra(cfg: DictConfig) -> None:
    pl_val(cfg)


if __name__ == "__main__":
    main_hydra()
