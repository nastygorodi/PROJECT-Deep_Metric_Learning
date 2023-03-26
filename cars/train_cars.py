import hydra
from omegaconf import DictConfig

from utils.train import pl_train


@hydra.main(config_path="configs", config_name="train_cars.yaml")
def main_hydra(cfg: DictConfig) -> None:
    pl_train(cfg)


if __name__ == "__main__":
    main_hydra()
