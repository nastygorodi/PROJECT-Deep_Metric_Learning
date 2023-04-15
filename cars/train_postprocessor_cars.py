import hydra
from omegaconf import DictConfig

from utils.train_postprocessor import pl_train_postprocessor

from oml.registry.postprocessors import POSTPROCESSORS_REGISTRY
from oml.registry.models import PAIRWISE_MODELS_REGISTRY
from postprocessing.multiple_emb import MultiEmbeddingsPostprocessor
from models.multi_query_concat import MultiConcat


POSTPROCESSORS_REGISTRY['multiple_emb'] = MultiEmbeddingsPostprocessor
PAIRWISE_MODELS_REGISTRY['multi_query_cat'] = MultiConcat

@hydra.main(config_path="configs", config_name="train_postprocessor_cars_multiple_emb.yaml")
def main_hydra(cfg: DictConfig) -> None:
    pl_train_postprocessor(cfg)


if __name__ == "__main__":
    main_hydra()