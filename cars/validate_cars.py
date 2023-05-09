import hydra
from omegaconf import DictConfig

from oml.lightning.entrypoints.validate import pl_val

from oml.registry.postprocessors import POSTPROCESSORS_REGISTRY
from oml.registry.models import PAIRWISE_MODELS_REGISTRY
from postprocessing.top_freq import TopFrequencyPostprocessor
from postprocessing.multiple_emb import MultiEmbeddingsPostprocessor
from postprocessing.multiple_emb_with_top_freq import MultiEmbeddingsFreqPostprocessor
from models.multi_query_concat import MultiConcat
from models.multi_query_concat_attn import MultiConcatWithAttention


POSTPROCESSORS_REGISTRY['top_freq'] = TopFrequencyPostprocessor
POSTPROCESSORS_REGISTRY['multiple_emb'] = MultiEmbeddingsPostprocessor
POSTPROCESSORS_REGISTRY['multi_emb_freq'] = MultiEmbeddingsFreqPostprocessor
PAIRWISE_MODELS_REGISTRY['multi_query_cat'] = MultiConcat
PAIRWISE_MODELS_REGISTRY['multi_query_attn'] = MultiConcatWithAttention

@hydra.main(config_path="configs", config_name="val_cars.yaml")
def main_hydra(cfg: DictConfig) -> None:
    pl_val(cfg)


if __name__ == "__main__":
    main_hydra()
