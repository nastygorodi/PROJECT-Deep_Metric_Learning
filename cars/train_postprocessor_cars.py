import hydra
from omegaconf import DictConfig

from utils.train_postprocessor import pl_train_postprocessor

from oml.registry.postprocessors import POSTPROCESSORS_REGISTRY
from oml.registry.models import PAIRWISE_MODELS_REGISTRY
from postprocessing.multiple_emb import MultiEmbeddingsPostprocessor
from postprocessing.multiple_emb_with_top_freq import MultiEmbeddingsFreqPostprocessor
from models.multi_query_concat import MultiConcat
from models.multi_query_concat_attn import MultiConcatWithAttention
from models.multi_query_mhattn import MultiQueryWithAttention
from models.multi_query_concat_attn_v2 import MultiConcatWithAttention_v2


POSTPROCESSORS_REGISTRY['multiple_emb'] = MultiEmbeddingsPostprocessor
POSTPROCESSORS_REGISTRY['multi_emb_freq'] = MultiEmbeddingsFreqPostprocessor
PAIRWISE_MODELS_REGISTRY['multi_query_cat'] = MultiConcat
PAIRWISE_MODELS_REGISTRY['multi_query_attn'] = MultiConcatWithAttention
PAIRWISE_MODELS_REGISTRY['multi_query_attn_v2'] = MultiQueryWithAttention
PAIRWISE_MODELS_REGISTRY['multi_query_attn_v3'] = MultiConcatWithAttention_v2

@hydra.main(config_path="configs", config_name="train_postprocessor_cars_attn.yaml")
def main_hydra(cfg: DictConfig) -> None:
    pl_train_postprocessor(cfg)


if __name__ == "__main__":
    main_hydra()