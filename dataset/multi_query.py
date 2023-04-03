from typing import Dict

from torch import Tensor

from oml.const import INDEX_KEY, PAIR_1ST_KEY, PAIR_2ND_KEY
from oml.interfaces.datasets import IPairsDataset


class MultiQueryEmbeddingDataset(IPairsDataset):
    def __init__(
        self,
        embeddings1: Tensor,
        embeddings2: Tensor,
        pair_1st_key: str = PAIR_1ST_KEY,
        pair_2nd_key: str = PAIR_2ND_KEY,
        index_key: str = INDEX_KEY,
    ):
        assert embeddings1.ndim >= 2

        self.pair_1st_key = pair_1st_key
        self.pair_2nd_key = pair_2nd_key
        self.index_key = index_key

        self.embeddings1 = embeddings1
        self.embeddings2 = embeddings2

    def __getitem__(self, idx: int) -> Dict[str, Tensor]:
        return {self.pair_1st_key: self.embeddings1[idx], self.pair_2nd_key: self.embeddings2[idx], self.index_key: idx}

    def __len__(self) -> int:
        return len(self.embeddings1)
    
__all__ = ["MultiQueryEmbeddingDataset"]
