from itertools import combinations, product
from operator import itemgetter
from random import sample
from sys import maxsize
from typing import List

import numpy as np
import torch

from miners.miner import IMultiQueryMiner
from oml.utils.misc import find_value_ids


class MultiQueryMiner_naive(IMultiQueryMiner):
    """
    This miner selects all the possible triplets for the given batch.

    """

    def __init__(self, n_queries: int, max_out: int = 512, device: str = "cpu"):
        
        self.n_queries = n_queries
        self._max_out = max_out
        self._device = device

    def sample(self, labels: List[int]):  # type: ignore
        
        return get_available_items(self.n_queries, labels, max_out=self._max_out)


def get_available_items(n_queries: int, labels: List[int], max_out: int = maxsize):
    
    num_labels = len(labels)

    items = []
    for label in set(labels):
        ids_pos_cur = find_value_ids(labels, label)
        ids_neg_cur = set(range(num_labels)) - set(ids_pos_cur)

        pos_n = list(combinations(ids_pos_cur, r=n_queries))
        pos_p = list(combinations(ids_pos_cur, r=(n_queries + 1)))
        
        item = [(*p, n, 1) for p, n in product(pos_n, ids_neg_cur)]

        item += [(*p, 0) for p in pos_p]

        items.extend(item)

    items = sample(items, min(len(items), max_out))
    queries, gallery, labels = zip(*[(list(x[:n_queries]), x[-2], x[-1]) for x in items])

    return torch.tensor(list(queries)), torch.tensor(list(gallery)), torch.tensor(list(labels))

__all__ = ["MultiQueryMiner", "get_available_items"]