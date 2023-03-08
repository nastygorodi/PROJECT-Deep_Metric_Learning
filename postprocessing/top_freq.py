import itertools
from abc import ABC
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
from torch import Tensor

from oml.const import EMBEDDINGS_KEY, IS_GALLERY_KEY, IS_QUERY_KEY, PATHS_KEY, LABELS_KEY
from oml.inference.pairs import (
    pairwise_inference_on_embeddings,
    pairwise_inference_on_images,
)
from oml.interfaces.models import IPairwiseModel
from oml.interfaces.retrieval import IDistancesPostprocessor
from oml.transforms.images.utils import TTransforms
from oml.utils.misc_torch import assign_2d


class MultiQueryPostprocessor(IDistancesPostprocessor, ABC):
    top_n: int
    verbose: bool = False

    def process(self, distances: Tensor, queries: Any, galleries: Any, q_labels: Any) -> Tensor:
        
        n_queries = len(queries)
        n_galleries = len(galleries)

        assert list(distances.shape) == [n_queries, n_galleries]

        top_n = min(self.top_n, n_galleries)
        ii_top = torch.topk(distances, k=top_n, largest=False)[1]

        if self.verbose:
            print("\nPostprocessor's inference has been started...")
        new_ii_top = self.inference(q_labels=q_labels, ii_top=ii_top, top_n=top_n)
        new_ii_top = new_ii_top.to(ii_top.device).to(ii_top.dtype)
        distances_upd = torch.arange(1, top_n + 1).view(1, -1).repeat_interleave(new_ii_top.shape[0], dim=0)
        distances_upd = distances_upd.to(distances.device).to(distances.dtype)

        if top_n < n_galleries:
            min_in_old_distances = torch.topk(distances, k=top_n + 1, largest=False)[0][:, -1]
            max_in_new_distances = distances_upd.max(dim=1)[0]
            offset = max_in_new_distances - min_in_old_distances + 1e-5
            distances += offset.unsqueeze(-1)
        
        distances = assign_2d(x=distances, indices=new_ii_top, new_values=distances_upd)

        assert list(distances.shape) == [n_queries, n_galleries]

        return distances

    def inference(self, q_labels: Any, ii_top: Tensor, top_n: int) -> Tensor:
        raise NotImplementedError()


class TopFrequencyPostprocessor(MultiQueryPostprocessor):
    def __init__(
        self,
        top_n: int,
        num_workers: int,
        batch_size: int,
        verbose: bool = False,
        use_fp16: bool = False,
        label_key: str = LABELS_KEY,
        is_query_key: str = IS_QUERY_KEY,
        is_gallery_key: str = IS_GALLERY_KEY,
        embeddings_key: str = EMBEDDINGS_KEY,
    ):
        assert top_n > 1, "Number of galleries for each query to process has to be greater than 1."

        self.top_n = top_n
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.verbose = verbose
        self.use_fp16 = use_fp16

        self.is_query_key = is_query_key
        self.is_gallery_key = is_gallery_key
        self.embeddings_key = embeddings_key
        self.label_key = label_key

    def inference(self, q_labels: Tensor, ii_top: Tensor, top_n: int) -> Tensor:
        indexes_ = torch.arange(q_labels.shape[0])
        add_multi = 2
        def multi_query(ind):
            label = q_labels[ind]
            cur_possible_inds = indexes_[q_labels == label]
            cur_possible_inds = cur_possible_inds[cur_possible_inds != ind]
            inds = torch.randperm(len(cur_possible_inds))[:add_multi]
            return torch.cat((torch.tensor([ind]), cur_possible_inds[inds])).view(1, -1)
        
        q_inds = torch.cat([multi_query(i) for i in range(len(q_labels))], dim=0)
        
        multi = ii_top[q_inds].view(q_inds.shape[0], -1)
        
        def top_freq(x, q_i):
            vals, counts = x.unique(sorted=False, return_counts=True)
            _, freq = counts.sort(descending=True)
            res = vals[freq]
            return res[:top_n].view(1, -1)
        
        multi_ii = torch.cat([top_freq(multi[i], q_inds[i]) for i in range(multi.shape[0])], dim=0)
        return multi_ii

    def process_by_dict(self, distances: Tensor, data: Dict[str, Any]) -> Tensor:
        queries = data[self.embeddings_key][data[self.is_query_key]]
        galleries = data[self.embeddings_key][data[self.is_gallery_key]]
        q_labels = data[self.label_key][data[self.is_query_key]]
        return self.process(distances=distances, queries=queries, galleries=galleries, q_labels=q_labels)

    @property
    def needed_keys(self) -> List[str]:
        return [self.is_query_key, self.is_gallery_key, self.embeddings_key]

__all__ = ["TopFrequencyPostprocessor"]
