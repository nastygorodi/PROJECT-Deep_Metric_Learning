from abc import ABC
from typing import Any, Dict, List

import torch
from torch import Tensor

from oml.const import EMBEDDINGS_KEY, IS_GALLERY_KEY, IS_QUERY_KEY, LABELS_KEY
from models.model import IMultiQueryModel
from oml.utils.misc_torch import assign_2d
from multiple_emb import MultiEmbeddingsPostprocessor, multi_query_inference_on_embeddings


class MultiEmbeddingsFreqPostprocessor(MultiEmbeddingsPostprocessor):
    def __init__(
        self,
        top_n: int,
        n_queries: int,
        pairwise_model: IMultiQueryModel,
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
        self.n_queries = n_queries
        self.model = pairwise_model
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.verbose = verbose
        self.use_fp16 = use_fp16

        self.is_query_key = is_query_key
        self.is_gallery_key = is_gallery_key
        self.embeddings_key = embeddings_key
        self.label_key = label_key
    
    def inference(self, queries: Tensor, galleries: Tensor, distances: Tensor, top_n: int, q_labels: Tensor) -> Tensor:
        n_queries = len(queries)
        
        indexes_ = torch.arange(q_labels.shape[0])
        add_multi = self.n_queries - 1
        def multi_query(ind):
            label = q_labels[ind]
            cur_possible_inds = indexes_[q_labels == label]
            cur_possible_inds = cur_possible_inds[cur_possible_inds != ind]
            inds = torch.randperm(len(cur_possible_inds))[:add_multi]
            return torch.cat((torch.tensor([ind]), cur_possible_inds[inds])).view(1, -1)
        
        q_inds = torch.cat([multi_query(i) for i in range(len(q_labels))], dim=0)
        
        distances[torch.arange(0, distances.shape[0]), q_inds.T] = torch.inf
        new_ii_top = torch.topk(distances, k=top_n, largest=False)[1]
        multi_ii_top = new_ii_top[q_inds].view(q_inds.shape[0], -1)
        
        def top_freq(x):
            vals, counts = x.unique(sorted=False, return_counts=True)
            _, freq = counts.sort(descending=True)
            res = vals[freq]
            return res[:top_n].view(1, -1)
        
        multi_ii = torch.cat([top_freq(multi_ii_top[i]) for i in range(multi_ii_top.shape[0])], dim=0)
        
        multiple_queries = queries[q_inds].view(queries.shape[0], -1)
        
        multiple_queries = multiple_queries.repeat_interleave(top_n, dim=0)
        galleries = galleries[multi_ii.view(-1)]
        distances_upd = multi_query_inference_on_embeddings(
            model=self.model,
            embeddings_query=multiple_queries,
            embeddings_gallery=galleries,
            num_workers=self.num_workers,
            batch_size=self.batch_size,
            verbose=self.verbose,
            use_fp16=self.use_fp16,
        )
        distances_upd = distances_upd.view(n_queries, top_n)
        return distances_upd, multi_ii

    @property
    def needed_keys(self) -> List[str]:
        return [self.is_query_key, self.is_gallery_key, self.embeddings_key]


__all__ = ["MultiEmbeddingsFreqPostprocessor"]
