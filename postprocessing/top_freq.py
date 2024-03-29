from abc import ABC
from typing import Any, Dict, List

import torch
from torch import Tensor

from oml.const import EMBEDDINGS_KEY, IS_GALLERY_KEY, IS_QUERY_KEY, LABELS_KEY

from oml.interfaces.retrieval import IDistancesPostprocessor
from oml.utils.misc_torch import assign_2d


class TopFrequencyPostprocessor(IDistancesPostprocessor, ABC):
    def __init__(
        self,
        top_n: int,
        q_inds_path,
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
        with open(q_inds_path, 'rb') as f:
            self.q_inds = torch.load(f)
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.verbose = verbose
        self.use_fp16 = use_fp16

        self.is_query_key = is_query_key
        self.is_gallery_key = is_gallery_key
        self.embeddings_key = embeddings_key
        self.label_key = label_key
    
    def process(self, distances: Tensor, queries: Any, galleries: Any) -> Tensor:
        
        n_queries = len(queries)
        n_galleries = len(galleries)

        assert list(distances.shape) == [n_queries, n_galleries]

        top_n = min(self.top_n, n_galleries)
        ii_top = torch.topk(distances, k=top_n, largest=False)[1]

        if self.verbose:
            print("\nPostprocessor's inference has been started...")
        new_ii_top = self.inference(distances=torch.clone(distances), top_n=top_n)
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

    def inference(self, distances: Tensor, top_n: int) -> Tensor:
        q_inds = self.q_inds
        
        distances[torch.arange(0, distances.shape[0]), q_inds.T] = torch.inf
        ii_top = torch.topk(distances, k=top_n, largest=False)[1]
        
        multi = ii_top[q_inds].view(q_inds.shape[0], -1)
        
        def top_freq(x):
            vals, counts = x.unique(sorted=False, return_counts=True)
            _, freq = counts.sort(descending=True)
            res = vals[freq]
            return res[:top_n].view(1, -1)
        
        multi_ii = torch.cat([top_freq(multi[i]) for i in range(multi.shape[0])], dim=0)
        return multi_ii

    def process_by_dict(self, distances: Tensor, data: Dict[str, Any]) -> Tensor:
        queries = data[self.embeddings_key][data[self.is_query_key]]
        galleries = data[self.embeddings_key][data[self.is_gallery_key]]
        return self.process(distances=distances, queries=queries, galleries=galleries)

    @property
    def needed_keys(self) -> List[str]:
        return [self.is_query_key, self.is_gallery_key, self.embeddings_key]

__all__ = ["TopFrequencyPostprocessor"]
