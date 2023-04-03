from abc import ABC
from typing import Any, Dict, List

import torch
from torch import Tensor

from oml.const import EMBEDDINGS_KEY, IS_GALLERY_KEY, IS_QUERY_KEY, LABELS_KEY
from oml.inference.abstract import _inference
from oml.interfaces.retrieval import IDistancesPostprocessor
from oml.interfaces.models import IPairwiseModel
from oml.utils.misc_torch import assign_2d
from oml.utils.misc_torch import get_device
from dataset.multi_query import MultiQueryEmbeddingDataset

def multi_query_inference_on_embeddings(
    model: IPairwiseModel,
    embeddings_query: Tensor,
    embeddings_gallery: Tensor,
    num_workers: int,
    batch_size: int,
    verbose: bool = False,
    use_fp16: bool = False,
    accumulate_on_cpu: bool = True,
) -> Tensor:
    device = get_device(model)

    dataset = MultiQueryEmbeddingDataset(embeddings1=embeddings_query, embeddings2=embeddings_gallery)

    def _apply(
        model_: IPairwiseModel,
        batch_: Dict[str, Any],
    ) -> Tensor:
        pair1 = batch_[dataset.pair_1st_key].to(device)
        pair2 = batch_[dataset.pair_2nd_key].to(device)
        return model_.predict(pair1, pair2)

    output = _inference(
        model=model,
        apply_model=_apply,
        dataset=dataset,
        num_workers=num_workers,
        batch_size=batch_size,
        verbose=verbose,
        use_fp16=use_fp16,
        accumulate_on_cpu=accumulate_on_cpu,
    )

    return output


class MultiEmbeddingsPostprocessor(IDistancesPostprocessor, ABC):
    def __init__(
        self,
        top_n: int,
        pairwise_model: IPairwiseModel,
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
        self.model = pairwise_model
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.verbose = verbose
        self.use_fp16 = use_fp16

        self.is_query_key = is_query_key
        self.is_gallery_key = is_gallery_key
        self.embeddings_key = embeddings_key
        self.label_key = label_key

    def process(self, distances: Tensor, queries: Any, galleries: Any, q_labels: Any) -> Tensor:

        n_queries = len(queries)
        n_galleries = len(galleries)

        assert list(distances.shape) == [n_queries, n_galleries]

        top_n = min(self.top_n, n_galleries)
        ii_top = torch.topk(distances, k=top_n, largest=False)[1]

        if self.verbose:
            print("\nPostprocessor's inference has been started...")
        distances_upd, new_ii_top = self.inference(queries=queries, 
                                                   galleries=galleries, 
                                                   distances=torch.clone(distances), 
                                                   top_n=top_n, q_labels=q_labels)
        new_ii_top = new_ii_top.to(ii_top.device).to(ii_top.dtype)
        distances_upd = distances_upd.to(distances.device).to(distances.dtype)

        if top_n < n_galleries:
            min_in_old_distances = torch.topk(distances, k=top_n + 1, largest=False)[0][:, -1]
            max_in_new_distances = distances_upd.max(dim=1)[0]
            offset = max_in_new_distances - min_in_old_distances + 1e-5
            distances += offset.unsqueeze(-1)

        distances = assign_2d(x=distances, indices=new_ii_top, new_values=distances_upd)

        assert list(distances.shape) == [n_queries, n_galleries]

        return distances
    
    def inference(self, queries: Tensor, galleries: Tensor, distances: Tensor, top_n: int, q_labels: Tensor) -> Tensor:
        n_queries = len(queries)
        
        indexes_ = torch.arange(q_labels.shape[0])
        add_multi = 2
        def multi_query(ind):
            label = q_labels[ind]
            cur_possible_inds = indexes_[q_labels == label]
            cur_possible_inds = cur_possible_inds[cur_possible_inds != ind]
            inds = torch.randperm(len(cur_possible_inds))[:add_multi]
            return torch.cat((torch.tensor([ind]), cur_possible_inds[inds])).view(1, -1)
        
        q_inds = torch.cat([multi_query(i) for i in range(len(q_labels))], dim=0)
        
        distances[torch.arange(0, distances.shape[0]), q_inds.T] = torch.inf
        new_ii_top = torch.topk(distances, k=top_n, largest=False)[1]
        
        multiple_queries = queries[q_inds].view(queries.shape[0], -1)
        
        multiple_queries = multiple_queries.repeat_interleave(top_n, dim=0)
        galleries = galleries[new_ii_top.view(-1)]
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
        return distances_upd, new_ii_top

    def process_by_dict(self, distances: Tensor, data: Dict[str, Any]) -> Tensor:
        queries = data[self.embeddings_key][data[self.is_query_key]]
        galleries = data[self.embeddings_key][data[self.is_gallery_key]]
        q_labels = data[self.label_key][data[self.is_query_key]]
        return self.process(distances=distances, queries=queries, galleries=galleries, q_labels=q_labels)

    @property
    def needed_keys(self) -> List[str]:
        return [self.is_query_key, self.is_gallery_key, self.embeddings_key]


__all__ = ["MultiEmbeddingsPostprocessor"]
