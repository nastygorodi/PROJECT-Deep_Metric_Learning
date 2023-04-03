from pathlib import Path
from typing import Optional, Union

import torch
from torch import Tensor

from oml.interfaces.models import IPairwiseModel
from oml.utils.misc_torch import elementwise_dist
from oml.models.utils import remove_prefix_from_state_dict
from oml.utils.io import download_checkpoint


class MultiDistanceSiamese(IPairwiseModel):

    def __init__(self, extractor, hidden_dim: int, weights: Optional[Union[str, Path]] = None, strict_load: bool = True):
        super(MultiDistanceSiamese, self).__init__()
        #self.extractor = extractor
        
        self.feat_dim = 128

        self.proj1 = torch.nn.Linear(in_features=self.feat_dim * 3, out_features=hidden_dim, bias=False)
        self.proj2 = torch.nn.Linear(in_features=self.feat_dim, out_features=hidden_dim, bias=False)
        
        if weights:
            if weights in self.pretrained_models:
                url_or_fid, hash_md5, fname = self.pretrained_models[weights]  # type: ignore
                weights = download_checkpoint(url_or_fid=url_or_fid, hash_md5=hash_md5, fname=fname)

            loaded = torch.load(weights, map_location="cpu")
            loaded = loaded.get("state_dict", loaded)
            loaded = remove_prefix_from_state_dict(loaded, trial_key="extractor.")
            self.load_state_dict(loaded, strict=strict_load)

    def forward(self, x1: Tensor, x2: Tensor) -> Tensor:
        x1 = self.proj1(x1)
        x2 = self.proj2(x2)
        y = elementwise_dist(x1, x2, p=2)
        return y

    def predict(self, x1: Tensor, x2: Tensor) -> Tensor:
        return self.forward(x1=x1, x2=x2)

__all__ = ["MultiDistanceSiamese"]
