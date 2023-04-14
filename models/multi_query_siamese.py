from pathlib import Path
from typing import Optional, Union

import torch
from torch import Tensor

from models.model import IMultiQueryModel
from oml.utils.misc_torch import elementwise_dist

def remain_prefix_in_state_dict(state_dict, prefix: str):
    for k in list(state_dict.keys()):
        if k.startswith(prefix):
            state_dict[k[len(prefix) :]] = state_dict[k]
            del state_dict[k]
        else:
            del state_dict[k]

    return state_dict

class MultiDistanceSiamese(IMultiQueryModel):
    
    def __init__(self, extractor, hidden_dim: int, weights: Optional[Union[str, Path]] = None, strict_load: bool = True):
        super(MultiDistanceSiamese, self).__init__()
        
        self.extractor = extractor
        
        self.feat_dim = 384

        self.proj = torch.nn.Sequential(
            torch.nn.Linear(in_features=self.feat_dim, out_features=hidden_dim // 2, bias=False),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=hidden_dim // 2, out_features=hidden_dim // 2, bias=False),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=hidden_dim // 2, out_features=hidden_dim, bias=False)
            
        )
        
        if weights:

            loaded = torch.load(weights, map_location="cpu")
            loaded = loaded.get("state_dict", loaded)
            loaded = remain_prefix_in_state_dict(loaded, prefix="model_pairwise.")
            
            self.load_state_dict(loaded, strict=strict_load)
        

    def forward(self, x1: Tensor, x2: Tensor, train=True) -> Tensor:
        if train:
            x2_shape = x2.shape[-1]
            with torch.no_grad():
                x2_emb = self.extractor(x2)
                x1_emb = [self.extractor(x1[:, i]) for i in range(0, x1.shape[-1] // x2_shape)]
            
            x2 = self.proj(x2_emb)
            x1_new = [self.proj(item) for item in x1_emb]
            
            x1_ = torch.zeros_like(x2)
            for item in x1_new:
                x1_ += item
            x1_ /= (x1.shape[-1] / x2_shape)
        else:
            x2_shape = x2.shape[-1]
            x2 = self.proj(x2)
            x1_new = [self.proj(x1[:, i : i + x2_shape]) for i in range(0, x1.shape[-1], x2_shape)]
            x1_ = torch.zeros_like(x2)
            for item in x1_new:
                x1_ += item
            x1_ /= (x1.shape[-1] / x2_shape)
        y = elementwise_dist(x1_, x2, p=2)
        return y

    def predict(self, x1: Tensor, x2: Tensor, train=True) -> Tensor:
        return self.forward(x1=x1, x2=x2, train=train)

__all__ = ["MultiDistanceSiamese"]
