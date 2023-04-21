from pathlib import Path
from typing import Optional, Union

import torch
from torch import Tensor
from oml.utils.misc_torch import elementwise_dist

from models.model import IMultiQueryModel

def remain_prefix_in_state_dict(state_dict, prefix: str):
    for k in list(state_dict.keys()):
        if k.startswith(prefix):
            state_dict[k[len(prefix) :]] = state_dict[k]
            del state_dict[k]
        else:
            del state_dict[k]

    return state_dict


class MultiConcat(IMultiQueryModel):
    
    def __init__(self, n_query: int, hidden_dim: int, weights: Optional[Union[str, Path]] = None, strict_load: bool = True):
        super(MultiConcat, self).__init__()
        
        self.feat_dim = 384
        self.n_query = n_query

        self.proj = torch.nn.Sequential(
            torch.nn.Linear(in_features=self.feat_dim, out_features=hidden_dim // 2, bias=False),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=hidden_dim // 2, out_features=hidden_dim // 4, bias=False),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=hidden_dim // 4, out_features=1, bias=False)
            
        )
        
        if weights:

            loaded = torch.load(weights, map_location="cpu")
            loaded = loaded.get("state_dict", loaded)
            loaded = remain_prefix_in_state_dict(loaded, prefix="model_pairwise.")
            
            self.load_state_dict(loaded, strict=strict_load)
        

    def forward(self, x1: Tensor, x2: Tensor) -> Tensor:
        x1 = x1.view(x1.shape[0], self.n_query, -1)
        out = torch.ones(x1.shape[0]).to(x1.device) * torch.inf
        for i in range(self.n_query):
            cur_res = elementwise_dist(self.proj(x1[:, i]), self.proj(x2), p=2)
            out = torch.where(cur_res < out, cur_res, out)
       
        return out

    def predict(self, x1: Tensor, x2: Tensor) -> Tensor:
        return self.forward(x1=x1, x2=x2)

__all__ = ["MultiConcat"]
