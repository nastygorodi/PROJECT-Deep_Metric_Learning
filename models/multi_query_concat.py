from pathlib import Path
from typing import Optional, Union

import torch
from torch import Tensor

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

        self.proj = torch.nn.Sequential(
            torch.nn.Linear(in_features=self.feat_dim * (n_query + 1), out_features=hidden_dim, bias=False),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=hidden_dim, out_features=hidden_dim // 4, bias=False),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=hidden_dim // 4, out_features=1, bias=False)
            
        )
        
        if weights:

            loaded = torch.load(weights, map_location="cpu")
            loaded = loaded.get("state_dict", loaded)
            loaded = remain_prefix_in_state_dict(loaded, prefix="model_pairwise.")
            
            self.load_state_dict(loaded, strict=strict_load)
        

    def forward(self, x1: Tensor, x2: Tensor) -> Tensor:
        res = torch.cat([x1.view(x1.shape[0], -1), x2], dim=-1)
        res = self.proj(res)
        return res

    def predict(self, x1: Tensor, x2: Tensor) -> Tensor:
        return self.forward(x1=x1, x2=x2)

__all__ = ["MultiConcat"]
