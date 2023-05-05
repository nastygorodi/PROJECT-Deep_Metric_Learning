from pathlib import Path
from typing import Optional, Union

import torch
from torch import Tensor

from models.model import IMultiQueryModel
from models.multi_query_concat import remain_prefix_in_state_dict


class MultiQueryWithAttention(IMultiQueryModel):
    
    def __init__(self, n_query: int, n_heads: int = 4, weights: Optional[Union[str, Path]] = None, strict_load: bool = True):
        super(MultiQueryWithAttention, self).__init__()
        
        self.feat_dim = 384
        self.n_query = n_query
        self.n_heads = n_heads
        self.attn = torch.nn.MultiheadAttention(embed_dim=self.feat_dim * (self.n_query), num_heads=self.n_heads, batch_first=True)

        self.proj = torch.nn.Linear(in_features=self.feat_dim * (self.n_query), out_features=1, bias=False)
        
        if weights:

            loaded = torch.load(weights, map_location="cpu")
            loaded = loaded.get("state_dict", loaded)
            loaded = remain_prefix_in_state_dict(loaded, prefix="model_pairwise.")
            
            self.load_state_dict(loaded, strict=strict_load)
        

    def forward(self, x1: Tensor, x2: Tensor) -> Tensor:
        x2_ = x2.repeat_interleave(self.n_query, dim=0).view(x2.shape[0], -1)
        x1_ = x1.view(x1.shape[0], -1)
        res_c, _ = self.attn(x2_, x2_, x1_)
        res_out = self.proj(res_c)
        return res_out

    def predict(self, x1: Tensor, x2: Tensor) -> Tensor:
        return self.forward(x1=x1, x2=x2)

__all__ = ["MultiQueryWithAttention"]
