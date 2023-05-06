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
        self.layer = torch.nn.TransformerEncoderLayer(d_model=self.feat_dim, nhead=self.n_heads, batch_first=True)
        self.attn = torch.nn.TransformerEncoder(self.layer, num_layers=4)
        #self.relu = torch.nn.ReLU()
        self.proj = torch.nn.Linear(in_features=self.feat_dim, out_features=1)
        
        if weights:

            loaded = torch.load(weights, map_location="cpu")
            loaded = loaded.get("state_dict", loaded)
            loaded = remain_prefix_in_state_dict(loaded, prefix="model_pairwise.")
            
            self.load_state_dict(loaded, strict=strict_load)
        

    def forward(self, x1: Tensor, x2: Tensor) -> Tensor:
        x2_shape = x2.shape
        res = torch.cat([x1.view(x1.shape[0], -1), x2], dim=-1).view(x2_shape[0], (self.n_query + 1), -1) # B x n x emb_dim
        res_c = self.attn(res)
        res_out = self.proj(res_c[:, -1])
        return res_out

    def predict(self, x1: Tensor, x2: Tensor) -> Tensor:
        return self.forward(x1=x1, x2=x2)

__all__ = ["MultiQueryWithAttention"]
