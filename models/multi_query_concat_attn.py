from pathlib import Path
from typing import Optional, Union

import torch
from torch import Tensor

from models.model import IMultiQueryModel
from models.multi_query_concat import remain_prefix_in_state_dict

    
class Attention(torch.nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5

        self.q_proj = torch.nn.Linear(dim, dim // 2, bias=qkv_bias)
        self.k_proj = torch.nn.Linear(dim, dim // 2, bias=qkv_bias)
        self.v_proj = torch.nn.Linear(dim, dim // 2, bias=qkv_bias)
        self.mhattn = torch.nn.MultiheadAttention(dim, num_heads, batch_first=True)

    def forward(self, x):
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        
        attn, _ = self.mhattn(q, k, v)
        return attn


class MultiConcatWithAttention(IMultiQueryModel):
    
    def __init__(self, n_query: int, n_heads: int = 4, weights: Optional[Union[str, Path]] = None, strict_load: bool = True):
        super(MultiConcatWithAttention, self).__init__()
        
        self.feat_dim = 384
        self.n_query = n_query
        self.n_heads = n_heads
        self.attn = Attention(dim=self.feat_dim, num_heads=self.n_heads)

        self.proj = torch.nn.Linear(in_features=self.feat_dim, out_features=1, bias=False)
        
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

__all__ = ["MultiConcatWithAttention"]
