from pathlib import Path
from typing import Optional, Union

import torch
from torch import Tensor

from models.model import IMultiQueryModel
from models.multi_query_concat import remain_prefix_in_state_dict

    
class Attention(torch.nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5

        self.qkv = torch.nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = torch.nn.Dropout(attn_drop)

    def forward(self, x):
        B, emb_size, n  = x.shape
        qkv = self.qkv(x.transpose(-1, -2)).reshape(B, -1, n, self.num_heads, emb_size // self.num_heads).permute(1, 0, 2, 3, 4) # 3 x B x n x heads x emb
        q, k, v = qkv[0], qkv[1], qkv[2] # B x n x heads x dim
        
        q = q.permute(2, 0, 1, 3).contiguous().view(B * self.num_heads, n, -1)
        k = k.permute(2, 0, 1, 3).contiguous().view(B * self.num_heads, n, -1)
        v = v.permute(2, 0, 1, 3).contiguous().view(B * self.num_heads, n, -1)

        attn = (q @ k.transpose(-2, -1)) * self.scale # Bheads x n x n
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v)
        x = x.view(self.num_heads, B, n, -1)
        x = x.permute(1, 2, 0, 3).contiguous().view(B, n, emb_size).transpose(-1, -2)
        return x, attn


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
        res = torch.cat([x1.view(x1.shape[0], -1), x2], dim=-1).view(x2_shape[0], (self.n_query + 1), -1)
        res = res.transpose(-1, -2) # B x emb_dim x n
        res_c, _ = self.attn(res)
        res_out = self.proj(res_c[:, :, -1])
        return res_out

    def predict(self, x1: Tensor, x2: Tensor) -> Tensor:
        return self.forward(x1=x1, x2=x2)

__all__ = ["MultiConcatWithAttention"]
