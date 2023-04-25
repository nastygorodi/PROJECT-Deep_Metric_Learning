from pathlib import Path
from typing import Optional, Union

import torch
from torch import Tensor

from models.model import IMultiQueryModel
from models.multi_query_concat import remain_prefix_in_state_dict

class MLP(torch.nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=torch.nn.ReLU, drop=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = torch.nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = torch.nn.Linear(hidden_features, out_features)
        self.drop = torch.nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x
    
class Attention(torch.nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5

        self.qkv = torch.nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = torch.nn.Dropout(attn_drop)
        self.proj = torch.nn.Linear(dim, dim)
        self.proj_drop = torch.nn.Dropout(proj_drop)

    def forward(self, x):
        B, C = x.shape
        qkv = self.qkv(x).reshape(B, 3, self.num_heads, C // self.num_heads).permute(1, 0, 2, 3)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, attn

class Block(torch.nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.0, qkv_bias=False, qk_scale=None, drop=0.0, 
                 attn_drop=0.0, act_layer=torch.nn.ReLU, norm_layer=torch.nn.LayerNorm,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop
        )
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, return_attention: bool = False):
        y, attn = self.attn(self.norm1(x))
        if return_attention:
            return attn
        x = x + y
        x = x + self.mlp(self.norm2(x))
        return x


class MultiConcatWithAttention(IMultiQueryModel):
    
    def __init__(self, n_query: int, hidden_dim: int, weights: Optional[Union[str, Path]] = None, strict_load: bool = True):
        super(MultiConcatWithAttention, self).__init__()
        
        self.feat_dim = 384
        
        self.attn = Block(dim=self.feat_dim * (n_query + 1), num_heads=8)

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
        res_a = self.attn(res)
        res_out = self.proj(res_a)
        return res_out

    def predict(self, x1: Tensor, x2: Tensor) -> Tensor:
        return self.forward(x1=x1, x2=x2)

__all__ = ["MultiConcatWithAttention"]
