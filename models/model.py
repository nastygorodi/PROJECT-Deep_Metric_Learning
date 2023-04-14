from typing import Any

from torch import Tensor, nn


class IMultiQueryModel(nn.Module):

    def forward(self, x1: Any, x2: Any) -> Tensor:

        raise NotImplementedError()

    def predict(self, x1: Any, x2: Any) -> Tensor:
        
        raise NotImplementedError()
    
__all__ = ["IMultiQueryModel"]
