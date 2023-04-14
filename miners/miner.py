from abc import ABC, abstractmethod
from typing import List, Union

from torch import Tensor


TLabels = Union[List[int], Tensor]

class IMultiQueryMiner(ABC):
    """
    An abstraction of multi query miner.

    """

    @abstractmethod
    def sample(self, features: Tensor, labels: TLabels):
        
        raise NotImplementedError()
