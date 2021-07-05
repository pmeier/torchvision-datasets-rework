from typing import Any

import torch

__all__ = ["Feature"]


class Feature(torch.Tensor):
    def __new__(cls, data: Any = None, *, requires_grad=False):
        if data is None:
            data = torch.tensor([])
        tensor = torch.Tensor._make_subclass(cls, data, requires_grad)
        return tensor
