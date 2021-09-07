from typing import Any, Dict, Optional, Type, TypeVar

import torch

__all__ = ["Feature", "TensorFeature"]

TF = TypeVar("TF", bound="TensorFeature")


# A Feature might not necessarily be a Tensor. Think text.
class Feature:
    pass


class TensorFeature(torch.Tensor, Feature):
    def __new__(cls, data: Any = None):
        if data is None:
            data = torch.tensor([])
        requires_grad = False
        return torch.Tensor._make_subclass(cls, data, requires_grad)

    @staticmethod
    def _parse_from_tensor_args(*, like: Optional["TensorFeature"], **attrs: Any) -> Dict[str, Any]:
        if not attrs:
            raise ValueError()

        params = {name: getattr(like, name) for name in attrs.keys()} if like is not None else {}
        params.update({name: value for name, value in attrs.items() if value is not None})
        return params

    @classmethod
    def from_tensor(cls: Type[TF], tensor: torch.Tensor, *, like: Optional[TF] = None) -> TF:
        return cls(tensor)
