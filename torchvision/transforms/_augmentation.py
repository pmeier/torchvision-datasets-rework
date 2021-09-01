from typing import Any, Dict

import torch

from torchvision.features import Image

from . import query
from ._core import Transform

__all__ = ["MixUp"]


class MixUp(Transform):
    def __init__(self, alpha: float = 0.5) -> None:
        super().__init__()
        self.alpha = alpha
        self._dist = torch.distributions.Beta(alpha, alpha)

    def get_params(self, sample: Any) -> Dict[str, Any]:
        batch_size = next(query.batch_size(sample))
        perm = torch.randperm(batch_size)

        lam = self._dist.sample()

        return dict(perm=perm, lam=lam)

    @staticmethod
    def image(input: Image, *, perm: torch.Tensor, lam: torch.Tensor) -> Image:
        shuffled = input[perm]
        mixed = lam * input + (1 - lam) * shuffled
        return Image.from_tensor(mixed, like=input)

    @staticmethod
    def label(input, *, perm: torch.Tensor, lam: torch.Tensor):
        # TODO
        pass
