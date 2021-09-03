from typing import Any, Dict

import torch

from torchvision.datasets.utils import Query
from torchvision.features import Image
from torchvision.transforms.utils import ImageRequirement

from ._transform import Transform

__all__ = ["MixUp"]


class MixUp(Transform):
    def __init__(self, alpha: float = 0.5) -> None:
        super().__init__()
        self.alpha = alpha
        self._dist = torch.distributions.Beta(alpha, alpha)

    def get_params(self, sample: Any) -> Dict[str, Any]:
        perm = torch.randperm(Query(sample).batch_size() or 1)
        lam = self._dist.sample()
        return dict(perm=perm, lam=lam)

    @staticmethod
    def _is_ordered(perm) -> bool:
        return bool(perm.eq(perm.sort().values).all())

    @staticmethod
    @ImageRequirement.batched(noop_if_single=True)
    def image(input: Image, *, perm: torch.Tensor, lam: torch.Tensor) -> Image:
        if MixUp._is_ordered(perm):
            return input

        shuffled = input[perm]
        mixed = lam * input + (1 - lam) * shuffled
        return Image.from_tensor(mixed, like=input)

    @staticmethod
    def label(input, *, perm: torch.Tensor, lam: torch.Tensor):
        # TODO
        pass
