from typing import Any, Dict, Tuple

import torch

from torchvision.features import BoundingBox, Image

from . import query
from ._core import Transform

__all__ = ["HorizontalFlip", "RandomRotate", "RandomErase"]


class HorizontalFlip(Transform):
    @staticmethod
    def image(input: Image) -> Image:
        return Image(input.flip((-1,)))

    @staticmethod
    def bounding_box(input: BoundingBox) -> BoundingBox:
        x, y, w, h = input.convert("xywh").to_parts()
        x = input.image_size[1] - (x + w)
        return BoundingBox.from_parts(x, y, w, h, like=input, format="xywh")


class RandomRotate(Transform):
    def __init__(self, low: float, high: float) -> None:
        super().__init__()
        self._dist = torch.distributions.Uniform(low, high)

    def get_params(self, sample: Any = None) -> Dict[str, Any]:
        return dict(degrees=self._dist.sample().item())

    @staticmethod
    def image(input: Image, *, degrees: torch.Tensor) -> Image:
        return input

    @staticmethod
    def bounding_box(input: BoundingBox, *, degrees: torch.Tensor) -> BoundingBox:
        return input


class RandomErase(Transform):
    def get_params(self, sample: Any) -> Dict[str, Any]:
        image_size = next(query.image_size(sample))
        return dict(erase_size=tuple(size // 2 for size in image_size))

    @staticmethod
    def image(input: Image, *, erase_size: Tuple[int, int]) -> Image:
        return input
