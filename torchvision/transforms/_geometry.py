from typing import Any, Dict, Tuple

import torch

from torchvision.datasets.utils import Query
from torchvision.features import BoundingBox, Image

from ._transform import Transform

__all__ = ["HorizontalFlip", "Rotate", "RandomRotate", "RandomErase"]


class HorizontalFlip(Transform):
    @staticmethod
    def image(input: Image) -> Image:
        return Image(input.flip((-1,)))

    @staticmethod
    def bounding_box(input: BoundingBox) -> BoundingBox:
        x, y, w, h = input.convert("xywh").to_parts()
        x = input.image_size[1] - (x + w)
        return BoundingBox.from_parts(x, y, w, h, like=input, format="xywh")


class Rotate(Transform):
    def __init__(self, degrees: float) -> None:
        super().__init__()
        self.degrees = degrees

    def get_params(self, sample: Any) -> Dict[str, Any]:
        return dict(degrees=self.degrees)

    @staticmethod
    def image(input: Image, *, degrees: torch.Tensor) -> Image:
        return input

    @staticmethod
    def bounding_box(input: BoundingBox, *, degrees: torch.Tensor) -> BoundingBox:
        return input


@Rotate.is_used_by
class RandomRotate(Transform):
    def __init__(self, low: float, high: float) -> None:
        super().__init__()
        self._dist = torch.distributions.Uniform(low, high)

    def get_params(self, sample: Any):
        return dict(degrees=self._dist.sample().item())


class RandomErase(Transform):
    def get_params(self, sample: Any) -> Dict[str, Any]:
        image_size = Query(sample).image_size()
        return dict(erase_size=tuple(size // 2 for size in image_size))

    @staticmethod
    def image(input: Image, *, erase_size: Tuple[int, int]) -> Image:
        return input
