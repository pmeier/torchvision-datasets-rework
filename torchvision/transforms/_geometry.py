from typing import Any, Dict, Optional, Tuple

import torch

from torchvision.features import BoundingBox, Image

from ._core import Transform, query_sample

__all__ = ["HorizontalFlip", "RandomRotate", "RandomErase"]


class HorizontalFlip(Transform):
    @staticmethod
    def image(image: Image) -> Image:
        return Image(image.flip((-1,)))

    @staticmethod
    def bounding_box(bounding_box: BoundingBox) -> BoundingBox:
        x, y, w, h = bounding_box.convert("xywh").to_parts()
        x = bounding_box.image_size[1] - (x + w)
        return BoundingBox.from_parts(x, y, w, h, image_size=bounding_box.image_size, format="xywh")


class RandomRotate(Transform):
    def __init__(self, low: float, high: float) -> None:
        super().__init__()
        self._dist = torch.distributions.Uniform(low, high)

    def get_params(self, sample: Any = None) -> Dict[str, Any]:
        return dict(degrees=self._dist.sample().item())

    @staticmethod
    def image(image: Image, *, degrees: float) -> Image:
        return image

    @staticmethod
    def bounding_box(bounding_box: BoundingBox, *, degrees: float) -> BoundingBox:
        return bounding_box


class RandomErase(Transform):
    @staticmethod
    def _find_image_size(sample: Any) -> Optional[Tuple[int, int]]:
        if not isinstance(sample, torch.Tensor):
            return None
        elif type(sample) is torch.Tensor:
            return sample.shape[-2:]
        elif isinstance(sample, (Image, BoundingBox)):
            return sample.image_size
        else:
            return None

    def get_params(self, sample: Any) -> Dict[str, Any]:
        image_size = next(query_sample(sample, self._find_image_size))
        return dict(erase_size=tuple(size // 2 for size in image_size))

    @staticmethod
    def image(image: Image, *, erase_size: Tuple[int, int]) -> Image:
        return image
