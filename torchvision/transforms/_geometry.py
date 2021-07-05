from typing import Any, Tuple

import torch

from torchvision.features import BoundingBox, Image

from ._core import Transform

__all__ = ["HorizontalFlip"]


class HorizontalFlip(Transform):
    _DEFAULT_FEATURE_TYPE = Image

    @staticmethod
    def image(image: torch.Tensor) -> Image:
        return Image(image.flip((-1,)))

    @staticmethod
    def bounding_box(bounding_box: BoundingBox, image_size: Tuple[int, int]) -> BoundingBox:
        _, image_width = image_size

        x, y, w, h = bounding_box.convert("xywh").unbind(-1)
        x = (image_width - (x + w)).to(x)
        bounding_box = torch.stack((x, y, w, h), dim=-1)

        return BoundingBox(bounding_box, format="xywh")

    def _apply_feature_transforms(self, sample: Any) -> Any:
        return self._apply_image_and_bounding_box_transforms(
            sample, image_transform=self.image, bounding_box_transform=self.bounding_box
        )
