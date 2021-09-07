from typing import Any, Dict, Optional, Tuple, Union

import torch
from torch.nn.functional import interpolate

from torchvision.datasets.utils import Query
from torchvision.features import BoundingBox, Image, Segmentation
from torchvision.transforms import Transform
from torchvision.transforms.utils import ImageRequirement

__all__ = [
    "HorizontalFlip",
    "Resize",
    "RandomResize",
    "Crop",
    "RandomCrop",
    "CenterCrop",
]


class HorizontalFlip(Transform):
    @staticmethod
    def image(input: Image) -> Image:
        return Image(input.flip((-1,)))

    @staticmethod
    def segmentation(input: Segmentation) -> Segmentation:
        return Segmentation(HorizontalFlip.image(input))

    @staticmethod
    def bounding_box(input: BoundingBox) -> BoundingBox:
        x, y, w, h = input.convert("xywh").to_parts()
        x = input.image_size[1] - (x + w)
        return BoundingBox.from_parts(x, y, w, h, like=input, format="xywh")


class Resize(Transform):
    def __init__(self, size: Union[int, Tuple[int, int]], *, interpolation_mode: str = "bilinear") -> None:
        super().__init__()
        self.size = (size, size) if isinstance(size, int) else size
        self.interpolation_mode = interpolation_mode

    def get_params(self, sample: Any) -> Dict[str, Any]:
        # TODO: this is a little weird since we are passing interpolation_mode to segmentation() but ignore it.
        #  Currently we don't support passing different parameters to the feature transforms. We could allow returning a
        #  {Image: dict(..., interpolation_mode="bilinear"), Segmentation: dict(..., interpolation_mode="bilinear")}
        #  In case we get a single parameter dictionary, it will be used for all feature transforms for convenience
        return dict(size=self.size, interpolation_mode=self.interpolation_mode)

    @staticmethod
    @ImageRequirement.batched()
    def image(input: Image, *, size: Tuple[int, int], interpolation_mode: str = "bilinear") -> Image:
        return interpolate(input, size, mode=interpolation_mode)

    @staticmethod
    def segmentation(input: Segmentation, *, size: Tuple[int, int], **_: Any) -> Segmentation:
        return Segmentation(Resize.image(input, size=size, interpolation_mode="nearest"))

    def extra_repr(self) -> str:
        extra_repr = f"size={self.size}"
        if self.interpolation_mode != "bilinear":
            extra_repr += f", interpolation_mode={self.interpolation_mode}"
        return extra_repr


@Resize.is_used_by
class RandomResize(Transform):
    def __init__(self, min_size: Union[int, Tuple[int, int]], max_size: Optional[Union[int, Tuple[int, int]]]) -> None:
        super().__init__()
        self.min_size = (min_size, min_size) if isinstance(min_size, int) else min_size
        self.max_size = (max_size, max_size) if isinstance(max_size, int) else max_size

    def get_params(self, sample: Any) -> Dict[str, Any]:
        min_height, min_width = self.min_size
        max_height, max_width = self.max_size
        height = int(torch.randint(min_height, max_height + 1, size=()))
        width = int(torch.randint(min_width, max_width + 1, size=()))
        return dict(size=(height, width))

    def extra_repr(self) -> str:
        return f"min_size={self.min_size}, max_size={self.max_size}"


class Crop(Transform):
    def __init__(self, crop_box: BoundingBox) -> None:
        super().__init__()
        self.crop_box = crop_box.convert("xyxy")

    def get_params(self, sample: Any) -> Dict[str, Any]:
        return dict(crop_box=self.crop_box)

    @staticmethod
    def image(input: Image, *, crop_box: BoundingBox) -> Image:
        # FIXME: pad input in case it is smaller than crop_box
        x1, y1, x2, y2 = crop_box.convert("xyxy").to_parts()
        return input[..., y1 : y2 + 1, x1 : x2 + 1]

    @staticmethod
    def segmentation(input: Segmentation, *, crop_box) -> Segmentation:
        return Segmentation(Crop.image(input, crop_box=crop_box))


@Crop.is_used_by
class CenterCrop(Transform):
    def __init__(self, crop_size: Union[int, Tuple[int, int]]) -> None:
        super().__init__()
        self.crop_size = (crop_size, crop_size) if isinstance(crop_size, int) else crop_size

    def get_params(self, sample: Any) -> Dict[str, Any]:
        image_size = Query(sample).image_size()
        image_height, image_width = image_size
        cx = image_width // 2
        cy = image_height // 2
        h, w = self.crop_size
        crop_box = BoundingBox.from_parts(cx, cy, w, h, image_size=image_size, format="cxcywh")
        return dict(crop_box=crop_box.convert("xyxy"))

    def extra_repr(self) -> str:
        return f"crop_size={self.crop_size}"


@Crop.is_used_by
class RandomCrop(Transform):
    def __init__(self, crop_size: Union[int, Tuple[int, int]]) -> None:
        super().__init__()
        self.crop_size = (crop_size, crop_size) if isinstance(crop_size, int) else crop_size

    def get_params(self, sample: Any) -> Dict[str, Any]:
        image_size = Query(sample).image_size()
        image_height, image_width = image_size
        crop_height, crop_width = self.crop_size
        x = torch.randint(0, image_width - crop_width + 1, size=()) if crop_width < image_width else 0
        y = torch.randint(0, image_height - crop_height + 1, size=()) if crop_height < image_height else 0
        crop_box = BoundingBox.from_parts(x, y, crop_width, crop_height, image_size=image_size, format="xywh")
        return dict(crop_box=crop_box.convert("xyxy"))

    def extra_repr(self) -> str:
        return f"crop_size={self.crop_size}"
