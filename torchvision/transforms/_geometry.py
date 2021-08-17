import torch

from torchvision.features import BoundingBox, Image

from ._core import Transform

__all__ = ["HorizontalFlip"]


class HorizontalFlip(Transform):
    def __init__(self):
        super().__init__()
        # TODO: Automate this!
        self.register_feature_transform(Image, self.image)
        self.register_feature_transform(BoundingBox, self.bounding_box)

    @staticmethod
    def image(image: Image) -> Image:
        return Image(image.flip((-1,)))

    @staticmethod
    def bounding_box(bounding_box: BoundingBox) -> BoundingBox:
        x, y, w, h = bounding_box.convert("xywh").to_parts()
        x = bounding_box.image_size[1] - (x + w)
        return BoundingBox.from_parts(x, y, w, h, image_size=bounding_box.image_size, format="xywh")


class Rotate(Transform):
    def __init__(self, low, high):
        super().__init__()
        self._dist = torch.distributions.Uniform(low, high)

    def forward(self, *inputs: torch.Tensor):
        return super().forward(*inputs, degrees=self._dist.sample().item())

    @staticmethod
    def image(image: Image, *, degrees: float) -> Image:
        return image

    @staticmethod
    def bounding_box(bounding_box: BoundingBox, *, degrees: float) -> BoundingBox:
        return bounding_box
