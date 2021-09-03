from typing import Tuple

from ._core import TensorFeature

__all__ = ["Image"]


class Image(TensorFeature):
    @property
    def image_size(self) -> Tuple[int, int]:
        return self.shape[-2:]

    @property
    def batch_size(self) -> int:
        return self.shape[0] if self.ndim == 4 else 0

    def batch(self) -> "Image":
        if self.batch_size > 0:
            return self

        return Image.from_tensor(self.unsqueeze(0), like=self)

    def unbatch(self) -> "Image":
        if self.batch_size == 0:
            return self
        elif self.batch_size == 1:
            return Image.from_tensor(self.squeeze(0), like=self)
        else:
            raise RuntimeError("Cannot unbatch an image tensor if batch containes more than one image.")
