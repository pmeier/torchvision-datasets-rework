from typing import Tuple

from ._core import TensorFeature

__all__ = ["Image"]


class Image(TensorFeature):
    @property
    def image_size(self) -> Tuple[int, int]:
        return self.shape[-2:]
