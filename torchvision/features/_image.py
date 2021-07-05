from typing import Tuple

from ._core import Feature

__all__ = ["Image"]


class Image(Feature):
    @property
    def image_size(self) -> Tuple[int, int]:
        return self.shape[-2:]
