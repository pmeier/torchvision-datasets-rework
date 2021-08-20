import enum
from typing import Any, Optional, Tuple, Union

import torch

from ._core import TensorFeature

__all__ = ["BoundingBox", "BoundingBoxFormat"]


class BoundingBoxFormat(enum.Enum):
    XYXY = "XYXY"
    XYWH = "XYWH"
    CXCYWH = "CXCYWH"


class BoundingBox(TensorFeature):
    formats = BoundingBoxFormat

    @staticmethod
    def _parse_format(format: Union[str, BoundingBoxFormat]) -> BoundingBoxFormat:
        if isinstance(format, str):
            format = format.upper()
        return BoundingBox.formats(format)

    def __init__(
        self,
        data: Any = None,
        *,
        image_size: Tuple[int, int],
        format: Union[str, BoundingBoxFormat],
    ):
        super().__init__()
        self._image_size = image_size
        self._format = self._parse_format(format)

        self._convert_to_xyxy = {
            self.formats.XYWH: self._xywh_to_xyxy,
            self.formats.CXCYWH: self._cxcywh_to_xyxy,
        }
        self._convert_from_xyxy = {
            self.formats.XYWH: self._xyxy_to_xywh,
            self.formats.CXCYWH: self._xyxy_to_cxcywh,
        }

    def __new__(
        cls,
        data: Any = None,
        *,
        image_size: Tuple[int, int],
        format: Union[str, BoundingBoxFormat],
    ):
        # Since torch.Tensor defines both __new__ and __init__, we also need to do that since we change the signature
        return super().__new__(cls, data)

    @classmethod
    def from_tensor(
        cls,
        tensor: torch.Tensor,
        *,
        like: Optional["BoundingBox"] = None,
        image_size: Optional[Tuple[int, int]] = None,
        format: Optional[Union[str, BoundingBoxFormat]] = None,
    ) -> "BoundingBox":
        params = cls._parse_from_tensor_args(like=like, image_size=image_size, format=format)

        format = params.get("format") or "xyxy"

        image_size = params.get("image_size")
        if image_size is None:
            # TODO: compute minimum image size needed to hold this bounding box depending on format
            image_size = (0, 0)

        return cls(tensor, image_size=image_size, format=format)

    @property
    def image_size(self) -> Tuple[int, int]:
        return self._image_size

    @property
    def format(self) -> BoundingBoxFormat:
        return self._format

    @classmethod
    def from_parts(
        cls,
        a,
        b,
        c,
        d,
        *,
        format: Union[str, BoundingBoxFormat],
        like: Optional["BoundingBox"] = None,
        image_size: Optional[Tuple[int, int]] = None,
    ) -> "BoundingBox":
        return cls.from_tensor(torch.stack((a, b, c, d), dim=-1), like=like, image_size=image_size, format=format)

    def to_parts(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.unbind(-1)

    def convert(self, format: Union[str, BoundingBoxFormat]) -> "BoundingBox":
        format = self._parse_format(format)
        # FIXME: cloning does not preserve custom attributes such as image_size or format
        # bounding_box = self.clone()
        bounding_box = self

        if format == self.format:
            return bounding_box

        if self.format != self.formats.XYXY:
            bounding_box = self._convert_to_xyxy[self.format](bounding_box)

        if format != self.formats.XYXY:
            bounding_box = self._convert_from_xyxy[format](bounding_box)

        return bounding_box

    @staticmethod
    def _xywh_to_xyxy(input: "BoundingBox") -> "BoundingBox":
        x, y, w, h = input.to_parts()

        x1 = x
        y1 = y
        x2 = x + w
        y2 = y + h

        return BoundingBox.from_parts(x1, y1, x2, y2, like=input, format="xyxy")

    @staticmethod
    def _xyxy_to_xywh(input: "BoundingBox") -> "BoundingBox":
        x1, y1, x2, y2 = input.to_parts()

        x = x1
        y = y1
        w = x2 - x1
        h = y2 - y1

        return BoundingBox.from_parts(x, y, w, h, format="xywh")

    @staticmethod
    def _cxcywh_to_xyxy(input: "BoundingBox") -> "BoundingBox":
        cx, cy, w, h = input.to_parts()

        x1 = cx - 0.5 * w
        y1 = cy - 0.5 * h
        x2 = cx + 0.5 * w
        y2 = cy + 0.5 * h

        return BoundingBox.from_parts(x1, y1, x2, y2, like=input, format="xyxy")

    @staticmethod
    def _xyxy_to_cxcywh(input: "BoundingBox") -> "BoundingBox":
        x1, y1, x2, y2 = input.to_parts()

        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        w = x2 - x1
        h = y2 - y1

        return BoundingBox.from_parts(cx, cy, w, h, like=input, format="cxcywh")
