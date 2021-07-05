import enum
from typing import Any, Union

import torch

from ._core import Feature

__all__ = ["BoundingBox", "BoundingBoxFormat"]


class BoundingBoxFormat(enum.Enum):
    XYXY = "XYXY"
    XYWH = "XYWH"
    CXCYWH = "CXCYWH"


class BoundingBox(Feature):
    formats = BoundingBoxFormat

    @staticmethod
    def _parse_format(format: Union[str, BoundingBoxFormat]) -> BoundingBoxFormat:
        if isinstance(format, str):
            format = format.upper()
        return BoundingBox.formats(format)

    def __init__(self, data: Any = None, *, format: Union[str, BoundingBoxFormat], requires_grad=False):
        super().__init__()
        self._format = self._parse_format(format)
        self._to_xyxy = {
            self.formats.XYWH: self._from_xywh,
            self.formats.CXCYWH: self._from_cxcywh,
        }
        self._from_xyxy = {
            self.formats.XYWH: self._to_xywh,
            self.formats.CXCYWH: self._to_cxcywh,
        }

    def __new__(cls, data: Any = None, *, format: Union[str, BoundingBoxFormat], requires_grad=False):
        return super().__new__(cls, data, requires_grad=requires_grad)

    @property
    def format(self) -> BoundingBoxFormat:
        return self._format

    def convert(self, format: Union[str, BoundingBoxFormat]) -> "BoundingBox":
        format = self._parse_format(format)
        bounding_box = self.clone()

        if format == self.format:
            return bounding_box

        if self.format != self.formats.XYXY:
            bounding_box = self._to_xyxy[self.format](bounding_box)

        if format != self.formats.XYXY:
            bounding_box = self._from_xyxy[format](bounding_box)

        return bounding_box

    @staticmethod
    def _from_xywh(bounding_box: "BoundingBox") -> "BoundingBox":
        x, y, w, h = bounding_box.unbind(-1)

        x1 = x
        y1 = y
        x2 = x + w
        y2 = y + h

        return BoundingBox(torch.stack((x1, y1, x2, y2), dim=-1), format="xywh")

    @staticmethod
    def _from_cxcywh(bounding_box: "BoundingBox") -> "BoundingBox":
        cx, cy, w, h = bounding_box.unbind(-1)

        x1 = cx - 0.5 * w
        y1 = cy - 0.5 * h
        x2 = cx + 0.5 * w
        y2 = cy + 0.5 * h

        return BoundingBox(torch.stack((x1, y1, x2, y2), dim=-1), format="cxcywh")

    @staticmethod
    def _to_xywh(bounding_box: "BoundingBox") -> "BoundingBox":
        x1, y1, x2, y2 = bounding_box.unbind(-1)

        x = x1
        y = y1
        w = x2 - x1
        h = y2 - y1

        return BoundingBox(torch.stack((x, y, w, h), dim=-1), format="xywh")

    @staticmethod
    def _to_cxcywh(bounding_box: "BoundingBox") -> "BoundingBox":
        x1, y1, x2, y2 = bounding_box.unbind(-1)

        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        w = x2 - x1
        h = y2 - y1

        return BoundingBox(torch.stack((cx, cy, w, h), dim=-1), format="cxcywh")
