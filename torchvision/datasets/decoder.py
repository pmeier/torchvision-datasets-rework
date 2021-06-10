import io
from typing import Any, Callable, Dict

import PIL.Image

import torch as torch_

__all__ = ["_decode_sample", "pil", "torch", "byte"]


def _decode_sample(sample: Dict[str, Any], decoder: Callable[[io.BufferedIOBase], Any]) -> Dict[str, Any]:
    return {key: decoder(data) if isinstance(data, io.BufferedIOBase) else data for key, data in sample.items()}


def byte(data: io.BufferedIOBase) -> bytes:
    return bytes(data.read())


def pil(data: io.BufferedIOBase, mode="RGB") -> PIL.Image.Image:
    return PIL.Image.open(data).convert(mode.upper())


def torch(data: io.BufferedIOBase) -> torch_.Tensor:
    # TODO: this should use torchvision.io instead doing a detour through PIL
    pil_image = pil(data)
    return torch_.as_tensor(pil_image)
