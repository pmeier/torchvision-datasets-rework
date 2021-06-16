import io
from typing import Any, Callable

import PIL.Image

import torch as torch_

from torchvision.datasets.utils import Sample

__all__ = ["decode_sample", "pil", "torch", "byte"]


def decode_sample(sample: Sample, decoder: Callable[[io.BufferedIOBase], Any]) -> Sample:
    return Sample({key: decoder(data) if isinstance(data, io.BufferedIOBase) else data for key, data in sample.items()})


def byte(data: io.BufferedIOBase) -> bytes:
    return bytes(data.read())


def pil(data: io.BufferedIOBase, mode="RGB") -> PIL.Image.Image:
    return PIL.Image.open(data).convert(mode.upper())


def torch(data: io.BufferedIOBase) -> torch_.Tensor:
    # TODO: this should use torchvision.io instead doing a detour through PIL
    pil_image = pil(data)
    return torch_.as_tensor(pil_image)
