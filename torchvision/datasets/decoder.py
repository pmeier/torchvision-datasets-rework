import io
from typing import Any, Callable, Dict

import PIL.Image

__all__ = ["_decode_sample", "pil"]


def _decode_sample(sample: Dict[str, Any], decoder: Callable[[io.BufferedIOBase], Any]) -> Dict[str, Any]:
    return {key: decoder(data) if isinstance(data, io.BufferedIOBase) else data for key, data in sample.items()}


def pil(data: io.BufferedIOBase, mode="RGB") -> PIL.Image.Image:
    try:
        with io.BytesIO(data.read()) as stream:
            return PIL.Image.open(stream).convert(mode.upper())
    except Exception:
        print()
