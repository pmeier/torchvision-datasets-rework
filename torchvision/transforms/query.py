import collections.abc
from functools import partial
from typing import Any, Callable, Iterator, Optional, Tuple, TypeVar

import torch

from torchvision.features import BoundingBox, Image

__all__ = ["query_sample", "image_size", "batch_size"]

T = TypeVar("T")


def query_sample(sample: Any, query_fn: Callable[[Any], Optional[T]]) -> Iterator[T]:
    if isinstance(sample, (collections.abc.Sequence, collections.abc.Mapping)):
        for item in sample.values() if isinstance(sample, collections.abc.Mapping) else sample:
            yield from query_sample(item, query_fn)
    else:
        result = query_fn(sample)
        if result is not None:
            yield result


def _query_image_size(sample: Any) -> Optional[Tuple[int, int]]:
    if not isinstance(sample, torch.Tensor):
        return None
    elif type(sample) is torch.Tensor:
        return sample.shape[-2:]
    elif isinstance(sample, (Image, BoundingBox)):
        return sample.image_size
    else:
        return None


image_size = partial(query_sample, query_fn=_query_image_size)


def _query_batch_size(sample: Any) -> Optional[int]:
    if isinstance(sample, torch.Tensor):
        return sample.shape[0]
    else:
        return None


batch_size = partial(query_sample, query_fn=_query_batch_size)
