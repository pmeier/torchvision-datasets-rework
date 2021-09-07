import collections.abc
from typing import Any, Callable, Iterator, Optional, Set, Tuple, TypeVar, Union

import torch

from torchvision.features import BoundingBox, Image

T = TypeVar("T")

__all__ = ["Query"]


class Query:
    def __init__(self, sample: Any) -> None:
        self.sample = sample

    @staticmethod
    def _query_recursively(sample: Any, fn: Callable[[Any], Optional[T]]) -> Iterator[T]:
        if isinstance(sample, (collections.abc.Sequence, collections.abc.Mapping)):
            for item in sample.values() if isinstance(sample, collections.abc.Mapping) else sample:
                yield from Query._query_recursively(item, fn)
        else:
            result = fn(sample)
            if result is not None:
                yield result

    def query(self, fn: Callable[[Any], Optional[T]], *, unique: bool = True) -> Union[T, Set[T]]:
        results = set(self._query_recursively(self.sample, fn))
        if not results:
            raise RuntimeError("Query turned up empty.")

        if not unique:
            return results

        if len(results) > 1:
            raise RuntimeError(f"Found more than one result: {sorted(results)}")

        return results.pop()

    def image_size(self) -> Optional[Tuple[int, int]]:
        def fn(sample: Any) -> Optional[Tuple[int, int]]:
            if not isinstance(sample, torch.Tensor):
                return None
            elif type(sample) is torch.Tensor:
                return sample.shape[-2:]
            elif isinstance(sample, (Image, BoundingBox)):
                return sample.image_size
            else:
                return None

        return self.query(fn)

    def batch_size(self) -> Optional[int]:
        def fn(sample: Any) -> Optional[int]:
            if not isinstance(sample, torch.Tensor):
                return None
            elif isinstance(sample, Image):
                return sample.batch_size
            else:
                return None

        return self.query(fn)
