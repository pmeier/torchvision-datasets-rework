import collections.abc
from typing import Any, Callable, Optional, Set, Tuple, Type, TypeVar

import torch
from torch import nn

from torchvision.features import BoundingBox, Feature, Image

T = TypeVar("T")


__all__ = ["Transform"]


class Transform(nn.Module):
    _DEFAULT_FEATURE_TYPE: Type[Feature]

    def _apply_feature_transforms(self, sample: Any) -> Any:
        raise NotImplementedError

    @staticmethod
    def _apply_to_sample(sample: Any, fn: Callable[[Any], Any]) -> Any:
        if isinstance(sample, collections.abc.Sequence):
            return [Transform._apply_to_sample(data, fn) for data in sample]
        elif isinstance(sample, collections.abc.Mapping):
            return {name: Transform._apply_to_sample(data, fn) for name, data in sample.items()}
        else:
            return fn(sample)

    @staticmethod
    def _apply_to_feature_type(sample: Any, feature_type: Type[Feature], fn: Callable[[Any], Any]) -> Any:
        return Transform._apply_to_sample(sample, lambda data: fn(data) if isinstance(data, feature_type) else data)

    @staticmethod
    def _query_sample(data: Any, fn: Callable[[Any], Optional[T]]) -> Set[T]:
        if isinstance(data, (collections.abc.Sequence, collections.abc.Mapping)):
            results = set()
            for item in data.values() if isinstance(data, collections.abc.Mapping) else data:
                results.update(Transform._query_sample(item, fn))
            return results
        else:
            result = fn(data)
            return {result} if result is not None else {}

    @staticmethod
    def _query_feature_type(sample: Any, feature_type: Type[Feature], fn: Callable[[Any], Optional[T]]) -> Set[T]:
        return Transform._query_sample(sample, lambda data: fn(data) if isinstance(data, feature_type) else None)

    def forward(self, *inputs):
        sample = self._apply_to_sample(
            inputs if len(inputs) > 1 else inputs[0],
            lambda data: self._DEFAULT_FEATURE_TYPE(data) if type(data) is torch.Tensor else data,
        )
        return self._apply_feature_transforms(sample)

    @staticmethod
    def _apply_image_and_bounding_box_transforms(
        sample: Any,
        *,
        image_transform: Callable[[Image], Image],
        bounding_box_transform: Callable[[BoundingBox, Tuple[int, int]], BoundingBox],
    ) -> Any:
        feature_types = Transform._query_feature_type(sample, Feature, lambda data: type(data))
        if Image in feature_types:
            sample = Transform._apply_to_feature_type(sample, Image, image_transform)
        elif BoundingBox in feature_types:
            raise RuntimeError("Can't transform bounding boxes without image!")
        else:
            return sample

        if BoundingBox not in feature_types:
            return sample

        image_sizes = Transform._query_feature_type(sample, Image, lambda data: data.image_size)
        if len(image_sizes) > 1:
            raise RuntimeError("Found more than one image size!")
        image_size = image_sizes.pop()

        return Transform._apply_to_feature_type(
            sample,
            BoundingBox,
            lambda data: bounding_box_transform(data, image_size),
        )
