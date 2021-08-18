import collections.abc
import inspect
import re
from typing import Any, Callable, Dict, Iterator, Optional, Type, TypeVar

import torch
from torch import nn

from torchvision.features import BoundingBox, Feature, Image

__all__ = ["Transform", "Compose", "query_sample"]

T = TypeVar("T")


def query_sample(sample: Any, fn: Callable[[Any], Optional[T]]) -> Iterator[T]:
    if isinstance(sample, (collections.abc.Sequence, collections.abc.Mapping)):
        for item in sample.values() if isinstance(sample, collections.abc.Mapping) else sample:
            yield from query_sample(item, fn)
    else:
        result = fn(sample)
        if result is not None:
            yield result


class Transform(nn.Module):
    _KNOWN_FEATURE_TYPES = {
        "_".join([part.lower() for part in re.findall("[A-Z][^A-Z]*", feature_type.__name__)]): feature_type
        for feature_type in (Image, BoundingBox)
    }

    def __init__(self, *, auto_register: bool = True) -> None:
        super().__init__()
        self._feature_transforms: Dict[Type[Feature], Callable] = {}
        if auto_register:
            self._auto_register()

    def _auto_register(self) -> None:
        for name, value in inspect.getmembers(self):
            # attribute is private
            if name.startswith("_"):
                continue

            # attribute name does not match any known feature
            try:
                feature_type = self._KNOWN_FEATURE_TYPES[name]
            except KeyError:
                continue

            # attribute is not a static method
            if not inspect.isfunction(value):
                continue
            argspec = inspect.getfullargspec(value)

            # transform takes none or more than one than one positional arguments
            if len(argspec.args) != 1:
                continue

            self.register_feature_transform(feature_type, value)

    def register_feature_transform(self, feature_type: Type[Feature], transform: Callable) -> None:
        self._feature_transforms[feature_type] = transform

    def get_params(self, sample: Any) -> Dict[str, Any]:
        return dict()

    def forward(self, *inputs: Any, params: Optional[Dict[str, Any]] = None) -> Any:
        sample = inputs if len(inputs) > 1 else inputs[0]
        if params is None:
            params = self.get_params(sample)

        def apply(sample: Any):
            if isinstance(sample, collections.abc.Sequence):
                return [apply(item) for item in sample]
            elif isinstance(sample, collections.abc.Mapping):
                return {name: apply(item) for name, item in sample.items()}
            else:
                if not isinstance(sample, torch.Tensor):
                    return sample
                elif type(sample) is torch.Tensor:
                    # TODO: if the other domain libraries adopt our approach, we need to make the default type variable.
                    #  The only constraint is that it needs to instantiable from a regular tensor without any additional
                    #  parameters.
                    sample = Image(sample)

                feature_type = type(sample)
                if feature_type not in self._feature_transforms:
                    return sample

                feature_transform = self._feature_transforms[feature_type]
                return feature_transform(sample, **params)

        return apply(sample)


class Compose(nn.Module):
    def __init__(self, *transforms: Transform) -> None:
        super().__init__()
        self._transforms = transforms

    def forward(self, *inputs: Any) -> Any:
        for transform in self._transforms:
            inputs = transform(*inputs)
        return inputs
