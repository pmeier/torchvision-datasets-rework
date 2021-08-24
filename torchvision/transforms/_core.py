import collections.abc
import inspect
import re
from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence, Type, TypeVar, Union

import torch
from torch import nn

from torchvision.features import BoundingBox, Feature, Image

__all__ = ["Transform", "Compose", "apply_to_sample"]

T = TypeVar("T")


class Transform(nn.Module):
    _KNOWN_FEATURE_TYPES = {
        "_".join([part.lower() for part in re.findall("[A-Z][^A-Z]*", feature_type.__name__)]): feature_type
        for feature_type in (Image, BoundingBox)
    }

    def __init_subclass__(cls, *, auto_register: bool = True):
        cls._feature_transforms: Dict[Type[Feature], Callable] = {}
        if auto_register:
            cls._auto_register()

    @classmethod
    def _auto_register(cls) -> None:
        for name, value in inspect.getmembers(cls):
            # attribute is private
            if name.startswith("_"):
                continue

            # attribute name does not match any known feature
            try:
                feature_type = cls._KNOWN_FEATURE_TYPES[name]
            except KeyError:
                continue

            # attribute is not a static method
            if not inspect.isfunction(value):
                continue
            argspec = inspect.getfullargspec(value)

            # transform takes none or more than one than one positional arguments
            if len(argspec.args) != 1:
                continue

            cls.register_feature_transform(feature_type, value)

    @classmethod
    def register_feature_transform(cls, feature_type: Type[Feature], transform: Callable) -> None:
        cls._feature_transforms[feature_type] = transform

    def get_params(self, input: torch.Tensor) -> Dict[str, Any]:
        return dict()

    def forward(self, input: torch.Tensor, *, params: Optional[Dict[str, Any]] = None) -> torch.Tensor:
        feature_type = type(input)
        if not (feature_type is torch.Tensor or feature_type in self._feature_transforms):
            raise TypeError(f"{type(self).__name__} is not able to handle inputs of type {feature_type}.")

        # TODO: if the other domain libraries adopt our approach, we need to make the default type variable.
        if feature_type is torch.Tensor:
            feature_type = Image
            input = Image.from_tensor(input)

        feature_transform = self._feature_transforms[feature_type]
        return feature_transform(input, **params or self.get_params(input))


class Compose(nn.Module):
    def __init__(self, *transforms: Transform) -> None:
        super().__init__()
        self.transforms = transforms

    def get_params(self, input: torch.Tensor) -> List[Dict[str, Any]]:
        return [transform.get_params(input) for transform in self.transforms]

    def forward(
        self, input: torch.Tensor, params: Optional[Union[Sequence[Dict[str, Any]], Dict[str, Any]]] = None
    ) -> Any:
        if not isinstance(params, collections.abc.Sequence):
            params = [params] * len(self.transforms)
        else:
            if len(params) != len(self.transforms):
                raise ValueError()

        for transform, params_ in zip(self.transforms, params):
            input = transform(input, params=params_)
        return input


def apply_to_sample(transform: Union[Transform, Compose]) -> Callable:
    def apply_recursively(sample: Any, *, params: Dict[str, Any]) -> Any:
        if isinstance(sample, collections.abc.Sequence):
            return [apply_recursively(item, params=params) for item in sample]
        elif isinstance(sample, collections.abc.Mapping):
            return {name: apply_recursively(item, params=params) for name, item in sample.items()}
        else:
            return transform(sample, params=params)

    def sample_transform(*inputs: Any, params: Optional[Dict[str, Any]] = None) -> Any:
        sample = inputs if len(inputs) > 1 else inputs[0]
        return apply_recursively(sample, params=params)

    return sample_transform
