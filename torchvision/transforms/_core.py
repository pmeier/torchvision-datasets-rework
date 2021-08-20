import collections.abc
import inspect
import re
from typing import Any, Callable, Dict, Iterator, Optional, Type, TypeVar
from warnings import warn

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


class TransformDispatch():
    def __init__(cls, transforms):
        cls.transforms = transforms
        cls.transforms.set_reset_auto(False)

    def __call__(cls, input_dict):
        input_dict = {key: cls.transform(value) for key, value in input_dict.items()}
        cls.transform.wipeout_()
        return input_dict


class Transform(nn.Module):
    _KNOWN_FEATURE_TYPES = {
        "_".join([part.lower() for part in re.findall("[A-Z][^A-Z]*", feature_type.__name__)]): feature_type
        for feature_type in (Image, BoundingBox)
    }

    def __init_subclass__(cls, *, auto_register: bool = True):
        cls.initialized = False
        cls.reset_auto = True
        cls._feature_transforms: Dict[Type[Feature], Callable] = {}
        if auto_register:
            cls._auto_register()

    def set_reset_auto(cls, mode=True):
        cls.reset_auto = mode
    
    def wipeout_(cls):
        cls.initialized = False
        cls.saved_params = None

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

    @classmethod
    def apply(cls, input: torch.Tensor, **params: Any) -> torch.Tensor:
        feature_type = type(input)
        if not (feature_type is torch.Tensor or feature_type in cls._feature_transforms):
            raise TypeError(f"{cls}() is not able to handle inputs of type {feature_type}.")

        # TODO: if the other domain libraries adopt our approach, we need to make the default type variable.
        #  The only constraint is that it needs to instantiable from a regular tensor without any additional
        #  parameters.
        if feature_type is torch.Tensor:
            feature_type = Image
            input = feature_type(input)

        feature_transform = cls._feature_transforms[feature_type]
        return feature_transform(input, **params)

    def get_params(cls, sample: Any) -> Dict[str, Any]:
        return dict()

    def forward(cls, input: torch.Tensor) -> torch.Tensor:
        if not cls.initialized:
            cls.saved_params = cls.get_params(input)

        feature_type = type(input)
        if not (feature_type is torch.Tensor or feature_type in cls._feature_transforms):
            return input

        output = cls.apply(input, **cls.saved_params)
        if cls.reset_auto:
            cls.wipeout_()
        return output


class Compose(nn.Module):
    def __init__(cls, *transforms: Transform) -> None:
        super().__init__()
        cls._transforms = transforms

    def forward(cls, *inputs: Any) -> Any:
        for transform in cls._transforms:
            inputs = transform(*inputs)
        return inputs

    def set_reset_auto(cls, mode=True):
        for t in cls._transforms:
            if not isinstance(t, (Compose, Transform)):
                warn(f"transform of type {t} cannot be set to reset_auto={mode} as it is not a Transform instance")
            else:
                t.set_reset_auto(mode)
    
    def wipeout_(cls):
        for t in cls._transforms:
            if not isinstance(t, (Compose, Transform)):
                warn(f"transform of type {t} cannot be wiped out as it is not a Transform instance")
            else:
                t.wipeout_()