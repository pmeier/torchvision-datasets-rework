import collections.abc
import difflib
import inspect
import re
from typing import Any, Callable, Dict, Optional, Type

import torch
from torch import nn

from torchvision.features import BoundingBox, Feature, Image

__all__ = ["Transform", "Compose"]


class Transform(nn.Module):
    _KNOWN_FEATURE_TYPES = {
        "_".join([part.lower() for part in re.findall("[A-Z][^A-Z]*", feature_type.__name__)]): feature_type
        for feature_type in (Image, BoundingBox)
    }

    def __init_subclass__(cls, *, auto_register: bool = True, verbose: bool = False):
        cls._feature_transforms: Dict[Type[Feature], Callable] = {}
        if auto_register:
            cls._auto_register(verbose=verbose)

    @staticmethod
    def _has_allowed_signature(feature_transform: Callable) -> bool:
        parameters = tuple(inspect.signature(feature_transform).parameters.values())
        if not parameters:
            return False
        elif len(parameters) == 1:
            return parameters[0].kind != inspect.Parameter.KEYWORD_ONLY
        else:
            return parameters[1].kind != inspect.Parameter.POSITIONAL_ONLY

    @classmethod
    def _auto_register(cls, *, verbose: bool = False) -> None:
        for name, value in inspect.getmembers(cls):
            # check if attribute is a static method and was defined in the subclass
            # TODO: this needs to be revisited to allow subclassing of custom transforms
            if not (name in cls.__dict__ and inspect.isfunction(value)):
                continue

            not_registered_prefix = f"{cls.__name__}.{name}() was not registered as feature transform, because"

            if not cls._has_allowed_signature(value):
                if verbose:
                    print(f"{not_registered_prefix} it cannot be invoked with {name}(input, **params).")
                continue

            if name.startswith("_"):
                if verbose:
                    print(f"{not_registered_prefix} it is private.")
                continue

            try:
                feature_type = cls._KNOWN_FEATURE_TYPES[name]
            except KeyError:
                if verbose:
                    msg = f"{not_registered_prefix} its name doesn't match any known feature type."
                    suggestions = difflib.get_close_matches(name, cls._KNOWN_FEATURE_TYPES.keys(), n=1)
                    if suggestions:
                        msg = (
                            f"{msg} Did you mean to name it '{suggestions[0]}' "
                            f"to be registered for type '{cls._KNOWN_FEATURE_TYPES[suggestions[0]].__name__}'?"
                        )
                    print(msg)
                continue

            cls.register_feature_transform(feature_type, value)
            if verbose:
                print(
                    f"{cls.__name__}.{name}() was registered as feature transform for type '{feature_type.__name__}'."
                )

    @classmethod
    def register_feature_transform(cls, feature_type: Type[Feature], transform: Callable) -> None:
        if not cls._has_allowed_signature(transform):
            raise TypeError("Transform cannot be invoked with trasnform(input, **params)")
        cls._feature_transforms[feature_type] = transform

    @classmethod
    def apply(cls, input: torch.Tensor, **params: Any) -> torch.Tensor:
        feature_type = type(input)
        if not (feature_type is torch.Tensor or feature_type in cls._feature_transforms):
            raise TypeError(f"{cls}() is not able to handle inputs of type {feature_type}.")

        # TODO: if the other domain libraries adopt our approach, we need to make the default type variable.
        if feature_type is torch.Tensor:
            feature_type = Image
            input = Image.from_tensor(input)

        feature_transform = cls._feature_transforms[feature_type]
        return feature_transform(input, **params)

    def _apply_recursively(self, sample: Any, *, params: Dict[str, Any]) -> Any:
        if isinstance(sample, collections.abc.Sequence):
            return [self._apply_recursively(item, params=params) for item in sample]
        elif isinstance(sample, collections.abc.Mapping):
            return {name: self._apply_recursively(item, params=params) for name, item in sample.items()}
        else:
            feature_type = type(sample)
            if not (feature_type is torch.Tensor or feature_type in self._feature_transforms):
                return sample

            return self.apply(sample, **params)

    def get_params(self, sample: Any) -> Dict[str, Any]:
        return dict()

    def forward(self, *inputs: Any, params: Optional[Dict[str, Any]] = None) -> Any:
        sample = inputs if len(inputs) > 1 else inputs[0]
        if params is None:
            params = self.get_params(sample)
        return self._apply_recursively(sample, params=params)


class Compose(nn.Module):
    def __init__(self, *transforms: Transform) -> None:
        super().__init__()
        self._transforms = transforms

    def forward(self, *inputs: Any) -> Any:
        for transform in self._transforms:
            inputs = transform(*inputs)
        return inputs
