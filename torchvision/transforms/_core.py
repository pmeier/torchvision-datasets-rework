import collections.abc
from typing import Any, Callable, Type, Dict

import torch
from torch import nn

from torchvision.features import Feature, Image


__all__ = ["Transform"]


class Transform(nn.Module):
    _DEFAULT_FEATURE_TYPE: Type[Feature]

    def __init__(self):
        super().__init__()
        self._feature_transforms: Dict[Type[Feature], Callable] = {}

    def register_feature_transform(self, feature_type: Type[Feature], transform: Callable) -> None:
        self._feature_transforms[feature_type] = transform

    def forward(self, *inputs: Any, **transform_kwargs: Any) -> Any:
        def apply(sample: Any):
            if isinstance(sample, collections.abc.Sequence):
                return [apply(data) for data in sample]
            elif isinstance(sample, collections.abc.Mapping):
                return {name: apply(data) for name, data in sample.items()}
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
                return feature_transform(sample, **transform_kwargs)

        sample = inputs if len(inputs) > 1 else inputs[0]
        return apply(sample)
