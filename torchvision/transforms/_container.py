from typing import Any

import torch

from ._transform import _TransformBase

__all__ = ["Compose", "RandomApply", "RandomChoice"]


class _ContainerTransform(_TransformBase):
    def supports(self, obj: Any) -> bool:
        raise NotImplementedError()

    def forward(self, *inputs: Any, strict: bool = False) -> Any:
        raise NotImplementedError()


class _WrapperTransform(_TransformBase):
    def __init__(self, transform: _TransformBase):
        super().__init__()
        self._transform = transform

    def supports(self, obj: Any) -> bool:
        return self._transform.supports(obj)


class _MultiTransform(_TransformBase):
    def __init__(self, *transforms: _TransformBase) -> None:
        super().__init__()
        self._transforms = transforms

    def supports(self, obj: Any, *, strict: bool = False) -> bool:
        aggregator = all if strict else any
        return aggregator(transform.supports(obj) for transform in self._transforms)


class Compose(_MultiTransform):
    def forward(self, *inputs: Any, strict: bool = False) -> Any:
        for transform in self._transforms:
            inputs = transform(*inputs, strict=strict)
        return inputs


class RandomApply(_WrapperTransform):
    def __init__(self, transform: _TransformBase, *, p: float = 0.5) -> None:
        super().__init__(transform)
        self._p = p

    def forward(self, *inputs: Any, strict: bool = False) -> Any:
        sample = inputs if len(inputs) > 1 else inputs[0]
        if torch.rand() < self._p:
            # TODO: Should we check here is sample is supported if strict=True?
            return sample

        return self._transform(sample, strict=strict)


class RandomChoice(_MultiTransform):
    def forward(self, *inputs: Any, strict: bool = True) -> Any:
        idx = torch.randint(len(self._transforms), size=()).item()
        transform = self._transforms[idx]
        return transform(*inputs, strict=strict)
