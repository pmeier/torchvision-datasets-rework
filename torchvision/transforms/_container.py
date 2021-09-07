from typing import Any, List

import torch

from ._transform import _TransformBase

__all__ = ["Compose", "RandomApply", "RandomChoice"]


class _ContainerTransform(_TransformBase):
    def supports(self, obj: Any) -> bool:
        raise NotImplementedError()

    def forward(self, *inputs: Any, strict: bool = False) -> Any:
        raise NotImplementedError()

    def _make_repr(self, lines: List[str]) -> str:
        extra_repr = self.extra_repr()
        if extra_repr:
            lines = [self.extra_repr(), *lines]
        head = f"{type(self).__name__}("
        tail = ")"
        body = [f"  {line.rstrip()}" for line in lines]
        return "\n".join([head, *body, tail])


class _WrapperTransform(_ContainerTransform):
    def __init__(self, transform: _TransformBase):
        super().__init__()
        self._transform = transform

    def supports(self, obj: Any) -> bool:
        return self._transform.supports(obj)

    def __repr__(self) -> str:
        return self._make_repr(repr(self._transform).splitlines())


class _MultiTransform(_ContainerTransform):
    def __init__(self, *transforms: _TransformBase) -> None:
        super().__init__()
        self._transforms = transforms

    def supports(self, obj: Any, *, strict: bool = False) -> bool:
        aggregator = all if strict else any
        return aggregator(transform.supports(obj) for transform in self._transforms)

    def __repr__(self) -> str:
        lines = []
        for idx, transform in enumerate(self._transforms):
            partial_lines = repr(transform).splitlines()
            lines.append(f"({idx:d}): {partial_lines[0]}")
            lines.extend(partial_lines[1:])
        return self._make_repr(lines)


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
        if float(torch.rand(())) < self._p:
            # TODO: Should we check here is sample is supported if strict=True?
            return sample

        return self._transform(sample, strict=strict)

    def extra_repr(self) -> str:
        return f"p={self._p}"


class RandomChoice(_MultiTransform):
    def forward(self, *inputs: Any, strict: bool = True) -> Any:
        idx = torch.randint(len(self._transforms), size=()).item()
        transform = self._transforms[idx]
        return transform(*inputs, strict=strict)
