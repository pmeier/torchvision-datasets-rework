import functools
from typing import Any

from torchvision.features import Image

__all__ = ["Requirement", "ImageRequirement"]


class Requirement:
    pass


class ImageRequirement(Requirement):
    @classmethod
    def batched(cls, noop_if_single: bool = False):
        def outer_wrapper(feature_transform):
            @functools.wraps(feature_transform)
            def inner_wrapper(input: Image, **params: Any) -> Image:
                if noop_if_single and input.batch_size <= 1:
                    return input
                elif input.batch_size >= 1:
                    return feature_transform(input, **params)

                output = feature_transform(input.batch(), **params)
                if output.batch_size <= 1:
                    output = output.unbatch()

                return output

            return inner_wrapper

        return outer_wrapper
