import collections.abc
import difflib
import inspect
import re
import warnings
from typing import Any, Callable, Dict, Optional, Type, Union, cast

import torch
from torch import nn

from torchvision import features

__all__ = ["Transform", "Identity", "Lambda"]


class _TransformBase(nn.Module):
    _BUILTIN_FEATURE_TYPES = (
        features.BoundingBox,
        features.Image,
        features.Segmentation,
    )


# TODO: Maybe we should name this 'SampleTransform'?
class Transform(_TransformBase):
    """Base class for transforms.

    A transform operates on a full sample at once, which might be a nested container of elements to transform. The
    non-container elements of the sample will be dispatched to feature transforms based on their type in case it is
    supported by the transform. Each transform needs to define at least one feature transform, which is canonical done
    as static method:

    .. code-block::

        class ImageIdentity(Transform):
            @staticmethod
            def image(input):
                return input

    To achieve correct results for a complete sample, each transform should implement feature transforms for every
    :class:`Feature` it can handle:

    .. code-block::

        class Identity(Transform):
            @staticmethod
            def image(input):
                return input

            @staticmethod
            def bounding_box(input):
                return input

            ...

    If the name of a static method in camel-case matches the name of a :class:`Feature`, the feature transform is
    auto-registered. Supported pairs are:

    +----------------+---------------+
    | method name    | `Feature`     |
    +================+===============+
    | `image`        | `Image`       |
    +----------------+---------------+
    | `bounding_box` | `BoundingBox` |
    +----------------+---------------+

    If you don't want to stick to this scheme, you can disable the auto-registration and perform it manually.

    .. code-block::

        def my_image_transform(input):
            ...

        class MyTransform(Transform, auto_register=False):
            def __init__(self):
                super().__init__()
                self.register_feature_transform(Image, my_image_transform)
                self.register_feature_transform(BoundingBox, self.my_bounding_box_transform)

            @staticmethod
            def my_bounding_box_transform(input):
                ...

    In any case, the registration will assert that the feature transform can be invoked with
    ``feature_transform(input, **params)``.

    .. warning::

        Feature transforms are **registered on the class and not on the instance**. This means you cannot have two
        instances of the same :class:`Transform` with different feature transforms.

    If the feature transforms needs additional parameters, you need to
    overwrite the :meth:`~Transform.get_params` method. It needs to return the parameter dictionary that will be
    unpacked and its contents passed to each feature transform:

    .. code-block::

        class Rotate(Transform):
            def __init__(self, degrees):
                super().__init__()
                self.degrees = degrees

            def get_params(self, sample):
                return dict(degrees=self.degrees)

            def image(input, *, degrees):
                ...

    The :meth:`~Transform.get_params` method will be invoked once per sample. Thus, in case of randomly sampled
    parameters they will be the same for all features of the whole sample.

    .. code-block::

        class RandomRotate(Transform)
            def __init__(self, range):
                super().__init__()
                self._dist = torch.distributions.Uniform(rangem range)

            def get_params(self, sample):
                return dict(degrees=self._dist.sample().item())

            @staticmethod
            def image(input, *, degrees):
                ...

    The :meth:`Transform.get_params` method gets passed the complete ``sample`` in case the sampling depends
    on one or more features at runtime.

    To transform a sample, you simply call an instance of the transform with it:

    .. code-block::

        transform = MyTransform()
        sample = dict(input=Image(torch.tensor(...)), target=BoundingBox(torch.tensor(...)), ...)
        transformed_sample = transform(sample)

    By default elements in the ``sample`` that are not supported by the transform are returned without modification.
    You can set the ``strict=True`` flag to force a transformation of every element or bail out in case one is not
    supported.

    .. note::

        To use a :class:`Transform` with a dataset, simply use it as map:

        .. code-block::

            torchvision.datasets.load(...).map(MyTransform())
    """

    _FEATURE_NAME_MAP = {
        "_".join([part.lower() for part in re.findall("[A-Z][^A-Z]*", feature_type.__name__)]): feature_type
        for feature_type in _TransformBase._BUILTIN_FEATURE_TYPES
    }

    def __init_subclass__(cls, *, auto_register: bool = True, verbose: bool = False):
        cls._feature_transforms: Dict[Type[features.Feature], Callable] = {}
        if auto_register:
            cls._auto_register(verbose=verbose)

    @staticmethod
    def _has_allowed_signature(feature_transform: Callable) -> bool:
        """Checks if ``feature_transform`` can be invoked with ``feature_transform(input, **params)``"""

        parameters = tuple(inspect.signature(feature_transform).parameters.values())
        if not parameters:
            return False
        elif len(parameters) == 1:
            return parameters[0].kind != inspect.Parameter.KEYWORD_ONLY
        else:
            return parameters[1].kind != inspect.Parameter.POSITIONAL_ONLY

    @classmethod
    def register_feature_transform(cls, feature_type: Type[features.Feature], transform: Callable) -> None:
        """Registers a transform for given feature on the class.

        If a transform object is called or :meth:`Transform.apply` is invoked, inputs are dispatched to the registered
        transforms based on their type.

        Args:
            feature_type: Feature type the transformation is registered for.
            transform: Feature transformation.

        Raises:
            TypeError: If ``transform`` cannot be invoked with ``transform(input, **params)``.
        """
        if not cls._has_allowed_signature(transform):
            raise TypeError("Transform cannot be invoked with transform(input, **params)")
        cls._feature_transforms[feature_type] = transform

    @classmethod
    def _auto_register(cls, *, verbose: bool = False) -> None:
        """Auto-registers methods on the class as feature transforms if they meet the following criteria:

        1. They are static.
        2. They can be invoked with `cls.feature_transform(input, **params)`.
        3. They are public.
        4. Their name in camel case matches the name of a builtin feature, e.g. 'bounding_box' and 'BoundingBox'.

        The name from 4. determines for which feature the method is registered.

        .. note::

            The ``auto_register`` and ``verbose`` flags need to be passed as keyword arguments to the class:

            .. code-block::

                class MyTransform(Transform, auto_register=True, verbose=True):
                    ...

        Args:
            verbose: If ``True``, prints to STDOUT which methods were registered or why a method was not registered
        """
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
                feature_type = cls._FEATURE_NAME_MAP[name]
            except KeyError:
                if verbose:
                    msg = f"{not_registered_prefix} its name doesn't match any known feature type."
                    suggestions = difflib.get_close_matches(name, cls._FEATURE_NAME_MAP.keys(), n=1)
                    if suggestions:
                        msg = (
                            f"{msg} Did you mean to name it '{suggestions[0]}' "
                            f"to be registered for type '{cls._FEATURE_NAME_MAP[suggestions[0]].__name__}'?"
                        )
                    print(msg)
                continue

            cls.register_feature_transform(feature_type, value)
            if verbose:
                print(
                    f"{cls.__name__}.{name}() was registered as feature transform for type '{feature_type.__name__}'."
                )

    @classmethod
    def from_callable(
        cls,
        feature_transform: Union[Callable, Dict[Type[features.Feature], Callable]],
        *,
        name: str = "FromCallable",
        get_params: Optional[Union[Dict[str, Any], Callable[[Any], Dict[str, Any]]]] = None,
    ) -> "Transform":
        """Creates a new transform from a callable.

        Args:
            feature_transform: Feature transform that will be registered to handle :class:`Image`'s. Can be passed as
                dictionary in which case each key-value-pair is needs to consists of a ``Feature`` type and the
                corresponding transform.
            name: Name of the transform.
            get_params: Parameter dictionary ``params`` that will be passed to ``feature_transform(input, **params)``.
                Can be passed as callable in which case it will be called with the transform instance (``self``) and
                the input of the transform.

        Raises:
            TypeError: If ``feature_transform`` cannot be invoked with ``feature_transform(input, **params)``.
        """
        if get_params is None:
            get_params = dict()
        attributes = dict(
            get_params=get_params if callable(get_params) else lambda self, sample: get_params,
        )
        transform_cls = cast(Type[Transform], type(name, (cls,), attributes))

        if callable(feature_transform):
            feature_transform = {features.Image: feature_transform}
        for feature_type, transform in feature_transform.items():
            transform_cls.register_feature_transform(feature_type, transform)

        return transform_cls()

    @classmethod
    def supports(cls, obj: Any) -> bool:
        """Checks if object or type is supported.

        Args:
            obj: Object or type.
        """
        # TODO: should this handle containers?
        feature_type = obj if isinstance(obj, type) else type(obj)
        return feature_type is torch.Tensor or feature_type in cls._feature_transforms

    @classmethod
    def apply(cls, input: torch.Tensor, **params: Any) -> torch.Tensor:
        """Applies the registered feature transform to the input based on its type.

        This can be uses as type generic functional interface:

            .. code-block::

                transform = Rotate.apply
                transformed_image = transform(Image(torch.tensor(...)), degrees=30.0)
                transformed_bbox = transform(BoundingBox(torch.tensor(...)), degrees=-10.0)

        Args:
            input: ``input`` in ``feature_transform(input, **params)``
            **params: Parameter dictionary ``params`` in ``feature_transform(input, **params)``.

        Returns:
            Transformed input.
        """
        feature_type = type(input)
        if not cls.supports(feature_type):
            raise TypeError(f"{cls.__name__}() is not able to handle inputs of type {feature_type}.")

        # TODO: if the other domain libraries adopt our approach, we need to make the default type variable.
        if feature_type is torch.Tensor:
            feature_type = features.Image
            input = feature_type.from_tensor(input)

        feature_transform = cls._feature_transforms[feature_type]
        return feature_transform(input, **params)

    @classmethod
    def is_used_by(cls, transform_cls: Type["Transform"]):
        """Decorates a :class:`Transform` to reuse all feature transforms.

        This is for example useful if a derivative transform only changes the parameter sampling, but the feature
        transformations are identical:

        .. code-block::

            class Rotate(Transform):
                def get_params(sample):
                    return dict(degrees=30.0)

                def image(input, *, degrees):
                    ...

                def bounding_box(input, *, degrees):
                    ...


            @Rotate.is_used_by
            class RandomRotate(Transform):
                def get_params(sample):
                    return dict(degrees=torch.rand() * 30.0)
        """
        transform_cls._feature_transforms = cls._feature_transforms.copy()
        return transform_cls

    def _apply_recursively(
        self, sample: Any, *, params: Union[Dict[str, Any], Dict[Type[features.Feature], Dict[str, Any]]], strict: bool
    ) -> Any:
        """Recurses through a sample and invokes :meth:`Transform.apply` on non-container elements.

        If an element is not supported by the transform, it is returned untransformed.

        Args:
            sample: Sample.
            params: Parameter dictionary ``params`` that will be passed to ``feature_transform(input, **params)``.
            strict: If ``True``, raises an error in case a non-container element of the ``sample`` is not supported by
            the transform.

        Raises:
            TypeError: If ``strict=True`` and a non-container element of the ``sample`` is not supported.
        """
        if isinstance(sample, collections.abc.Sequence):
            return [self._apply_recursively(item, params=params, strict=strict) for item in sample]
        elif isinstance(sample, collections.abc.Mapping):
            return {name: self._apply_recursively(item, params=params, strict=strict) for name, item in sample.items()}
        else:
            feature_type = type(sample)
            if not self.supports(feature_type):
                if not strict:
                    return sample

                raise TypeError(f"{type(self).__name__}() is not able to handle inputs of type {feature_type}.")

            if all(isinstance(key, type) and issubclass(key, features.Feature) for key in params.keys()):
                params = params[feature_type]
            return self.apply(sample, **params)

    def get_params(self, sample: Any) -> Union[Dict[str, Any], Dict[Type[features.Feature], Dict[str, Any]]]:
        """Returns the parameter dictionary used to transform the current sample.

        .. note::

            Since ``sample`` might be a nested container, it is recommended to use the
            :class:`torchvision.datasets.utils.Query` class if you need to extract information from it.

        Args:
            sample: Current sample.

        Returns:
            Parameter dictionary ``params`` in ``feature_transform(input, **params)``.
        """
        return dict()

    def forward(
        self,
        *inputs: Any,
        params: Optional[Union[Dict[str, Any], Dict[Type[features.Feature], Dict[str, Any]]]] = None,
        strict: bool = True,
    ) -> Any:
        if not self._feature_transforms:
            raise RuntimeError(f"{type(self).__name__}() has no registered feature transform.")

        sample = inputs if len(inputs) > 1 else inputs[0]
        if params is None:
            params = self.get_params(sample)
        return self._apply_recursively(sample, params=params, strict=strict)


class Identity(Transform):
    """Identity transform that supports all built-in :class:`Features`."""

    def __init__(self):
        super().__init__()
        for feature_type in self._BUILTIN_FEATURE_TYPES:
            self.register_feature_transform(feature_type, lambda input, **params: input)


class Lambda(Transform):
    def __new__(cls, lambd: Callable) -> Transform:
        warnings.warn("transforms.Lambda(...) is deprecated. Use transforms.Transform.from_callable(...) instead.")
        # We need to generate a new class everytime a Lambda transform is created, since the feature transforms are
        # registered on the class rather than on the instance. If we didn't, registering a feature transform will
        # overwrite it on **all** Lambda transform instances.
        return Transform.from_callable(lambd, name="Lambda")
