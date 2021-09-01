# How do I write a custom `torchvision.transforms.Transform`?

`torchvision.transforms.Transform`'s are set up in a way that hides their dispatch mechanism from you. In 99% of the cases this is a good thing, because it lets you worry only about the actual implementation of the transformation.

The canonical way to implement a custom `Transform` is to subclass from it and implement static methods for the feature transforms:

```python
class MyTransform(transforms.Transform):
    @staticmethod
    def image(input):
        ...
    
    @staticmethod
    def bounding_box(input):
        ...
```

If the names of the static methods in camel-case match the name of a `Feature`, the feature transform is auto-registered. Supported names are

| method name    | `Feature`     | 
|----------------|---------------|
| `image`        | `Image`       |
| `bounding_box` | `BoundingBox` |

If you don't want to stick to this naming scheme, you can disable the auto-registration and perform it manually.

```python
def my_image_transform(input):
    ...
    
class MyTransform(transforms.Transform, auto_register=False):
    def __init__(self):
        super().__init__()
        self.register_feature_transform(Image, my_image_transform)
        self.register_feature_transform(BoundingBox, self.my_bounding_box_transform)
        
    @staticmethod
    def my_bounding_box_transform(input):
        ...

```

In any case, the registration will check the signature of the feature transform so it can be invoked with `feature_transform(input, **params)`. If the feature transforms need additional parameters, you need to implement a `get_params` method. It needs to returns a `params` dictionary that will be unpacked and passed to all feature transforms.

```python
class MyTransform(transforms.Transform):
    def __init__(self, parameter):
        super().__init__()
        self.parameter = parameter
        
    def get_params(self, sample):
        return dict(parameter=self.parameter)
    
    @staticmethod
    def image(input, *, parameter):
        ...
```

The `get_params` method will be invoked once each call, so in case of randomly sampled parameters they will be the same for all features of the whole sample.

```python
class MyTransform(transforms.Transform):
    def __init__(self, low, high):
        super().__init__()
        self._dist = torch.distributions.Uniform(low, high)

    def get_params(self, sample):
        return dict(parameter=self._dist.sample().item())
    
    @staticmethod
    def image(input, *, parameter):
        ...

    @staticmethod
    def bounding_box(input, *, parameter):
        ...
```

Finally, the `get_params` method gets passed the complete `sample` in case the sampling depends on one or more features at runtime. Since `sample` can possibly be a nested container of features, you can use `transforms.query_sample` to extract the information you want without worrying about the structure.


# FAQ

*F*: My `transform` is not doing what I think it should be doing. How do I debug this?
*A*: 

Invoke `transform.is_supported()` with the problematic feature type or an instance of it. If this is `True`, the dispatch is functioning as intended and the problem most likely has its origin in your implementation.

If this comes back as `False`, something went wrong while registering the feature transforms. If you didn't register them manually, it is likely that you didn't stick to naming and signature scheme detailed above. >ou can set the `verbose` flag to get more information about what is happening during the auto-registering process. For example:

```python
class MyTransform(transforms.Transform, verbose=True):
    @staticmethod
    def incompatible_signature1():
        pass

    @staticmethod
    def incompatible_signature2(*, input):
        pass

    # Python >=3.8 only
    # @staticmethod
    # def incompatible_signature3(input, foo, /):
    #     pass

    @staticmethod
    def _private(input):
        pass

    @staticmethod
    def unknown(input):
        pass

    @staticmethod
    def imaeg(input):
        pass

    @staticmethod
    def image(input, foo):
        pass

    @staticmethod
    def boundingbox(input):
        pass

    @staticmethod
    def bounding_box(input, *, foo):
        pass
```

will `print`

```
MyTransform._private() was not registered as feature transform, because it is private.
MyTransform.bounding_box() was registered as feature transform for type 'BoundingBox'.
MyTransform.boundingbox() was not registered as feature transform, because its name doesn't match any known feature type. Did you mean to name it 'bounding_box' to be registered for type 'BoundingBox'?
MyTransform.imaeg() was not registered as feature transform, because its name doesn't match any known feature type. Did you mean to name it 'image' to be registered for type 'Image'?
MyTransform.image() was registered as feature transform for type 'Image'.
MyTransform.incompatible_signature1() was not registered as feature transform, because it cannot be invoked with incompatible_signature1(input, **params).
MyTransform.incompatible_signature2() was not registered as feature transform, because it cannot be invoked with incompatible_signature2(input, **params).
MyTransform.unknown() was not registered as feature transform, because its name doesn't match any known feature type.
```



