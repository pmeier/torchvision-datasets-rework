# RFC: Transformations

One major concern of the upstream implementation of datasets in `torchvision.dataset` are the optional `transform`, `target_transform`, and `transforms`. Apart from not being used in a [standardized way  across datasets](https://gist.github.com/pmeier/14756fe0501287b2974e03ab8d651c10), from a theoretical standpoint the transformation has nothing to do with the dataset.

We already decided that the rework will remove transforms from datasets. This means we need to provide an alternative that offers a similarly easy interface for simple use cases, while still being flexible enough to handle complex use cases. Opposed to the upstream implementation, the new API will return a datapipe and each sample drawn from it will be a dictionary that holds the data. This means we cannot simply use `dataset.map(torchvision.transforms.HorizontalFlip())` without modification.

This post outlines how I envision the dataset-transformation-interplay with the reworked dataset API. You can find a PoC implementation in the same PR.

## `Feature`'s of a dataset sample

PyTorch is flexible enough to allow powerful subclassing of the core class: the `torch.Tensor`. Thus instead of having only the raw `Tensor`, we could add some custom `Feature` classes that represent the individual elements returned by a dataset. These classes could have special fields

```python
class Image(Feature):
    @property
    def image_size(self) -> Tuple[int, int]:
        return self.shape[-2:]
```

or methods

```python
class BoundingBox(Feature):
    @property
    def image_size(self) -> Tuple[int, int]:
        ...
    
    @property
    def format(self) -> str:
        ...
   
    def convert(self, format: str) -> "BoundingBox":
        ...
```

that would make interacting with them a lot easier. Since for all intents and purposes they still act like regular tensors, a user does not have to worry about this at all. Plus, by returning these specific types instead of raw `Tensor`'s for a dataset, a transformation can know what to do with a passed argument.

## `Transform`'ing a dataset sample

Passing `transform=HorizontalFlip()` to the constructor of a dataset is hard to beat in terms of UX. Since we already decided, that this will not be a feature after the rework, the next best thing to apply a transform as a map to each sample, i.e. `dataset = dataset.map(HorizontalFlip())`. Unfortunately, this is not possible with our current transforms, since they cannot deal with a dictionary as input. In particular, all current transformations assume the input is an image.

The new API should have the following features:

1. Each transform should know which features it is able to handle and should do so without needing to explicitly calling anything. For example, `HorizontalFlip` needs to handle `Image`'s as well as `BoundingBox`'es. 
2. The interface should be kept BC, i.e. `HorizontalFlip()(torch.rand(3, 16, 16))` should still work. 
3. The transform should be able to handle the dataset output directly, i.e. a (possibly nested) dictionary of features. This means, by default inputs that are not supported should be ignored by the transform and returned without modification. 
4. Apart from passing a multi-feature sample as a dictionary, it should also be possible to multiple arguments to the transform for convenience, e.g. `image, bounding_box = HorizontalFlip()(image, bounding_box)`.

Points 2. - 4. only concern the dispatch, which can be handled in a custom `Transform` and thus be hidden from users as well someone who writes a new transform. Ignoring that, a transform could look like this:

```python
class HorizontalFlip(Transform):
    @staticmethod
    def image(input: Image) -> Image:
        return Image(input.flip((-1,)))

    @staticmethod
    def bounding_box(input: BoundingBox) -> BoundingBox:
        x, y, w, h = input.convert("xywh").to_parts()
        x = input.image_size[1] - (x + w)
        return BoundingBox.from_parts(x, y, w, h, image_size=input.image_size, format="xywh")
```

This is as simple as it gets (no I didn't leave out anything here; this is actually the full implementation!): we have a separate feature transforms to handle the different elements of the sample. As is, `HorizontalFlip()` will only transform `Image`'s and `BoundingBox`'es. If we later want to add support for `KeyPoints`, we can simply add a `key_points(input)` feature transform and the dispatch mechanism will handle everything else in the background.

The next complexity step are transformations that need additional parameters. Since the feature transforms are static methods, we do not have access to instance attributes. To solve this, every transform can overwrite the `get_params` method. It should return a parameter dictionary that will be unpacked and passed to each transform:

```python
class Rotate(Transform):
    def __init__(self, degrees: float) -> None:
        super().__init__()
        self.degrees = degrees

    def get_params(self, sample: Any = None) -> Dict[str, Any]:
        return dict(degrees=self.degrees)

    @staticmethod
    def image(input: Image, *, degrees: torch.Tensor) -> Image:
        return input

    @staticmethod
    def bounding_box(input: BoundingBox, *, degrees: torch.Tensor) -> BoundingBox:
        return input
```

This is a little more verbose than how our "normal" transformations work, but is very similar to interface of our random transforms, which will be discussed later.

The proposed approach will be sufficient to handle all transformations if the following assumptions hold:

1. The transform is applied to the elements of a single sample.
2. The parameters needed to perform the transform are either static or can be obtained from the sample at runtime.
3. After the parameter sampling, each feature can be transformed independent of the others.

## Functional interface

Since the actual feature transforms are static, they can completely replace the functional API that we currently have. For example, instead of using `transforms.functional.horizontal_flip(...)` we now can use `transforms.HorizontalFlip.image(...)`. Even better, through `Transform.apply` we have access to the same dispatch mechanism the stateful variant uses:

```python
transform = Rotate.apply
transformed_image = transform(image, degrees=30.0)
transformed_bbox = transform(bbox, degrees=-10.0)
```

One design principle that follows from the clean separation of parameter sampling and feature transforms, is that the latter should not contain be deterministic (ignoring non-deterministic behavior of PyTorch operators), i.e. do not contain any random elements. A conclusion from this is that random tranforms should not expose public feature transforms, but rather wrap the ones from the deterministic variant. For example, internally `RandomRotate` should call `Rotate.image()`, but should not expose `RandomRotate.image()` itself to avoid confusion.

## Random transforms

There are in general two types of random transforms:

1. Transforms that sample their parameters at random for each sample, but are always applied.
2. Transforms that are applied at random given a probability, but have fixed parameters.

These cases are not mutually exclusive, but since they address independent concepts they also can be handled independently.

### Random parameters

Since the dispatch for the complete sample happens from a single point in the `forward` method, it is easy to perform all feature transforms with the same random parameters. For example, for `RandomRotate` the implementation might look like

```python
class RandomRotate(Transform):
   def __init__(self, low: float, high: float) -> None:
       super().__init__()
       self._dist = torch.distributions.Uniform(low, high)

   def get_params(self, sample: Any = None) -> Dict[str, Any]:
       return dict(degrees=self._dist.sample().item())

   # The feature transforms are just defined for illustration purposes 
   # and should actually be hidden as explained above
   @staticmethod
   def image(input: Image, *, degrees: torch.Tensor) -> Image:
       return Rotate.image(input, degrees=degrees)

   @staticmethod
   def bounding_box(input: BoundingBox, *, degrees: torch.Tensor) -> BoundingBox:
       return Rotate.image(input, degrees=degrees)
```

Since the `get_params()` methods gets passes the complete sample, the sampled parameters can also depend on the input, which is required for some transforms. For example, [`RandomErasing`](https://github.com/pytorch/vision/blob/f3aff2fa60c110c25df671b6f99ffb26727cb8ae/torchvision/transforms/transforms.py#L1608) needs to know image size to determine location and size of the erased patch.

### Random application

Since the randomness in these transforms is independent of the actual feature transforms, it can be completely separated from the transform. One clean approach is to provide building blocks and let the user combine them as needed. For example, instead of providing `RandomHorizontalFlip`, we would provide `RandomApply` and `HorizontalFlip` such that `RandomHorizontalFlip() == RandomApply(HorizontalFlip())`. This has three advantages:

1. `RandomApply` can be reused for other transforms than `HorizontalFlip`
2. The implementation of a new transform is less error-prone, since a transform like `HorizontalFlip` does not need to deal with the randomness.
3. Some transforms, for example color space transforms, have use cases as deterministic and random variant. With this approach we have access to both without defining two distinct transforms. 


## Upgrade guide

Upgrading a simple image-only transformation

```python
class Foo(nn.Module):
    def __init__(self, bar, baz="baz"):
        super().__init__()
        self.bar = bar
        self.baz = baz

    def forward(self, input):
        return foo(input, self.bar, baz=self.baz)
```

is straight forward.

```python
class Foo(Transform):
    def __init__(self, bar, baz="baz"):
        super().__init__()
        self.bar = bar
        self.baz = baz

    def get_params(self):
        return dict(bar=self.bar, baz=self.baz)

    @staticmethod
    def image(input, *, bar, baz):
        return foo(input, bar, baz=baz)
```

1. The new style is a little more verbose. This is due to the fact that `Transform`'s are now able to handle multiple input types and a single source of parameters is needed for them. For random transforms the verbosity stays the same, since they already have this single source for the random parameters.
2. If `foo` only takes one positional argument, defining a custom `image` method could also be replaced by `self.register_feature_transform(Image, foo)` in `__init__()`.


## Other implications

- With this proposal, the new `Transform`'s will only work well with samples from the datasets, if we wrap everything in the proposed custom `Feature` classes. Since we own the datasets that won't be an issue for the builtin variants. This only becomes an issue if someone wants to use custom datasets while retaining full capability of the proposed transforms. In such a case I think it is reasonable to require them to adhere to the structure and also use the correct `Feature` types in their custom dataset.
- Since the `Feature`'s are custom `Tensor`'s, the `Transform`'s will no longer work with `PIL` images. AFAIK, the plan to drop `PIL` support is not new at all. If `torchvision.io` is able to handle all kinds of image I/O, we could also completely remove `PIL` as dependency. If `PIL` is available nevertheless, we can provide `pil_to_tensor` and `tensor_to_pil` functions under `torch.utils` for convenience, but don't rely on them in the normal workflow.