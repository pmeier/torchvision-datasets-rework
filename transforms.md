# RFC: Transformations

One major concern of the upstream implementation of datasets in `torchvision.dataset` are the optional `transform`, `target_transform`, and `transforms`. Apart from not being used in a [standardized way  across datasets](https://gist.github.com/pmeier/14756fe0501287b2974e03ab8d651c10), from a theoretical standpoint the transformation has nothing to do with the dataset.

We already decided that the rework will remove transforms from datasets. This means we need to provide an alternative that offers a similarly easy interface for simple use cases, while still being flexible enough to handle complex use cases. Opposed to the upstream implementation, the new API will return a datapipe and each sample drawn from it will be a dictionary that holds the data. This means we cannot simply use `dataset.map(torchvision.transforms.HorizontalFlip)` without modification.

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

that would make interacting with them a lot easier. Since for all intents and purposes they still act like regular tensors, a user does not have to worry about this at all. Plus, by returning these specific types instead of raw `Tensor`'s, a transformation can know what to do with a passed argument.

## `Transform`'ing a dataset sample

Passing `transform=HorizontalFlip()` to the constructor of a dataset is hard to beat in terms of UX. Since we already decided, that this will not be a feature after the rework, the next best thing to apply a transform as a map to each sample, i.e. `dataset = dataset.map(HorizontalFlip())`. Unfortunately, this is not possible with our current transforms, since they cannot deal with a dictionary as input. In particular, all current transformations assume the input is an image.

The new API should have the following features:

1. Each transform should know which features it is able to handle and should do so without needing to explicitly calling anything. For example, `HorizontalFlip` needs to handle `Image`'s as well as `BoundingBox`'es. 
2. The interface should be kept BC, i.e. `HorizontalFlip()(torch.rand(3, 16, 16))` should still work. 
3. The transform should be able to handle the dataset output directly, i.e. a (possibly nested) dictionary of features.
4. Apart from passing a multi-feature sample as a dictionary, it should also be possible to multiple arguments to the transform For example, `image, bounding_box = HorizontalFlip()(image, bounding_box)` should be possible.

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

This is as simple as it gets (no I didn't leave out anything here; this is actually the full implementation!): we have a separate methods to transform the different features. As is, `HorizontalFlip()` will only transform `Image`'s and `BoundingBox`'es. If we later want to add support for `KeyPoints`, we can simply add a `key_points(input)` method and the dispatch mechanism will handle everything else in the background.

With this proposal, we achieve all the four requirements above. 

## Functional interface

Since the actual transformation methods are static, they can completely replace the functional API that we currently have. For example, instead of using `transforms.functional.horizontal_flip(...)` we now can use `transforms.HorizontalFlip.image(...)`. Even better, through `Transform.apply` we have access to the same dispatch mechanism the stateful variant uses:

```python
foo = Foo.apply
transformed_image = foo(image, bar=bar)
transformed_bbox = foo(bbox, bar=bar)
```

Although the stateful transform will in general not be scriptable since it handles arbitrary and possibly nested containers, the functional transform might be since it only handles one specific input. This would bridge the gap between having a clean UX during training, but being able to script the transform for production.

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

## Random transforms

There are in general two types of random transforms:

1. Transforms that sample their parameters at random for each sample, but are always applied.
2. Transforms that are applied at random given a probability, but have fixed parameters.

These cases are not mutually exclusive, but since they address independent concepts they also can be handled independently.

### Random parameters

Since the dispatch for the complete sample happens from a single point in the `forward` method, it is easy to perform the transforms with the same random parameters. For example, for `RandomRotate` the implementation might look like

   ```python
   class RandomRotate(Transform):
       def __init__(self, low: float, high: float) -> None:
           super().__init__()
           self._dist = torch.distributions.Uniform(low, high)
   
       def get_params(self, sample: Any = None) -> Dict[str, Any]:
           return dict(degrees=self._dist.sample().item())
   
       @staticmethod
       def image(image: Image, *, degrees: torch.Tensor) -> Image:
           return image
   
       @staticmethod
       def bounding_box(bounding_box: BoundingBox, *, degrees: torch.Tensor) -> BoundingBox:
           return bounding_box
   ```

   Since we can optionally pass the `sample` to `.get_params()`, the parameters can also depend on the input, which is required for some transforms like [`RandomErasing`](https://github.com/pytorch/vision/blob/f3aff2fa60c110c25df671b6f99ffb26727cb8ae/torchvision/transforms/transforms.py#L1608). Additionally, if required, we can also use the same set of random parameters for multiple distinct transforms:

   ```python
   params = transform1.get_params()
   output1 = transform1(input1, params=params)
   output2 = transform2(input2, params=params)
   ```

### Random application

Since the randomness only decides if the transformation is applied or not, this could be handled well in the dispatch mechanism. By adding a `p=1.0` or `probability=1.0` keyword argument to `Transform`, the dispatch logic can decide if the transformation is applied or not without user interference. This has the advantage of combining a "standard" and random transform that otherwise do almost the same. For example `RandomHorizontalFlip(p=0.5)` would simply become `HorizontalFlip(p=0.5)`.

Let me know if you think this is a worthwhile addition.

## Composed transforms

`transforms.Compose` works out of the box with the proposed approach. Still, we can't use an `nn.Sequential` directly, since the proposed approach requires an arbitrary number of call arguments, e.g. `*inputs`, whereas `nn.Sequential` only uses one.

## Implications

- With this proposal, the new `Transform`'s will only work well with samples from the datasets, if we wrap everything in the proposed custom `Feature` classes. Since we own the datasets that won't be an issue for the builtin variants. This only becomes an issue if someone wants to use custom datasets while retaining full sample capability of the proposed transforms. In such a case I think it is reasonable to require them to adhere to the structure and also set the correct `Feature` types in their custom dataset.
- Since the `Feature`'s are custom `Tensor`'s, the `Transform`'s will no longer work with `PIL` images. AFAIK, the plan to drop `PIL` support is not new at all. If `torchvision.io` is able to handle all kinds of image I/O, we could also completely remove `PIL` as dependency. If `PIL` is available nevertheless, we can provide `pil_to_tensor` and `tensor_to_pil` functions under `torch.utils` for convenience, but don't rely on them in the normal workflow.