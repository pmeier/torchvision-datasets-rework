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
    def convert(self, format: str) -> "BoundingBox":
        pass
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
    def __init__(self):
        super().__init__()
        self.register_feature_transform(Image, self.image)
        self.register_feature_transform(BoundingBox, self.bounding_box)

    @staticmethod
    def image(image: Image) -> Image:
        return Image(image.flip((-1,)))

    @staticmethod
    def bounding_box(bounding_box: BoundingBox) -> BoundingBox:
        x, y, w, h = bounding_box.convert("xywh").to_parts()
        x = bounding_box.image_size[1] - (x + w)
        return BoundingBox.from_parts(x, y, w, h, image_size=bounding_box.image_size, format="xywh")
```

1. We have two separate methods to transform the different features. Since they are static, they can completely replace the functional interface we currently provide. For example, instead of using `transforms.functional.horizontal_flip()` we now can use `transforms.HorizontalFlip.image()`. Note that in the former case currently no namespacing for image or bounding box exist, since we only support images.
2. Instead of implementing the feature transform dispatch from scratch for every transform, we can simply register our feature transforms so the dispatch can happen automatically by feature type. We have additional option to automate this even more: if we agree on a identifying scheme, the registering process could happen at instantiation through introspection. One such scheme could be to register all methods that
   1. are public (no leading `_`),
   2. are static,
   3. their name matches a known feature type (converting snake case to camel case, i.e. `bounding_box` to `BoundingBox`), and,
   4. if annotations are available, the first argument as well as the return type is annotated with the corresponding feature type.

3. Since the dispatch for the complete sample happens from a single point in the `forward` method, it is easy to perform the transforms with the same random parameters. For example, for `RandomRotate` the dispatch might look like

   ```python
   class Rotate(Transform):
       def __init__(self, low, high):
           super().__init__()
           self._dist = torch.distributions.Uniform(low, high)
   
       def forward(self, *inputs: torch.Tensor):
           return super().forward(*inputs, degrees=self._dist.sample())
   
       @staticmethod
       def image(image: Image, *, degrees: float) -> Image:
           pass
   
       @staticmethod
       def bounding_box(bounding_box: BoundingBox, *, degrees: float) -> BoundingBox:
           pass
   ```

With this proposal, we achieve all the four requirements above. 

## Implications

- With this proposal, the new `Transform`'s will only work well with the datasets, if we wrap everything in the proposed custom `Feature` classes. 
- Since the `Feature`'s are custom `Tensor`'s, the `Transform`'s will no longer work with `PIL` images. AFAIK, the plan to drop `PIL` support is not new at all. If `torchvision.io` is able to handle all kinds of image I/O, we could also completely remove `PIL` as dependency. If `PIL` is available nevertheless, we can provide `pil_to_tensor` and `tensor_to_pil` functions under `torch.utils` for convenience, but don't rely on them in the normal workflow.
- Each transformation is now responsible for managing the return type. This should be fairly easy for all the builtin transforms, but might be an extra burden for custom transformations. For example, if a custom bounding box transform returns a raw `Tensor` instead of a `BoundingBox`, a successive transform will interpret it as `Image`. 

## Upgrade guide

Upgrading a simple image-only transformation

```python
class MyCustomTransform(nn.Module):
    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return my_custom_transform(image)
```

to the new structure is straight forward:

```python
class MyCustomTransform(Transform):
    def __init__(self) -> None:
        super().__init__()
        self.register_feature_transform(Image, my_custom_transform)
```
