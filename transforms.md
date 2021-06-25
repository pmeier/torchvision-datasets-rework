# RFC: Transformations

One major concern of the upstream implementation of datasets in `torchvision.dataset` are the optional `transform`, `target_transform`, and `transforms`. Apart from not being used in a [standardized way  across datasets](https://gist.github.com/pmeier/14756fe0501287b2974e03ab8d651c10), from a theoretical standpoint the transformation has nothing to do with the dataset.

We already decided that the rework will remove transforms from datasets. That also means we need to provide an alternative. Opposed to the upstream implementation, the new API will return a datapipe and each sample drawn from it will be a dictionary that holds the data. This means we cannot simply `dataset.map(torchvision.transforms.Rotate(30.0))` on since the rotation transformation expects an image.

This post outlines how I envision the dataset-transformation-interplay with the reworked dataset API.

## `FeatureTransform`

One pain point for the upstream implementation of transforms is the joint transformation of multiple features with the same transformation. For example images can be rotated with the `Rotate` transform, but corresponding bounding boxes cannot. To overcome this the first proposal of this post is the introduction of a `FeatureTransform` that is able to handle (possibly) multiple `Feature`'s (`Image`, `Video`, `BoundingBox`, ...):

```python
class FeatureTransform:
    def __init__(self):
        self._transforms: Dict[Type[Feature], Callable[[Feature, Any], Any]] = {}

    def _register_transform(
        self, feature_type: Type[Feature], transform: Callable[[Feature, Any], Any]
    ) -> None:
        self._transforms[feature_type] = transform

    @property
    def supported_feature_types(self) -> Set[Feature]:
        return set(self._transforms.keys())

    def __call__(self, feature: Feature, value: Any) -> Any:
        feature_type = type(feature)
        if feature_type not in self.supported_feature_types:
            raise TypeError(f"Can't handle feature type {feature_type}")

        transform = self._transforms[feature_type]
        return transform(feature, value)
```

With this the `Rotate` transforms might look like this

```python
class Rotate(FeatureTransform):
    def __init__(self, angle):
        super().__init__()
        self.angle = angle
        self._register_transform(Image, self._image)
        self._register_transform(BoundingBox, self._bounding_box)

    def _image(self, feature: Image, value: Any) -> Any:
        pass

    def _bounding_box(self, feature: BoundingBox, value: Tuple[int, int, int, int]):
        # the `feature` for example encodes the type of the bounding box, e.g. XYHW
        pass
```

and could be used like this:

```python
transform = Rotate(30.0)
transform(Image(), torch.rand(3, 256, 256))
transform(BoundingBox(type="XYHW"), (0, 0, 256, 256))
```

You might ask: "Why don't we simply add a new `rotate_bbox` function and not bother with the more complicated `Feature` passing?" Three reasons:

1. We would clutter the namespace with concepts that clearly belong together.
2. We would still need special handling for a joint transformation in case `angle` is not a fixed value but rather created at runtime, e.g. `RandomRotate`.

But the real kicker is:

3. The datasets can provide the `Feature` information so you don't have to supply it at all!

Introducing: `SampleTransform`.

## `SampleTransform`

A good UX for applying transforms to a dataset should mean that the user in the default case doesn't need to bother with the feature types at all. That is, it should look similar to what the upstream implementation in simple cases currently provides with the `transform` parameter. To achieve this the second proposal of this post is a `SampleTransform`:

```python
class SampleTransform:
    def __init__(
        self,
        dataset_features: Dict[str, Feature],
        *feature_transforms: FeatureTransform,
        **key_transforms: Callable[[Feature, Any], Any],
    ):
        self.dataset_features = dataset_features
        # populate this by cross-referencing dataset_features and feature_transforms
        # and bail out if a key would be transformed by multiple transformations
        self._key_transforms: Dict[str, Callable[[Feature, Any], Any]] = {}

    def __call__(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        return {
            key: self._key_transforms[key](self.dataset_features[key], value) if key in self._key_transforms else value
            for key, value in sample.items()
        }
```

A typical case might look like this:

```python
from torchvision import datasets, transforms

dataset, info = datasets.load("coco", with_info=True)
transform = transforms.SampleTransform(info.features, transforms.Rotate(30.0))
dataset = dataset.map(transform)
```

Without any knowledge of what features the dataset contains, all features of a sample that support rotation will now be rotated.

### What about handling multiple successive transformations?

`SampleTransform` relies on the fact that each key is only transformed by exactly one transformation. To overcome this we have 2 options:

1. Ask the user to create multiple `SampleTransform`'s and successively `.map()` the dataset with them.
    ```python
    dataset = (
        dataset
        .map(transforms.SampleTransform(info.features, transforms.Rotate(30.0)))
        .map(transforms.SampleTransform(info.features, transforms.HorizontalFlip()))
    )
    ```
2. Create a `Sequential` feature transformation that defines a clear order of execution of the included transformations.
    ```python
    transform = transforms.SampleTransform(
        info.features,
        transforms.Sequential(
            transforms.Rotate(30.0),
            transforms.HorizontalFlip(),
        )info.features
    )
    ```

I'll vote for 2. here, since although it is more code for us to maintain, it is a far better UX for the user.

### What if I want to apply a transformation to a single key rather than all features of a supported type?

Since internally `SampleTransform` dispatches by the sample key rather than `Feature` type, one can simply supply it as keyword argument. For example, if I wanted to rotate only the key `"image"` and not possibly available bounding boxes or segmentation masks, the transformation would look like:

```python
transform = SampleTransform(info.features, image=Rotate(30.0))
```
