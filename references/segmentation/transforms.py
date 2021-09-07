from typing import Sequence

from torchvision import transforms as T


def get_transform(
    *,
    train: bool,
    base_size: int = 520,
    crop_size: int = 480,
    horizontal_flip_probability: float = 0.5,
    mean: Sequence[float] = (0.485, 0.456, 0.406),
    std: Sequence[float] = (0.229, 0.224, 0.225),
):

    if train:
        min_size = base_size // 2
        max_size = base_size * 2
        transforms = [
            T.RandomResize(min_size, max_size),
        ]

        if horizontal_flip_probability > 0:
            transforms.append(T.RandomApply(T.HorizontalFlip(), p=horizontal_flip_probability))

        transforms.append(T.RandomCrop(crop_size))
        augmentation = T.Compose(*transforms)
    else:
        augmentation = T.Resize(base_size)

    return T.Compose(augmentation, T.Normalize(mean, std))
