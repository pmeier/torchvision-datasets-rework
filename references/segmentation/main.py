from transforms import get_transform

import torch

from torchvision.features import Image, Segmentation

image = Image(torch.rand(3, 480, 640))
seg = Segmentation(torch.randint(0, 256, size=image.shape, dtype=torch.uint8))
sample = image, seg

transform = get_transform(train=True)
train_image, train_seg = transform(sample)

transform = get_transform(train=False)
eval_image, eval_seg = transform(sample)
