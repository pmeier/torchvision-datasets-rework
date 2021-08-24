import torch

from torchvision import transforms
from torchvision.features import BoundingBox, Image

image = Image(torch.rand(3, 10, 10))
image_e = torch.flip(image, (-1,))

bbox = BoundingBox(torch.tensor([7, 3, 9, 8]), image_size=image.image_size, format="xyxy")
bbox_e = BoundingBox(torch.tensor([1, 3, 3, 8]), image_size=image_e.image_size, format="xyxy")

transform = transforms.HorizontalFlip()

torch.testing.assert_close(transform(image), image_e)
torch.testing.assert_close(transform(bbox).convert("xyxy"), bbox_e)

sample_transform = transforms.apply_to_sample(transform)

sample_a = sample_transform(dict(image=image, bbox=bbox))
torch.testing.assert_close(sample_a["image"], image_e)
torch.testing.assert_close(sample_a["bbox"].convert("xyxy"), bbox_e)

image_a, bbox_a = sample_transform(image, bbox)
torch.testing.assert_close(image_a, image_e)
torch.testing.assert_close(bbox_a.convert("xyxy"), bbox_e)

composed_transform = transforms.apply_to_sample(transforms.Compose(transform, transform))

image_a, bbox_a = composed_transform(image, bbox)
torch.testing.assert_close(image_a, image)
torch.testing.assert_close(bbox_a.convert("xyxy"), bbox)
