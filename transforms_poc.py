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

dispatch_transform = transforms.TransformDispatch(transform)

sample_a = dispatch_transform(dict(image=image, bbox=bbox))
torch.testing.assert_close(sample_a["image"], image_e)
torch.testing.assert_close(sample_a["bbox"].convert("xyxy"), bbox_e)

image_a, bbox_a = dispatch_transform((image, bbox))
torch.testing.assert_close(image_a, image_e)
torch.testing.assert_close(bbox_a.convert("xyxy"), bbox_e)

composed_transform = transforms.Compose(transform, transform)
dispatch_transform = transforms.TransformDispatch(composed_transform)
sample_a = dispatch_transform(dict(image=image, bbox=bbox))
image_a = composed_transform(image)
torch.testing.assert_close(image_a, image)
