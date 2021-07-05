import torch

from torchvision import transforms
from torchvision.features import BoundingBox, Image

image = Image(torch.rand(3, 10, 10))
image_e = torch.flip(image, (-1,))

bbox = BoundingBox(torch.tensor([7, 3, 9, 8]), format="xyxy")
bbox_e = BoundingBox(torch.tensor([1, 3, 3, 8]), format="xyxy")

transform = transforms.HorizontalFlip()


torch.testing.assert_close(transform(image), image_e)

try:
    transform(bbox)
except RuntimeError:
    pass
else:
    raise AssertionError(
        "Bounding boxes cannot be transformed without knowing the image shape, but no image was provided"
    )

sample_a = transform(dict(image=image, bbox=bbox))
torch.testing.assert_close(sample_a["image"], image_e)
torch.testing.assert_close(sample_a["bbox"].convert("xyxy"), bbox_e)

image_a, bbox_a = transform(image, bbox)
torch.testing.assert_close(image_a, image_e)
torch.testing.assert_close(bbox_a.convert("xyxy"), bbox_e)
