import torch
import torchvision.transforms as transforms
import torchvision.transforms.v2 as transforms_v2

from PIL import Image
from torchvision.transforms import InterpolationMode

from modules.datasets.helpers.OtsuBinarize import OtsuBinarize

TRANSFORMS_PRE = transforms.Compose(
    [
        transforms.Resize([155, 220], interpolation=InterpolationMode.BILINEAR),
        transforms.RandomInvert(p=1.0),
        transforms.PILToTensor(),
        OtsuBinarize(),
        transforms_v2.ToDtype(torch.float32, scale=False),
    ]
)
# transforms_v2.ToDtype(torch.float32, scale=False),


def TRANSFORMS_TRAIN(stdev):
    def BINARIZE_PIL(image):
        return

    return transforms.Compose(
        [
            transforms.Resize([155, 220], interpolation=InterpolationMode.BILINEAR),
            transforms.RandomInvert(p=1.0),
            transforms.PILToTensor(),
            OtsuBinarize(),
            transforms_v2.ToDtype(torch.float32, scale=False),
            # Divide by stdev but don't subtract by a mean value
            transforms.Normalize(mean=0, std=stdev),
        ]
    )
