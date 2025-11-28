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
        transforms.ToTensor(),
        # OtsuBinarize(),
    ]
)
# transforms_v2.ToDtype(torch.float32, scale=False),


"""
transforms.RandomAffine(
    degrees=10,  # Max absolute rotation angle in degrees (e.g., -10 to +10)
    scale=(0.9, 1.1),  # Range for random scaling (e.g., 90% to 110%)
    interpolation=InterpolationMode.BILINEAR,  # Interpolation method for the transformation
    fill=0,  # Pixel fill value for areas outside the rotated/scaled image
),
"""


def TRANSFORMS_TRAIN(mean, stdev):
    def BINARIZE_PIL(image):
        return

    return transforms.Compose(
        [
            transforms.Resize([155, 220], interpolation=InterpolationMode.BILINEAR),
            transforms.RandomInvert(p=1.0),
            transforms.ToTensor(),
            # OtsuBinarize(),
            # Divide by stdev but don't subtract by a mean value
            transforms.Normalize(mean=0, std=stdev),
        ]
    )


def TRANSFORMS_EVAL(mean, stdev):
    def BINARIZE_PIL(image):
        return

    return transforms.Compose(
        [
            transforms.Resize([155, 220], interpolation=InterpolationMode.BILINEAR),
            transforms.RandomInvert(p=1.0),
            transforms.ToTensor(),
            # OtsuBinarize(),
            # Divide by stdev but don't subtract by a mean value
            transforms.Normalize(mean=0, std=stdev),
        ]
    )
