import random
import numpy as np
import torch
import torchvision.transforms.functional as TF
from torchvision import transforms
from PIL import Image
from typing import Tuple, Optional


class Resize:
    def __init__(self, size: int):
        self.size = size

    def __call__(self, img_a: Image.Image, img_b: Image.Image, mask: Image.Image):
        img_a = TF.resize(img_a, [self.size, self.size], interpolation=TF.InterpolationMode.BICUBIC)
        img_b = TF.resize(img_b, [self.size, self.size], interpolation=TF.InterpolationMode.BICUBIC)
        mask = TF.resize(mask, [self.size, self.size], interpolation=TF.InterpolationMode.NEAREST)
        return img_a, img_b, mask


class RandomHorizontalFlip:
    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(self, img_a: Image.Image, img_b: Image.Image, mask: Image.Image):
        if random.random() < self.p:
            img_a = TF.hflip(img_a)
            img_b = TF.hflip(img_b)
            mask = TF.hflip(mask)
        return img_a, img_b, mask


class RandomVerticalFlip:
    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(self, img_a: Image.Image, img_b: Image.Image, mask: Image.Image):
        if random.random() < self.p:
            img_a = TF.vflip(img_a)
            img_b = TF.vflip(img_b)
            mask = TF.vflip(mask)
        return img_a, img_b, mask


class RandomRotation:
    def __init__(self, degrees: Tuple[float, float] = (-90, 90), p: float = 0.5):
        self.degrees = degrees
        self.p = p

    def __call__(self, img_a: Image.Image, img_b: Image.Image, mask: Image.Image):
        if random.random() < self.p:
            angle = random.uniform(*self.degrees)
            img_a = TF.rotate(img_a, angle, interpolation=TF.InterpolationMode.BICUBIC)
            img_b = TF.rotate(img_b, angle, interpolation=TF.InterpolationMode.BICUBIC)
            mask = TF.rotate(mask, angle, interpolation=TF.InterpolationMode.NEAREST)
        return img_a, img_b, mask


class RandomColorJitter:
    def __init__(self, brightness: float = 0.2, contrast: float = 0.2,
                 saturation: float = 0.2, hue: float = 0.05, p: float = 0.5):
        self.p = p
        self.jitter = transforms.ColorJitter(brightness, contrast, saturation, hue)

    def __call__(self, img_a: Image.Image, img_b: Image.Image, mask: Image.Image):
        if random.random() < self.p:
            img_a = self.jitter(img_a)
            img_b = self.jitter(img_b)
        return img_a, img_b, mask


class RandomGrayscale:
    def __init__(self, p: float = 0.1):
        self.p = p

    def __call__(self, img_a: Image.Image, img_b: Image.Image, mask: Image.Image):
        if random.random() < self.p:
            img_a = TF.to_grayscale(img_a, num_output_channels=3)
            img_b = TF.to_grayscale(img_b, num_output_channels=3)
        return img_a, img_b, mask


class RandomGaussianBlur:
    def __init__(self, kernel_size: int = 5, sigma: Tuple[float, float] = (0.1, 2.0), p: float = 0.3):
        self.p = p
        self.blur = transforms.GaussianBlur(kernel_size, sigma)

    def __call__(self, img_a: Image.Image, img_b: Image.Image, mask: Image.Image):
        if random.random() < self.p:
            img_a = self.blur(img_a)
            img_b = self.blur(img_b)
        return img_a, img_b, mask


class Normalize:
    def __init__(self, mean: Tuple[float, float, float], std: Tuple[float, float, float]):
        self.mean = mean
        self.std = std

    def __call__(self, img_a: torch.Tensor, img_b: torch.Tensor):
        img_a = TF.normalize(img_a, self.mean, self.std)
        img_b = TF.normalize(img_b, self.mean, self.std)
        return img_a, img_b


class ToTensor:
    def __call__(self, img_a: Image.Image, img_b: Image.Image, mask: Image.Image):
        img_a = TF.to_tensor(img_a)
        img_b = TF.to_tensor(img_b)
        mask = torch.from_numpy(np.array(mask)).long()
        mask = (mask > 0).long()
        return img_a, img_b, mask


class ChangeDetectionTransform:
    def __init__(self, img_size: int, mean: Tuple, std: Tuple, training: bool = True):
        self.img_size = img_size
        self.training = training

        self.spatial_augs = [
            Resize(img_size),
            RandomHorizontalFlip(p=0.5),
            RandomVerticalFlip(p=0.5),
            RandomRotation(degrees=(-90, 90), p=0.5),
        ]

        self.photometric_augs = [
            RandomColorJitter(p=0.5),
            RandomGrayscale(p=0.1),
            RandomGaussianBlur(p=0.3),
        ]

        self.val_spatial = [Resize(img_size)]
        self.to_tensor = ToTensor()
        self.normalize = Normalize(mean, std)

    def __call__(self, img_a: Image.Image, img_b: Image.Image,
                 mask: Image.Image) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.training:
            for aug in self.spatial_augs:
                img_a, img_b, mask = aug(img_a, img_b, mask)
            for aug in self.photometric_augs:
                img_a, img_b, mask = aug(img_a, img_b, mask)
        else:
            for aug in self.val_spatial:
                img_a, img_b, mask = aug(img_a, img_b, mask)

        img_a, img_b, mask = self.to_tensor(img_a, img_b, mask)
        img_a, img_b = self.normalize(img_a, img_b)
        return img_a, img_b, mask


def build_transform(img_size: int, mean: Tuple, std: Tuple, training: bool) -> ChangeDetectionTransform:
    return ChangeDetectionTransform(img_size=img_size, mean=mean, std=std, training=training)
