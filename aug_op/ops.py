import random

import numpy as np
from PIL import Image, ImageEnhance, ImageOps

from .registry import Registry

aug_ops_dict = Registry()


@aug_ops_dict.register
class ShearX(object):
    RANGES = np.linspace(0, 0.3, 10)
    
    def __init__(self, magnitude):
        self.magnitude = magnitude
    
    def __call__(self, img):
        return img.transform(
            img.size, Image.AFFINE, (
                1, self.magnitude * random.choice([-1, 1]), 0, 0, 1, 0
            ), Image.BICUBIC, fillcolor=(128, 128, 128)
        )


@aug_ops_dict.register
class ShearY(object):
    RANGES = np.linspace(0, 0.3, 10)
    
    def __init__(self, magnitude):
        self.magnitude = magnitude
    
    def __call__(self, img):
        return img.transform(
            img.size, Image.AFFINE, (
                1, 0, 0, self.magnitude * random.choice([-1, 1]), 1, 0
            ), Image.BICUBIC, fillcolor=(128, 128, 128)
        )


@aug_ops_dict.register
class TranslateX(object):
    RANGES = np.linspace(0, 150 / 331, 10)
    
    def __init__(self, magnitude):
        self.magnitude = magnitude
    
    def __call__(self, img):
        return img.transform(
            img.size, Image.AFFINE, (
                1, 0, self.magnitude * img.size[0] * random.choice([-1, 1]), 0, 1, 0
            ), fillcolor=(128, 128, 128)
        )


@aug_ops_dict.register
class TranslateY(object):
    RANGES = np.linspace(0, 150 / 331, 10)
    
    def __init__(self, magnitude):
        self.magnitude = magnitude
    
    def __call__(self, img):
        return img.transform(
            img.size, Image.AFFINE, (
                1, 0, 0, 0, 1, self.magnitude * img.size[1] * random.choice([-1, 1])
            ), fillcolor=(128, 128, 128)
        )


@aug_ops_dict.register
class Rotate(object):
    RANGES = np.linspace(0, 30, 10)
    
    def __init__(self, magnitude):
        self.magnitude = magnitude
    
    def __call__(self, img):
        return img.rotate(self.magnitude * random.choice([-1, 1]))


@aug_ops_dict.register
class Color(object):
    RANGES = np.linspace(0.0, 0.9, 10)
    
    def __init__(self, magnitude):
        self.magnitude = magnitude
    
    def __call__(self, img):
        return ImageEnhance.Color(img).enhance(
            1 + self.magnitude * random.choice([-1, 1])
        )


@aug_ops_dict.register
class Posterize(object):
    RANGES = np.round(np.linspace(8, 4, 10), 0).astype(np.int)
    
    def __init__(self, magnitude):
        self.magnitude = magnitude
    
    def __call__(self, img):
        return ImageOps.posterize(img, self.magnitude)


@aug_ops_dict.register
class Solarize(object):
    RANGES = np.linspace(256, 0, 10)
    
    def __init__(self, magnitude):
        self.magnitude = magnitude
    
    def __call__(self, img):
        return ImageOps.solarize(img, self.magnitude)


@aug_ops_dict.register
class Contrast(object):
    RANGES = np.linspace(0.0, 0.9, 10)
    
    def __init__(self, magnitude):
        self.magnitude = magnitude
    
    def __call__(self, img):
        return ImageEnhance.Contrast(img).enhance(
            1 + self.magnitude * random.choice([-1, 1])
        )


@aug_ops_dict.register
class Sharpness(object):
    RANGES = np.linspace(0.0, 0.9, 10)
    
    def __init__(self, magnitude):
        self.magnitude = magnitude
    
    def __call__(self, img):
        return ImageEnhance.Sharpness(img).enhance(
            1 + self.magnitude * random.choice([-1, 1])
        )


@aug_ops_dict.register
class Brightness(object):
    RANGES = np.linspace(0.0, 0.9, 10)
    
    def __init__(self, magnitude):
        self.magnitude = magnitude
    
    def __call__(self, img):
        return ImageEnhance.Brightness(img).enhance(
            1 + self.magnitude * random.choice([-1, 1])
        )


@aug_ops_dict.register
class AutoContrast(object):
    RANGES = None
    
    def __init__(self):
        pass
    
    def __call__(self, img):
        return ImageOps.autocontrast(img)


@aug_ops_dict.register
class Equalize(object):
    RANGES = None
    
    def __init__(self):
        pass
    
    def __call__(self, img):
        return ImageOps.equalize(img)


@aug_ops_dict.register
class Invert(object):
    RANGES = None
    
    def __init__(self):
        pass
    
    def __call__(self, img):
        return ImageOps.invert(img)
