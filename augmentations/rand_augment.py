import numpy as np
import albumentations as A


__all__ = [
    "RandAugment",
    "ShearX",
    "ShearY",
    "TranslateX",
    "TranslateY",
    "Rotate",
    "GaussianBlur",
    "Solarize",
    "RandomFog",
    "RandomRain",
    "GlassBlur",
    "RandomHue",
    "RandomValue",
    "RandomSaturation",
    "RandomBrightness",
    "RandomContrast",
    "OpticalDistortion",
    "Posterize",
    "Downscale",
    "Sharpen"
]

# define transforms
'''
Here we convert strength to the range [0, 1]
'''

def ShearX(strength: float, p: float = 1.0):
    shear = strength * 0.3
    return A.Affine(shear={'x': (0, shear), 'y': 0.0}, p=p)

def ShearY(strength: float, p: float = 1.0):
    shear = strength * 0.3
    return A.Affine(shear={'x': 0.0, 'y': (0, shear)}, p=p)

def TranslateX(strength: float, p: float = 1.0):
    translate = strength * 0.45
    return A.Affine(translate_percent={'x': (0, translate), 'y': 0.0}, p=p)

def TranslateY(strength: float, p: float = 1.0):
    translate = strength * 0.45
    return A.Affine(translate_percent={'x': 0.0, 'y': (0, translate)}, p=p)

def Rotate(strength: float, p: float = 1.0):
    rotate = strength * 30.0
    return A.Affine(rotate=rotate, p=p)

def GaussianBlur(strength: float, p: float = 1.0):
    assert 0 <= strength <= 1
    blur_limit = (1, 1 + 2 * int(strength * 10))
    return A.GaussianBlur(blur_limit=blur_limit, p=p)

def Solarize(strength: float, p: float = 1.0):
    assert 0 <= strength <= 1
    threshold = int(strength * 255)
    return A.Solarize(threshold=threshold, p=p)

def Superpixels(strength: float, p: float = 1.0):
    assert 0 <= strength <= 1
    p_replace = 0.5 * strength
    return A.Superpixels(p_replace=(0, p_replace), p=p)

def RandomRain(strength: float, p: float = 1.0):
    assert 0 <= strength <= 1
    drop_width = int(1 + 2 * strength)
    blur_value = int(1 + 10 * strength)
    brightness_coefficient = 1.0 - 0.5 * strength
    return A.RandomRain(
        drop_width=drop_width,
        blur_value=blur_value,
        brightness_coefficient=brightness_coefficient,
        p=p
    )

def RandomFog(strength: float, p: float = 1.0):
    assert 0 <= strength <= 1
    fog_coef_upper = strength
    alpha_coef = 0.5 * strength
    return A.RandomFog(
        fog_coef_lower=0.0, 
        fog_coef_upper=fog_coef_upper, 
        alpha_coef=alpha_coef, 
        p=p
    )

def RandomBrightness(limit: float, p: float = 1.0):
    assert 0 <= limit <= 1
    return A.RandomBrightnessContrast(brightness_limit=limit, p=p)

def RandomContrast(limit: float, p: float = 1.0):
    assert 0 <= limit <= 1
    return A.RandomBrightnessContrast(contrast_limit=limit, p=p)

def GlassBlur(strength: float, p: float = 1.0):
    assert 0 <= strength <= 1
    max_delta = int(1 + 10 * strength)
    return A.GlassBlur(max_delta=max_delta, iterations=1, p=p)

def RandomHue(strength: float, p: float = 1.0):
    assert 0 <= strength <= 1
    hue_shift_limit = 40 * strength
    return A.HueSaturationValue(hue_shift_limit=hue_shift_limit, p=p)

def RandomSaturation(strength: float, p: float = 1.0):
    assert 0 <= strength <= 1
    sat_shift_limit = 60 * strength
    return A.HueSaturationValue(sat_shift_limit=sat_shift_limit, p=p)

def RandomValue(strength: float, p: float = 1.0):
    assert 0 <= strength <= 1
    val_shift_limit = 40 * strength
    return A.HueSaturationValue(val_shift_limit=val_shift_limit, p=p)

def OpticalDistortion(strength: float, p: float = 1.0):
    assert 0 <= strength <= 1
    distort_limit = strength
    return A.OpticalDistortion(distort_limit=distort_limit, p=p)

def Posterize(strength: float, p: float = 1.0):
    assert 0 <= strength <= 1
    num_bits = 8 - int(7 * strength)
    return A.Posterize(num_bits=num_bits, p=p)

def Downscale(strength: float, p: float = 1.0):
    assert 0 <= strength <= 1
    scale_min = 0.98 - 0.9 * strength
    return A.Downscale(scale_min, scale_max=0.99, p=p)

def Sharpen(strength: float, p: float = 1.0):
    assert 0 <= strength <= 1
    return A.Sharpen(
        alpha=(0.5 * strength, strength),
        lightness=(1 - 0.5 * strength, 1.0),
        p=p
    )

RAND_TRANSFORMS = [
    ShearX,
    ShearY,
    TranslateX,
    TranslateY,
    Rotate,
    GaussianBlur,
    Solarize,
    RandomFog,
    RandomRain,
    GlassBlur,
    RandomHue,
    RandomValue,
    RandomSaturation,
    RandomBrightness,
    RandomContrast,
    OpticalDistortion,
    Posterize,
    Downscale,
    Sharpen
]

# weights choice

RAND_CHOICE_WEIGHTS_0 = [
    0.1, 
    0.1, 
    0.13,
    0.13, 
    0.2,
    0.025,
    0.025,
    0.005,
    0.005,
    0.005,
    0.05,
    0.05,
    0.05,
    0.020,
    0.020,
    0.025, 
    0.020, 
    0.020,
    0.020
]

# define RandAugment Transform

class RandAugment:

    def __init__(
        self,
        num_transforms: int = 2,
        mag_transforms: float = 0.5,
        mag_deviation: float = 0.0,
        p: float = 1.0,
        w0: bool = False,
    ) -> None:
        assert 0 <= num_transforms <= len(RAND_TRANSFORMS)
        assert 0 <= mag_transforms <= 1
        # set params
        self.num_transforms=num_transforms
        self.mag_transforms=mag_transforms
        self.mag_deviation=mag_deviation
        self.w0 = w0

        self.transforms_list = []
        for RAND_TRANSFORM in RAND_TRANSFORMS: 
            mag_transform = np.random.uniform(
                low=mag_transforms - mag_deviation, 
                high=mag_transforms - mag_deviation
            )
            mag_transform = np.clip(mag_transform, 0, 1)

            self.transforms_list.append(
                RAND_TRANSFORM(mag_transform, p=p)
            )

    def __call__(self, **inputs):
        transforms = np.random.choice(
            self.transforms_list, 
            self.num_transforms,
            p=(RAND_CHOICE_WEIGHTS_0 if self.w0 else None)
        )
        for transform in transforms:
            inputs = transform(**inputs)
        return inputs

    def __repr__(self) -> str:
        fs = f"{self.__class__.__name__}({self.num_transforms}/{self.mag_transforms} +- {self.mag_deviation})"
        return fs
