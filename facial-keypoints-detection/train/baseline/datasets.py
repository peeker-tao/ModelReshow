import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import torchvision.transforms as T
import torchvision.transforms.functional as F
from torchvision.transforms import InterpolationMode

DEFAULT_FEATURE_NAMES = [
    "left_eye_center_x",
    "left_eye_center_y",
    "right_eye_center_x",
    "right_eye_center_y",
    "left_eye_inner_corner_x",
    "left_eye_inner_corner_y",
    "left_eye_outer_corner_x",
    "left_eye_outer_corner_y",
    "right_eye_inner_corner_x",
    "right_eye_inner_corner_y",
    "right_eye_outer_corner_x",
    "right_eye_outer_corner_y",
    "left_eyebrow_inner_end_x",
    "left_eyebrow_inner_end_y",
    "left_eyebrow_outer_end_x",
    "left_eyebrow_outer_end_y",
    "right_eyebrow_inner_end_x",
    "right_eyebrow_inner_end_y",
    "right_eyebrow_outer_end_x",
    "right_eyebrow_outer_end_y",
    "nose_tip_x",
    "nose_tip_y",
    "mouth_left_corner_x",
    "mouth_left_corner_y",
    "mouth_right_corner_x",
    "mouth_right_corner_y",
    "mouth_center_top_lip_x",
    "mouth_center_top_lip_y",
    "mouth_center_bottom_lip_x",
    "mouth_center_bottom_lip_y",
]


def _feature_index(name):
    return DEFAULT_FEATURE_NAMES.index(name)


def default_horizontal_flip_pairs():
    pair_names = [
        ("left_eye_center", "right_eye_center"),
        ("left_eye_inner_corner", "right_eye_inner_corner"),
        ("left_eye_outer_corner", "right_eye_outer_corner"),
        ("left_eyebrow_inner_end", "right_eyebrow_inner_end"),
        ("left_eyebrow_outer_end", "right_eyebrow_outer_end"),
        ("mouth_left_corner", "mouth_right_corner"),
    ]
    pairs = []
    for left_name, right_name in pair_names:
        left_x = _feature_index(f"{left_name}_x")
        left_y = _feature_index(f"{left_name}_y")
        right_x = _feature_index(f"{right_name}_x")
        right_y = _feature_index(f"{right_name}_y")
        pairs.append(((left_x, left_y), (right_x, right_y)))
    return pairs


def _to_tensor_image(image):
    if torch.is_tensor(image):
        return image.float()
    return torch.from_numpy(image).unsqueeze(0).float()


def _flip_keypoints(keypoints, image_width=96.0):
    flipped = keypoints.copy()
    flipped[0::2] = image_width - flipped[0::2]
    for left, right in default_horizontal_flip_pairs():
        flipped[[left[0], left[1], right[0], right[1]]] = flipped[
            [right[0], right[1], left[0], left[1]]
        ]
    return flipped


def _affine_keypoints(
    keypoints, angle, translate, scale, center, image_width=96.0, image_height=96.0
):
    theta = np.deg2rad(angle)
    cos_theta = np.cos(theta) * scale
    sin_theta = np.sin(theta) * scale
    cx, cy = center
    tx, ty = translate

    transformed = keypoints.copy().reshape(-1, 2)
    x = transformed[:, 0]
    y = transformed[:, 1]
    x_shifted = x - cx
    y_shifted = y - cy
    x_new = cos_theta * x_shifted - sin_theta * y_shifted + cx + tx
    y_new = sin_theta * x_shifted + cos_theta * y_shifted + cy + ty
    transformed[:, 0] = np.clip(x_new, 0.0, image_width)
    transformed[:, 1] = np.clip(y_new, 0.0, image_height)
    return transformed.reshape(-1).astype(np.float32)


class KeypointCompose:
    def __init__(self, transforms):
        self.transforms = list(transforms)

    def __call__(self, image, keypoints, mask=None):
        for transform in self.transforms:
            image, keypoints, mask = transform(image, keypoints, mask)
        return image, keypoints, mask


class RandomHorizontalFlipWithKeypoints:
    def __init__(self, p=0.5, image_width=96.0):
        self.p = p
        self.image_width = image_width

    def __call__(self, image, keypoints, mask=None):
        if np.random.rand() >= self.p:
            return image, keypoints, mask
        image = torch.flip(_to_tensor_image(image), dims=[2])
        keypoints = _flip_keypoints(
            np.asarray(keypoints, dtype=np.float32), self.image_width
        )
        return image, keypoints, mask


class RandomAffineWithKeypoints:
    def __init__(self, degrees=10, translate=0.02, scale=(0.95, 1.05), p=0.5):
        self.degrees = degrees
        self.translate = translate
        self.scale = scale
        self.p = p

    def __call__(self, image, keypoints, mask=None):
        if np.random.rand() >= self.p:
            return image, keypoints, mask

        image = _to_tensor_image(image)
        height, width = image.shape[-2], image.shape[-1]
        angle = float(np.random.uniform(-self.degrees, self.degrees))
        max_dx = (
            float(self.translate) * width
            if isinstance(self.translate, (int, float))
            else self.translate[0] * width
        )
        max_dy = (
            float(self.translate) * height
            if isinstance(self.translate, (int, float))
            else self.translate[1] * height
        )
        translations = (
            int(np.random.uniform(-max_dx, max_dx)),
            int(np.random.uniform(-max_dy, max_dy)),
        )
        if isinstance(self.scale, (tuple, list)):
            scale = float(np.random.uniform(self.scale[0], self.scale[1]))
        else:
            scale = float(self.scale)

        center = [width * 0.5, height * 0.5]
        image = F.affine(
            image,
            angle=angle,
            translate=list(translations),
            scale=scale,
            shear=[0.0, 0.0],
            interpolation=InterpolationMode.BILINEAR,
            fill=0,
            center=center,
        )
        keypoints = _affine_keypoints(
            np.asarray(keypoints, dtype=np.float32),
            angle=angle,
            translate=translations,
            scale=scale,
            center=center,
            image_width=float(width),
            image_height=float(height),
        )
        return image, keypoints, mask


class RandomBrightnessContrast:
    def __init__(self, brightness=0.2, contrast=0.2, p=0.5):
        self.brightness = brightness
        self.contrast = contrast
        self.p = p

    def __call__(self, image, keypoints, mask=None):
        if np.random.rand() >= self.p:
            return image, keypoints, mask
        image = _to_tensor_image(image)
        brightness_factor = float(
            np.random.uniform(1.0 - self.brightness, 1.0 + self.brightness)
        )
        contrast_factor = float(
            np.random.uniform(1.0 - self.contrast, 1.0 + self.contrast)
        )
        image = F.adjust_brightness(image, brightness_factor)
        image = F.adjust_contrast(image, contrast_factor)
        return image, keypoints, mask


class RandomGaussianNoise:
    def __init__(self, std=0.03, p=0.5):
        self.std = std
        self.p = p

    def __call__(self, image, keypoints, mask=None):
        if np.random.rand() >= self.p:
            return image, keypoints, mask
        image = _to_tensor_image(image)
        noise = torch.randn_like(image) * float(self.std)
        image = torch.clamp(image + noise, 0.0, 1.0)
        return image, keypoints, mask


class RandomGaussianBlur:
    def __init__(self, p=0.2, kernel_size=3, sigma=(0.1, 1.0)):
        self.p = p
        self.blur = T.GaussianBlur(kernel_size=kernel_size, sigma=sigma)

    def __call__(self, image, keypoints, mask=None):
        if np.random.rand() >= self.p:
            return image, keypoints, mask
        image = _to_tensor_image(image)
        image = self.blur(image)
        return image, keypoints, mask


class RandomCutout:
    def __init__(self, p=0.3, size=12, fill=0.0):
        self.p = p
        self.size = size
        self.fill = fill

    def __call__(self, image, keypoints, mask=None):
        if np.random.rand() >= self.p:
            return image, keypoints, mask
        image = _to_tensor_image(image)
        _, height, width = image.shape
        cutout_size = int(self.size)
        top = np.random.randint(0, max(1, height - cutout_size + 1))
        left = np.random.randint(0, max(1, width - cutout_size + 1))
        image[:, top : top + cutout_size, left : left + cutout_size] = float(self.fill)
        return image, keypoints, mask


class DatasetLoader(Dataset):
    def __init__(self, df, transform=None, augment=None):
        self.transform = transform
        self.augment = augment
        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        image = parse_image(self.df, idx)
        keypoint = self.df.iloc[idx, :-1].values.astype(np.float32)
        mask = ~np.isnan(keypoint)
        keypoint = np.nan_to_num(keypoint, nan=0.0).astype(np.float32)
        if self.augment is not None:
            image, keypoint, mask = self.augment(image, keypoint, mask)
        if self.transform is not None and not torch.is_tensor(image):
            image = _to_tensor_image(image)
        if self.transform:
            image = self.transform(image)
        else:
            transform = T.Compose(
                [
                    T.ToTensor(),
                    T.Resize((96, 96)),
                ]
            )
            image = transform(image)
        return image, keypoint, mask.astype(np.float32)


def parse_image(df, idx, size=(96, 96)):
    # 解析图像文件并返回图像数据
    arr = (
        np.array(df.Image.iloc[idx].split(" "), dtype=np.float32).reshape(size) / 255.0
    )
    return arr


class TestDataset(Dataset):
    def __init__(self, df, transform=None):
        self.transform = transform
        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        image = parse_image(self.df, idx)
        if self.transform is not None and not torch.is_tensor(image):
            image = _to_tensor_image(image)
        if self.transform:
            image = self.transform(image)
        else:
            transform = T.Compose(
                [
                    T.ToTensor(),
                    T.Resize((96, 96)),
                ]
            )
            image = transform(image)
        return image
