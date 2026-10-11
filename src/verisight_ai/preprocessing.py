"""Safe image decoding, validation and torchvision transforms.

Augmentation is applied to training data only. JPEG recompression, resizing, cropping and
screenshot-like processing are NOT training augmentations here; they are robustness tests
(see evaluate.py) so transformed copies cannot leak across splits.
"""

from __future__ import annotations

import io
from dataclasses import dataclass

from PIL import Image, ImageOps, UnidentifiedImageError
from torchvision import transforms

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
ALLOWED_FORMATS = {"JPEG": "jpeg", "PNG": "png", "WEBP": "webp"}


class InvalidImageError(ValueError):
    """Raised when an upload is rejected. The message is safe to show to the user."""


@dataclass(frozen=True)
class DecodedImage:
    image: Image.Image  # RGB
    file_type: str  # jpeg | png | webp
    width: int
    height: int


def decode_image_bytes(
    data: bytes, max_bytes: int = 8 * 1024 * 1024, max_pixels: int = 25_000_000
) -> DecodedImage:
    """Validate by actually decoding (extension/MIME checks are not security controls)."""
    if not data:
        raise InvalidImageError("The file is empty.")
    if len(data) > max_bytes:
        raise InvalidImageError(
            f"The file is larger than the {max_bytes / 1024 / 1024:.0f} MB limit."
        )
    try:
        probe = Image.open(io.BytesIO(data))
        fmt = probe.format
        width, height = probe.size
    except (UnidentifiedImageError, OSError, ValueError, SyntaxError) as exc:
        raise InvalidImageError("This file could not be read as an image.") from exc
    if fmt not in ALLOWED_FORMATS:
        raise InvalidImageError("Unsupported image type. Please use JPEG, PNG or WebP.")
    if width <= 0 or height <= 0 or width * height > max_pixels:
        raise InvalidImageError("The image dimensions exceed the allowed limit.")
    if getattr(probe, "is_animated", False) and getattr(probe, "n_frames", 1) > 1:
        raise InvalidImageError("Animated images are not supported. Upload a single still image.")
    try:
        # Fresh handle: verify() invalidates the image, so decode from a second handle.
        Image.open(io.BytesIO(data)).verify()
        img = Image.open(io.BytesIO(data))
        img.load()
        img = ImageOps.exif_transpose(img)
        img = img.convert("RGB")
    except (OSError, ValueError, SyntaxError, Image.DecompressionBombError) as exc:
        raise InvalidImageError("This image appears to be corrupted or truncated.") from exc
    return DecodedImage(img, ALLOWED_FORMATS[fmt], width, height)


def build_transform(image_size: int, train: bool, mean=IMAGENET_MEAN, std=IMAGENET_STD):
    """Eval transform is deterministic; train adds modest label-preserving augmentation."""
    ops: list = []
    if train:
        ops += [
            transforms.RandomResizedCrop(
                image_size,
                scale=(0.8, 1.0),
                ratio=(0.9, 1.1),
                interpolation=transforms.InterpolationMode.BILINEAR,
            ),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
        ]
    else:
        ops += [
            transforms.Resize(
                (image_size, image_size), interpolation=transforms.InterpolationMode.BILINEAR
            )
        ]
    ops += [transforms.ToTensor(), transforms.Normalize(mean=list(mean), std=list(std))]
    return transforms.Compose(ops)
