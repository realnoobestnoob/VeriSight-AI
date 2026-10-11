import pytest

pytest.importorskip("torchvision")
from conftest import image_bytes

from verisight_ai.preprocessing import InvalidImageError, build_transform, decode_image_bytes


@pytest.mark.parametrize("fmt,expected", [("PNG", "png"), ("JPEG", "jpeg"), ("WEBP", "webp")])
def test_supported_formats(fmt, expected):
    d = decode_image_bytes(image_bytes(fmt))
    assert d.file_type == expected and d.image.mode == "RGB"


def test_rejects_empty():
    with pytest.raises(InvalidImageError):
        decode_image_bytes(b"")


def test_rejects_garbage_even_with_image_name():
    with pytest.raises(InvalidImageError):
        decode_image_bytes(b"this is not an image" * 10)


def test_rejects_unsupported_format():
    with pytest.raises(InvalidImageError, match="Unsupported"):
        decode_image_bytes(image_bytes("GIF"))


def test_rejects_oversize_bytes():
    with pytest.raises(InvalidImageError, match="larger"):
        decode_image_bytes(image_bytes("PNG"), max_bytes=100)


def test_rejects_too_many_pixels():
    with pytest.raises(InvalidImageError, match="dimensions"):
        decode_image_bytes(image_bytes("PNG", (50, 50)), max_pixels=100)


def test_rejects_truncated_file():
    data = image_bytes("PNG", (200, 200))
    with pytest.raises(InvalidImageError):
        decode_image_bytes(data[: len(data) // 2])


def test_eval_transform_shape_and_determinism():
    from conftest import make_image

    t = build_transform(32, train=False)
    img = make_image()
    a, b = t(img), t(img)
    assert a.shape == (3, 32, 32) and (a == b).all()


def test_train_transform_shape():
    from conftest import make_image

    assert build_transform(32, train=True)(make_image()).shape == (3, 32, 32)
