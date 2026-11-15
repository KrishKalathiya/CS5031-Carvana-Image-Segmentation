import numpy as np

from src.models.classical.otsu import (
    otsu_threshold,
    otsu_segment
)

# To run this file, go to the project root and run: python -m pytest tests/test_otsu.py -v


def test_two_intensity_groups():
    gray = np.array([
        [30, 30, 30, 30],
        [30, 30, 30, 30],
        [200, 200, 200, 200],
        [200, 200, 200, 200]
    ], dtype=np.uint8)

    threshold = otsu_threshold(gray)

    # A valid separating threshold
    assert 30 <= threshold < 200


def test_binary_segmentation():
    image = np.zeros((10, 10, 3), dtype=np.uint8)

    image[:5, :, :] = 30
    image[5:, :, :] = 200

    mask = otsu_segment(image, invert=False)

    assert mask.shape == (10, 10)
    assert mask.dtype == np.uint8
    assert set(np.unique(mask)) == {0, 1}

    assert np.all(mask[:5] == 0)
    assert np.all(mask[5:] == 1)


def test_inverted_segmentation():
    image = np.zeros((10, 10, 3), dtype=np.uint8)

    image[:5, :, :] = 30
    image[5:, :, :] = 200

    mask = otsu_segment(image, invert=True)

    assert np.all(mask[:5] == 1)
    assert np.all(mask[5:] == 0)
