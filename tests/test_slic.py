import numpy as np
import pytest
from skimage.measure import label

from src.models.classical.slic import slic_regions
from src.evaluation.superpixels import majority_label_diagnostic


def test_majority_counts_crossing_and_ties():
    regions = np.array([[0, 0, 0, 4], [2, 2, 4, 4]])
    truth = np.array([[1, 1, 0, 0], [0, 1, 0, 0]], dtype=np.uint8)
    reconstructed, metrics = majority_label_diagnostic(regions, truth)
    np.testing.assert_array_equal(reconstructed, [[1, 1, 1, 0], [0, 0, 0, 0]])
    assert metrics['actual_regions'] == 3
    assert metrics['crossing_regions'] == 2
    assert metrics['minority_pixels'] == 2
    assert metrics['pixel_error_fraction'] == 0.25
    assert metrics['dice'] == pytest.approx(2 / 3)
    assert metrics['false_negative_pixels'] == metrics['false_positive_pixels'] == 1


@pytest.mark.parametrize('value', [0, 1])
def test_pure_regions_are_lossless(value):
    truth = np.full((5, 6), value, dtype=np.uint8)
    reconstruction, metrics = majority_label_diagnostic(np.zeros_like(truth), truth)
    np.testing.assert_array_equal(reconstruction, truth)
    assert metrics['dice'] == 1
    assert metrics['crossing_regions'] == 0


def test_rgb_integer_and_float_equivalence_and_connectivity():
    rng = np.random.default_rng(1)
    image = rng.integers(0, 256, (40, 60, 3), dtype=np.uint8)
    regions = slic_regions(image, n_segments=20)
    np.testing.assert_array_equal(regions, slic_regions(image.astype(np.float32) / 255, n_segments=20))
    assert regions.shape == image.shape[:2]
    assert regions.min() == 0
    for region_id in np.unique(regions):
        assert label(regions == region_id, connectivity=1).max() == 1


@pytest.mark.parametrize('image', [np.zeros((4, 5)), np.full((4, 5, 3), 255.0), np.full((4, 5, 3), np.nan)])
def test_reject_invalid_rgb(image):
    with pytest.raises(ValueError):
        slic_regions(image)


def test_reject_nonbinary_ground_truth_and_mismatch():
    with pytest.raises(ValueError):
        majority_label_diagnostic(np.zeros((2, 2), dtype=int), np.full((2, 2), 255))
    with pytest.raises(ValueError):
        majority_label_diagnostic(np.zeros((2, 2), dtype=int), np.zeros((3, 2)))
