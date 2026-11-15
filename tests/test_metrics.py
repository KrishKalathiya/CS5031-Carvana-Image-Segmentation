import numpy as np
import pytest

from src.evaluation.metrics import binary_segmentation_metrics


# This file contains unit tests for the binary_segmentation_metrics function in src/evaluation/metrics.py
# To run, go to the project root and run: pytest -v tests/test_metrics.py

def test_perfect_prediction():
    pred = np.array([
        [0, 0, 0],
        [0, 1, 1],
        [0, 1, 1],
    ])

    true = np.array([
            [0, 0, 0],
            [0, 1, 1],
            [0, 1, 1],
        ])

    result = binary_segmentation_metrics(pred, true)

    assert result["dice"] == 1.0
    assert result["iou"] == 1.0
    assert result["precision"] == 1.0
    assert result["recall"] == 1.0


def test_completely_wrong_prediction():
    true = np.array([
        [1, 1],
        [0, 0],
    ])

    pred = np.array([
        [0, 0],
        [1, 1],
    ])

    result = binary_segmentation_metrics(pred, true)

    assert result["dice"] == 0.0
    assert result["iou"] == 0.0
    assert result["precision"] == 0.0
    assert result["recall"] == 0.0


def test_partial_overlap():
    true = np.array([
        [1, 1],
        [0, 0],
    ])

    pred = np.array([
        [1, 0],
        [1, 0],
    ])

    # TP=1, FP=1, FN=1
    result = binary_segmentation_metrics(pred, true)

    assert result["dice"] == pytest.approx(0.5)
    assert result["iou"] == pytest.approx(1 / 3)
    assert result["precision"] == pytest.approx(0.5)
    assert result["recall"] == pytest.approx(0.5)


def test_both_masks_empty():
    mask = np.zeros((3, 3), dtype=np.uint8)

    result = binary_segmentation_metrics(mask, mask)

    assert all(value == 1.0 for value in result.values())


def test_shape_mismatch():
    pred = np.zeros((2, 2))
    true = np.zeros((3, 3))

    with pytest.raises(ValueError):
        binary_segmentation_metrics(pred, true)


def test_nonbinary_prediction():
    pred = np.array([[0, 0.5], [1, 0]])
    true = np.array([[0, 1], [1, 0]])

    with pytest.raises(ValueError):
        binary_segmentation_metrics(pred, true)
