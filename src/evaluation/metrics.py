import numpy as np

# This file contains functions for evaluating binary segmentation predictions against ground truth masks


def binary_segmentation_metrics(pred_mask, true_mask):
    """
    Calculate binary segmentation metrics for one image.

    Parameters
    ----------
    pred_mask : numpy.ndarray
        Predicted binary mask (0 = background, 1 = car).
    true_mask : numpy.ndarray
        Ground-truth binary mask (0 = background, 1 = car).

    Returns
    -------
    dict
        Dice, IoU, precision, and recall.
    """
    pred = np.asarray(pred_mask)
    true = np.asarray(true_mask)

    if pred.shape != true.shape:
        raise ValueError(
            f"Shape mismatch: {pred.shape} vs {true.shape}"
        )

    if not np.isin(pred, [0, 1]).all():
        raise ValueError("Prediction must be a binary mask.")

    if not np.isin(true, [0, 1]).all():
        raise ValueError("Ground truth must be a binary mask.")

    pred = pred.astype(bool)
    true = true.astype(bool)

    tp = np.count_nonzero(pred & true)
    fp = np.count_nonzero(pred & ~true)
    fn = np.count_nonzero(~pred & true)

    # Empty-mask convention:
    # Both empty -> perfect match (1.0)
    # Only one empty -> zero overlap (0.0)
    dice_den = 2 * tp + fp + fn
    iou_den = tp + fp + fn

    dice = (
        2 * tp / dice_den
        if dice_den > 0 else 1.0
    )

    iou = (
        tp / iou_den
        if iou_den > 0 else 1.0
    )

    # No predicted positives:
    # precision = 1 if GT also empty, otherwise 0.
    precision = (
        tp / (tp + fp)
        if (tp + fp) > 0
        else float(tp + fn == 0)
    )

    # No actual positives:
    # recall = 1 if prediction also empty, otherwise 0.
    recall = (
        tp / (tp + fn)
        if (tp + fn) > 0
        else float(tp + fp == 0)
    )

    return {
        "dice": float(dice),
        "iou": float(iou),
        "precision": float(precision),
        "recall": float(recall),
    }
