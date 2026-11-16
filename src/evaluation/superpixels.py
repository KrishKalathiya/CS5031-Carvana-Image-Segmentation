"""Ground-truth diagnostics for information lost by one label per region."""

import numpy as np

from src.evaluation.metrics import binary_segmentation_metrics


def majority_label_diagnostic(regions, true_mask):
    """Return majority reconstruction, region counts, and image metrics.

    A crossing region contains at least one foreground AND background pixel.
    Exact ties receive background. Majority vote minimizes pixel errors for
    fixed regions, but is not necessarily the maximum attainable Dice.
    This uses ground truth and is not a deployable predictor.
    """
    regions = np.asarray(regions)
    truth = np.asarray(true_mask)
    if regions.ndim != 2 or regions.size == 0 or regions.shape != truth.shape:
        raise ValueError("Regions and mask must be nonempty matching 2D arrays.")
    if not np.issubdtype(regions.dtype, np.integer) or np.any(regions < 0):
        raise ValueError("Region IDs must be nonnegative integers.")
    if not np.isin(truth, [0, 1]).all():
        raise ValueError("Ground truth must be binary.")
    # Compact indexing also handles nonconsecutive region IDs safely.
    ids, inverse = np.unique(regions, return_inverse=True)
    inverse = inverse.ravel()
    sizes = np.bincount(inverse)
    foreground = np.bincount(inverse, weights=truth.ravel()).astype(np.int64)
    background = sizes - foreground
    crossing = (foreground > 0) & (background > 0)
    predicted_labels = foreground > background
    reconstructed = predicted_labels[inverse].reshape(truth.shape).astype(np.uint8)
    metrics = binary_segmentation_metrics(reconstructed, truth)
    metrics.update(
        actual_regions=int(ids.size),
        crossing_regions=int(crossing.sum()),
        crossing_fraction=float(crossing.mean()),
        minority_pixels=int(np.minimum(foreground, background).sum()),
        pixel_error_fraction=float(np.minimum(foreground, background).sum() / truth.size),
        false_negative_pixels=int(np.count_nonzero((truth == 1) & (reconstructed == 0))),
        false_positive_pixels=int(np.count_nonzero((truth == 0) & (reconstructed == 1))),
    )
    return reconstructed, metrics
