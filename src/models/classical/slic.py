"""RGB SLIC regions. Ground truth is never used to construct regions."""

import numpy as np
from skimage.segmentation import slic
from skimage.util import img_as_float32


def slic_regions(image, n_segments=800, compactness=10, sigma=1):
    """Return zero-based connected regions from RGB uint8 or float [0, 1].

    These defaults are starting settings, not paper-established optima.
    The requested region count is approximate, especially after connectivity.
    """
    image = np.asarray(image)
    if image.ndim != 3 or image.shape[2] != 3 or 0 in image.shape:
        raise ValueError("Expected a nonempty RGB image with shape (H, W, 3).")
    if image.dtype != np.uint8 and not np.issubdtype(image.dtype, np.floating):
        raise ValueError("Expected uint8 or floating-point RGB input.")
    rgb = img_as_float32(image)
    if not np.isfinite(rgb).all() or rgb.min() < 0 or rgb.max() > 1:
        raise ValueError("Floating-point RGB values must be finite and in [0, 1].")
    if isinstance(n_segments, bool) or not isinstance(n_segments, (int, np.integer)) or n_segments < 1:
        raise ValueError("n_segments must be a positive integer.")
    if not np.isfinite(compactness) or compactness <= 0:
        raise ValueError("compactness must be finite and positive.")
    if not np.isfinite(sigma) or sigma < 0:
        raise ValueError("sigma must be finite and nonnegative.")
    return slic(
        rgb, n_segments=n_segments, compactness=compactness, sigma=sigma,
        convert2lab=True, enforce_connectivity=True, start_label=0,
        channel_axis=-1,
    )
