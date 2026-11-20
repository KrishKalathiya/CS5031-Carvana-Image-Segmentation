import numpy as np


# Custom implementation of Otsu's thresholding algorithm
def otsu_threshold(gray):
    """
    Compute Otsu's optimal grayscale threshold
    using NumPy, without OpenCV.

    Parameters
    ----------
    gray : ndarray
        Grayscale image, uint8, values 0-255.

    Returns
    -------
    int
        Optimal threshold.
    """
    gray = np.asarray(gray)

    if gray.ndim != 2 or gray.dtype != np.uint8:
        raise ValueError("Expected a 2D uint8 grayscale image.")

    # Count pixels at each grayscale intensity
    histogram = np.bincount(
        gray.ravel(), minlength=256
    ).astype(np.float64)

    # Normalize histogram into probabilities
    probabilities = histogram / histogram.sum()

    intensities = np.arange(256)

    # Cumulative class probabilities
    w0 = np.cumsum(probabilities)
    w1 = 1.0 - w0

    # Cumulative weighted intensities
    cumulative_mean = np.cumsum(
        probabilities * intensities
    )

    total_mean = cumulative_mean[-1]

    # Between-class variance for each threshold
    variance = np.zeros(256, dtype=np.float64)

    valid = (w0 > 0) & (w1 > 0)

    variance[valid] = (
        (total_mean * w0[valid]
         - cumulative_mean[valid]) ** 2
        / (w0[valid] * w1[valid])
    )

    return int(np.argmax(variance))


def otsu_segment(image, invert=False):
    """
    Segment an RGB image using our own
    Otsu threshold implementation.

    Returns a binary uint8 mask (0 or 1).
    """

    # Convert RGB to grayscale using luminance
    gray = (
        0.299 * image[:, :, 0].astype(np.float64)
        + 0.587 * image[:, :, 1].astype(np.float64)
        + 0.114 * image[:, :, 2].astype(np.float64)
    ).round().astype(np.uint8)

    threshold = otsu_threshold(gray)

    # Apply threshold
    if invert:
        mask = gray <= threshold
    else:
        mask = gray > threshold

    return mask.astype(np.uint8)
