import numpy as np
import pandas as pd
from skimage.color import rgb2gray, rgb2lab
from skimage.filters import sobel
from skimage.util import img_as_float32


# Fixed order for both training and inference.
FEATURE_COLUMNS = [
    "lab_l_mean", "lab_l_std",
    "lab_a_mean", "lab_a_std",
    "lab_b_mean", "lab_b_std",
    "sobel_mean", "sobel_std",
    "centroid_x", "centroid_y",
    "area_fraction"
]


def extract_region_features(image, regions):
    """
    Calculate one feature row per region, indexed by region ID.

    Input RGB is uint8 [0, 255] or float [0, 1]. Lab uses D65/2-degree
    units: L* is about [0, 100]; a*/b* are signed, typically roughly
    [-128, 127]. Channels are not clipped or standardized.

    Sobel uses grayscale [0, 1] with reflect padding; its magnitude is
    nonnegative and at most 1, apart from rounding. All standard deviations
    are population values (ddof=0), in the corresponding channel's units.
    Centroids are normalized to [0, 1]; area_fraction is in (0, 1].

    Features use only the image and regions. Ground truth is not an input.
    """
    rgb = img_as_float32(image)
    lab = rgb2lab(rgb)

    # Compute the edge map before selecting pixels from individual regions.
    gray = rgb2gray(rgb)
    edges = sobel(gray, mode="reflect")

    region_ids, pixel_regions = np.unique(
        regions, return_inverse=True
    )
    pixel_regions = pixel_regions.ravel()
    areas = np.bincount(pixel_regions)

    def mean_std(values):
        values = values.ravel()
        means = np.bincount(
            pixel_regions, weights=values
        ) / areas

        squared_errors = (values - means[pixel_regions]) ** 2
        variances = np.bincount(
            pixel_regions, weights=squared_errors
        ) / areas

        return means, np.sqrt(variances)

    features = {}

    for channel, name in enumerate(["l", "a", "b"]):
        means, stds = mean_std(lab[:, :, channel])
        features[f"lab_{name}_mean"] = means
        features[f"lab_{name}_std"] = stds

    features["sobel_mean"], features["sobel_std"] = mean_std(edges)

    height, width = regions.shape
    y, x = np.indices(regions.shape)

    # Divide by width-1 / height-1 so image edges map to 0 and 1.
    features["centroid_x"] = np.bincount(
        pixel_regions, weights=x.ravel()
    ) / areas / max(width - 1, 1)

    features["centroid_y"] = np.bincount(
        pixel_regions, weights=y.ravel()
    ) / areas / max(height - 1, 1)

    features["area_fraction"] = areas / regions.size

    return pd.DataFrame(
        features,
        index=pd.Index(region_ids, name="region_id"),
        columns=FEATURE_COLUMNS
    ).astype(np.float32)
