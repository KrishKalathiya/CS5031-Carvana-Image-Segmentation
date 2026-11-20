
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image


def load_image(image_path, size=None):
    """
    Load a Carvana RGB image.

    Parameters
    ----------
    image_path : str or Path
        Path to the input JPEG image.
    size : tuple or None
        Optional (width, height) for resizing.

    Returns
    -------
    numpy.ndarray
        RGB image with shape (H, W, 3), dtype uint8.
    """
    with Image.open(image_path) as image:
        image = image.convert("RGB")

        if size is not None:
            image = image.resize(
                size,
                resample=Image.Resampling.LANCZOS
            )

        return np.asarray(image)


def load_mask(mask_path, size=None):
    """
    Load a Carvana ground-truth mask.

    GIF masks are converted to grayscale before
    thresholding to avoid interpreting palette
    indices as actual pixel intensities.

    Returns
    -------
    numpy.ndarray
        Binary mask with shape (H, W), dtype uint8.
        1 = vehicle, 0 = background.
    """
    with Image.open(mask_path) as mask:
        mask = mask.convert("L")

        if size is not None:
            mask = mask.resize(
                size,
                resample=Image.Resampling.NEAREST
            )

        mask_array = np.asarray(mask)

    return (mask_array > 127).astype(np.uint8)


def load_pair(image_path, mask_path, size=None):
    """
    Load an input image and its ground-truth mask.
    """
    image = load_image(image_path, size=size)
    mask = load_mask(mask_path, size=size)

    if image.shape[:2] != mask.shape:
        raise ValueError(
            "Image and mask dimensions do not match."
        )

    return image, mask


def load_manifest(csv_path, project_root):
    """
    Read one of the saved dataset split CSV files.

    Converts paths relative to the project root
    into absolute filesystem paths.
    """
    df = pd.read_csv(csv_path)

    project_root = Path(project_root).resolve()

    df["image"] = df["image"].apply(
        lambda p: project_root / p
    )

    df["mask"] = df["mask"].apply(
        lambda p: project_root / p
    )

    return df
