"""Dataset and preprocessing utilities for the Kaggle Carvana dataset.

Carvana training files are commonly named:
    image:  <image_id>.jpg
    mask:   <image_id>_mask.gif

The dataset returns (image_tensor, mask_tensor, image_id), matching the
existing U-Net training loop.
"""

from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

try:  # Package import, e.g. from common.dataset import ...
    from .config import IMAGE_EXTENSIONS, MASK_EXTENSIONS
except ImportError:  # Direct import when this folder is on PYTHONPATH
    from config import IMAGE_EXTENSIONS, MASK_EXTENSIONS


ImageMaskPair = tuple[Path, Path, str]


def vehicle_id_from_image_id(image_id: str) -> str:
    """Return the vehicle identifier (the part before the view suffix)."""
    return image_id.split("_", maxsplit=1)[0]


def find_image_mask_pairs(image_dir: Path | str,
                          mask_dir: Path | str) -> list[ImageMaskPair]:
    """Match each image with its ``<image_id>_mask`` file.

    Raises a clear error if a directory is missing, no images are found, a
    duplicate mask ID is found, or any image has no corresponding mask.
    """
    image_dir = Path(image_dir)
    mask_dir = Path(mask_dir)

    if not image_dir.is_dir():
        raise NotADirectoryError(f"Image directory not found: {image_dir}")
    if not mask_dir.is_dir():
        raise NotADirectoryError(f"Mask directory not found: {mask_dir}")

    image_paths = sorted(
        p for p in image_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not image_paths:
        raise FileNotFoundError(f"No supported images found in: {image_dir}")

    mask_lookup: dict[str, Path] = {}
    for mask_path in sorted(mask_dir.iterdir()):
        if not mask_path.is_file() or mask_path.suffix.lower() not in MASK_EXTENSIONS:
            continue
        if not mask_path.stem.endswith("_mask"):
            continue

        image_id = mask_path.stem[:-5]
        if image_id in mask_lookup:
            raise ValueError(
                f"Multiple masks found for image ID {image_id!r}: "
                f"{mask_lookup[image_id]} and {mask_path}"
            )
        mask_lookup[image_id] = mask_path

    pairs: list[ImageMaskPair] = []
    missing: list[str] = []
    for image_path in image_paths:
        image_id = image_path.stem
        mask_path = mask_lookup.get(image_id)
        if mask_path is None:
            missing.append(image_id)
        else:
            pairs.append((image_path, mask_path, image_id))

    if missing:
        examples = ", ".join(missing[:5])
        raise FileNotFoundError(
            f"Missing masks for {len(missing)} image(s). Examples: {examples}. "
            f"Expected files named <image_id>_mask.gif (or another supported "
            f"mask extension) in {mask_dir}."
        )

    return pairs


def split_pairs(pairs: Sequence[ImageMaskPair],
                validation_fraction: float = 0.20,
                split_mode: str = "vehicle",
                seed: int = 42) -> tuple[list[ImageMaskPair], list[ImageMaskPair]]:
    """Split pairs randomly by image or safely by vehicle ID.

    ``vehicle`` prevents the 16 views of one vehicle from appearing in both
    train and validation sets. ``random`` is retained for a naive-split
    comparison only.
    """
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be strictly between 0 and 1")
    if split_mode not in {"vehicle", "random"}:
        raise ValueError("split_mode must be either 'vehicle' or 'random'")

    pairs = list(pairs)
    if len(pairs) < 2:
        raise ValueError("At least two image/mask pairs are required to split")

    import random
    rng = random.Random(seed)

    if split_mode == "random":
        shuffled = pairs.copy()
        rng.shuffle(shuffled)
        n_val = max(1, min(len(shuffled) - 1,
                           int(round(len(shuffled) * validation_fraction))))
        return shuffled[n_val:], shuffled[:n_val]

    groups: dict[str, list[ImageMaskPair]] = {}
    for pair in pairs:
        groups.setdefault(vehicle_id_from_image_id(pair[2]), []).append(pair)

    vehicle_ids = list(groups)
    if len(vehicle_ids) < 2:
        raise ValueError(
            "Vehicle-level splitting requires at least two distinct vehicle IDs"
        )
    rng.shuffle(vehicle_ids)

    target_val_images = max(1, int(round(len(pairs) * validation_fraction)))
    train_pairs: list[ImageMaskPair] = []
    val_pairs: list[ImageMaskPair] = []

    # Always reserve at least one entire vehicle group for training.
    for index, vehicle_id in enumerate(vehicle_ids):
        group = groups[vehicle_id]
        is_last_group = index == len(vehicle_ids) - 1
        if len(val_pairs) < target_val_images and not is_last_group:
            val_pairs.extend(group)
        else:
            train_pairs.extend(group)

    if not train_pairs or not val_pairs:
        raise RuntimeError("Could not create non-empty train and validation splits")

    train_vehicle_ids = {vehicle_id_from_image_id(pair[2]) for pair in train_pairs}
    val_vehicle_ids = {vehicle_id_from_image_id(pair[2]) for pair in val_pairs}
    overlap = train_vehicle_ids & val_vehicle_ids
    if overlap:
        raise RuntimeError(f"Vehicle leakage detected: {sorted(overlap)[:5]}")

    return train_pairs, val_pairs


def load_image_tensor(image_path: Path | str, image_size: int) -> tuple[torch.Tensor, tuple[int, int]]:
    """Load an RGB image and resize it to a square model input.

    Returns a float tensor [3, image_size, image_size] in [0, 1] and the
    original PIL size as (width, height).
    """
    image_path = Path(image_path)
    with Image.open(image_path) as source:
        image = source.convert("RGB")
        original_size = image.size
        image = image.resize(
            (image_size, image_size), resample=Image.Resampling.BILINEAR
        )
        array = np.asarray(image, dtype=np.float32) / 255.0

    tensor = torch.from_numpy(array).permute(2, 0, 1).contiguous()
    return tensor, original_size


def load_mask_tensor(mask_path: Path | str, image_size: int) -> torch.Tensor:
    """Load a GIF/PNG mask and return a binary float tensor [1, H, W]."""
    mask_path = Path(mask_path)
    with Image.open(mask_path) as source:
        try:
            source.seek(0)  # Use the first frame if the mask is an animated GIF.
        except EOFError:
            pass
        mask = source.convert("L")
        mask = mask.resize(
            (image_size, image_size), resample=Image.Resampling.NEAREST
        )
        array = np.asarray(mask, dtype=np.uint8)

    binary = (array >= 128).astype(np.float32)
    return torch.from_numpy(binary).unsqueeze(0).contiguous()


class CarvanaDataset(Dataset):
    """PyTorch dataset for pre-matched Carvana image/mask pairs."""

    def __init__(self, pairs: Sequence[ImageMaskPair], image_size: int = 256):
        if image_size <= 0:
            raise ValueError("image_size must be positive")
        self.pairs = list(pairs)
        self.image_size = int(image_size)

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int):
        image_path, mask_path, image_id = self.pairs[index]
        image, _ = load_image_tensor(image_path, self.image_size)
        mask = load_mask_tensor(mask_path, self.image_size)
        return image, mask, image_id


def preprocess_image(image_path: Path | str, image_size: int):
    """Compatibility helper for inference scripts.

    Returns a batched tensor [1, 3, H, W] and original size (width, height),
    matching the preprocessing in the existing predict.py.
    """
    image_tensor, original_size = load_image_tensor(image_path, image_size)
    return image_tensor.unsqueeze(0).contiguous(), original_size
