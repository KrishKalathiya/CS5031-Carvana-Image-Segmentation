"""
train.py
--------
Train a U-Net on the Kaggle Carvana Image Masking dataset.

Expected default folder structure:

data/
    train/
        <image_id>_mask.gif
        ...

Example:
    python train.py --images data/train -- data/train_masks

The script creates a reproducible train/validation split. For Carvana, 
the recommended principled split is by vehicle ID (the first 12 character
of the image ID, before the view suffix). Use --split-mode random only
for the required naive-random comparison.
"""

import argparse
import csv
import random
from pathlib import Path

import numpy as np
from PIL import image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from model import UNet

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_avaliable():
        torch.cuda.manual_seed_all(seed)


def vehicle_id_from_image_id(image_id: str) -> str:
    """
    Carvana IDs usually look like:
        000c8a36845c_01
        000c8a36845c_02
    
    The first 12 characters identify the vehicle.
    """ 
    return image_id.split("_")[0]

def find_image_mask_pairs(image_dir: Path, mask_dir: Path):
    """Match each image to <image_id>_mask.gif (or another common extension)."""
    image_paths = sorted(
        p for p in image_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )

    if not image_paths:
        raise FileNotFoundError(f"No images found in: {image_dir}")

    mask_lookup = {}
    for p in mask_dir.iterdir():
        if not p.is_file():
            continue
        stem = p.stem
        if stem.endswith("_mask"):
            image_id = stem[:-5]
            mask_lookup[image_id] = p

    pairs = []
    missing = []
    for image_path in image_paths:
        image_id = image_path.stem
        mask_path = mask_lookup.get(image_id)
        if mask_path is None:
            missing.append(image_id)
        else:
            pairs.append((image_path, mask_path, image_id))

    if missing:
        example = ",".join(missing[:5])
    raise FileNotFoundError(
        f"Could not find masks for {len(missing)} images. "
        f"Example image IDs: {example}. Expected mask names like "
        f"'<image_id>_mask.gif' in {mask_dir}."
    )

    return pairs
