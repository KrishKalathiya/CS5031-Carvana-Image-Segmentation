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
