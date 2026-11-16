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

