"""
Predict.py
----------

Run a trained Carvana U-Net on one image or every image in a folder.

Examples:
    python predict.py --image data/train/000c8a36845c_01.jpg
    python predict.py --input-dir data/sample_images
    python predict.py --image data/train/example.jpg --checkpoint outputs/best_unet_vehicle.pth

Outputs:
    A binary PNG mask for each input image.
    Optionally, an overlay showing the predicted mask on the original image.
"""

import argparse 
from pathlib import Path 

import numpy as np
from PIL import Image

import torch

from model import UNet

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}

def preprocess_image(image_path: Path, image_size: int):
    image = Image.open(image_path).convert("RGB")
    original_size = image.size # (width, height)

    resized = image.resize(
            (image_size, image_size),
            resample=Image.Resampling.BILINEAR
    )
    array = np.asarray(resized, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0).contiguous()

    return tensor, original_size

