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

def save_prediction(probabilities, original_image_path, output_dir,
                    original_size, threshold, save_overlay):
    output_dir.mkdir(parents=True, exist_ok=True)
    image_id = original_image_path.stem

    mask_array = (probabilities >= threshold).astype(np.uint8) * 255
    mask_image = Image.fromarray(mask_array, mode="L")
    mask_image = mask_image.resize(original_size, resample=Image.Resampling.NEAREST)

    mask_path = output_dir / f"{image_id}_predicted_mask.png"
    mask_image.save(mask_path)

    if save_overlay:
        original = Image.open(original_image_path).convert("RGB")
        overlay = np.asarray(original, dtype=np.uint8).copy()
        mask_full = np.asarray(mask_image, dtype=np.uint8) > 0

    # Tint predicted foreground red so the car boundary is easy to inspect.
        overlay[mask_full, 0] = np.clip(
            overlay[mask_full, 0].astype(np.int16) * 0.5 + 127, 0, 255
        ).astype(np.uint8)
        overlay[mask_full, 1] = (
            overlay[mask_full, 1].astype(np.float32) * 0.5
        ).astype(np.uint8)
        overlay[mask_full, 2] = (
            overlay[mask_full, 2].astype(np.float32) * 0.5
        ).astype(np.uint8)

        overlay_path = output_dir / f"{image_id}_overlay.png"
        Image.fromarray(overlay).save(overlay_path)
        return mask_path, overlay_path

    return mask_path, None

