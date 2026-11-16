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

def main():
    parser = argparse.ArgumentParser(description="Predict Carvana segmentation masks.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--image", type=Path, help="Path to one image.")
    source.add_argument("--input-dir", type=Path, help="Folder of images.")
    parser.add_argument(
        "--checkpoint", type=Path,
        default=Path("outputs/best_unet_vehicle.pth")
    )
    parser.add_argument("--output", type=Path, default=Path("predictions"))
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--save-overlay", action="store_true")
    args = parser.parse_args()

    if not 0.0 <= args.threshold <= 1.0:
        raise ValueError("--threshold must be between 0 and 1.")

    if not args.checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location=device)

    image_size = int(checkpoint.get("image_size", 256))
    model = UNet().to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    if args.image is not None:
        image_paths = [args.image]
    else:
        if not args.input_dir.is_dir():
            raise NotADirectoryError(f"Input folder not found: {args.input_dir}")
        image_paths = sorted(
            p for p in args.input_dir.iterdir()
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        )

    if not image_paths:
        raise FileNotFoundError("No supported images were found.")

    args.output.mkdir(parents=True, exist_ok=True)
    print(f"Device: {device}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Images to predict: {len(image_paths)}")

    with torch.no_grad():
        for image_path in image_paths:
            if not image_path.is_file():
                print(f"Skipping missing image: {image_path}")
                continue

            tensor, original_size = preprocess_image(image_path, image_size)
            tensor = tensor.to(device)

            logits = model(tensor)
            probabilities = torch.sigmoid(logits)[0, 0].cpu().numpy()

            mask_path, overlay_path = save_prediction(
                probabilities=probabilities,
                original_image_path=image_path,
                output_dir=args.output,
                original_size=original_size,
                threshold=args.threshold,
                save_overlay=args.save_overlay,
            )

            print(f"Saved mask: {mask_path}")
            if overlay_path is not None:
                print(f"Saved overlay: {overlay_path}")


if __name__ == "__main__":
    main()
   