"""
predict.py
----------
Run a trained U-Net or LinkNet on one image or a folder of images.

Examples (run from the deep_learning directory):

    python predict.py --image data/train/example.jpg

    python predict.py --input-dir data/sample_images

    python predict.py --image data/train/example.jpg --model-dir U_net

    python predict.py --image data/train/example.jpg --model-dir Link_net

    python predict.py --input-dir data/sample_images --model-dir U_net --save-overlay

Outputs:
    A binary PNG mask for each input image.
    Optionally, an overlay showing the predicted mask on the original image.
"""

import argparse
import sys
from pathlib import Path
import importlib.util

import numpy as np
from PIL import Image
import torch
from common.dataset import preprocess_image

# ---------------------------------------------------------
# Make deep_learning available for imports
# ---------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

# ---------------------------------------------------------
# Import shared utilities
# ---------------------------------------------------------

from common.utils import (
    get_device,
    load_checkpoint,
    import_model_class,
    extract_logits,
)


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


# ---------------------------------------------------------
# Save predicted mask and optional overlay
# ---------------------------------------------------------

def save_prediction(
    probabilities,
    original_image_path,
    output_dir,
    original_size,
    threshold,
    save_overlay
):

    output_dir.mkdir(parents=True, exist_ok=True)

    image_id = original_image_path.stem

    # Convert probabilities into a binary mask
    mask_array = (
        (probabilities >= threshold).astype(np.uint8) * 255
    )

    mask_image = Image.fromarray(mask_array, mode="L")

    # Resize mask back to original image dimensions
    mask_image = mask_image.resize(
        original_size,
        resample=Image.Resampling.NEAREST
    )

    mask_path = output_dir / f"{image_id}_predicted_mask.png"

    mask_image.save(mask_path)

    overlay_path = None

    # -----------------------------------------------------
    # Create overlay if requested
    # -----------------------------------------------------

    if save_overlay:

        original = Image.open(
            original_image_path
        ).convert("RGB")

        overlay = np.asarray(
            original,
            dtype=np.uint8
        ).copy()

        mask_full = (
            np.asarray(mask_image, dtype=np.uint8) > 0
        )

        # Tint predicted foreground red
        overlay[mask_full, 0] = np.clip(
            overlay[mask_full, 0].astype(np.float32) * 0.5 + 127,
            0,
            255
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


# ---------------------------------------------------------
# Main prediction function
# ---------------------------------------------------------

def main():

    parser = argparse.ArgumentParser(
        description="Predict Carvana segmentation masks."
    )

    # Input selection
    source = parser.add_mutually_exclusive_group(required=True)

    source.add_argument(
        "--image",
        type=Path,
        help="Path to one image."
    )

    source.add_argument(
        "--input-dir",
        type=Path,
        help="Folder containing images."
    )

    # Model selection
    parser.add_argument(
        "--model-dir",
        choices=("U_net", "Link_net"),
        default=None,
        help="Model architecture folder. Can be inferred from the checkpoint."
    )

    parser.add_argument(
        "--model-class",
        type=str,
        default=None,
        help="Model class name. Can be inferred from the checkpoint."
    )

    # Checkpoint and output
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Path to the trained model checkpoint."
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path("predictions")
    )

    # Prediction settings
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5
    )

    parser.add_argument(
        "--save-overlay",
        action="store_true"
    )

    args = parser.parse_args()

    # -----------------------------------------------------
    # Validate threshold
    # -----------------------------------------------------

    if not 0.0 <= args.threshold <= 1.0:
        raise ValueError(
            "--threshold must be between 0 and 1."
        )

    # -----------------------------------------------------
    # Determine checkpoint path
    # -----------------------------------------------------

    if args.checkpoint is None:

        selected_model = args.model_dir or "U_net"

        model_filename = (
            "unet" if selected_model == "U_net" else "linknet"
        )

        args.checkpoint = Path(
            f"outputs/best_{model_filename}_vehicle.pth"
        )

    if not args.checkpoint.is_file():
        raise FileNotFoundError(
            f"Checkpoint not found: {args.checkpoint}"
        )

    # -----------------------------------------------------
    # Load checkpoint metadata
    # -----------------------------------------------------

    device = get_device()

    checkpoint = load_checkpoint(
        args.checkpoint,
        map_location=device
    )

    # Use checkpoint metadata when available
    model_dir = (
        args.model_dir
        or checkpoint.get("model_dir")
        or "U_net"
    )

    default_class = (
        "UNet" if model_dir == "U_net" else "LinkNet"
    )

    model_class_name = (
        args.model_class
        or checkpoint.get("model_class")
        or default_class
    )

    # Check that the selected model matches the checkpoint
    saved_model_class = checkpoint.get("model_class")

    if saved_model_class is not None and saved_model_class != model_class_name:
        raise ValueError(
            f"Checkpoint was trained with {saved_model_class}, "
            f"but you selected {model_class_name}."
        )

    image_size = int(
        checkpoint.get("image_size", 256)
    )

    # Validate image size
    if image_size <= 0:
        raise ValueError(
            "Checkpoint image_size must be positive."
        )

    # -----------------------------------------------------
    # Load selected model architecture
    # -----------------------------------------------------

    model_path = BASE_DIR / model_dir / "model.py"

    model_class = import_model_class(
        model_path,
        model_class_name
    )

    model = model_class().to(device)

    state_dict = checkpoint.get(
        "model_state_dict",
        checkpoint
    )

    model.load_state_dict(state_dict)

    model.eval()

    # -----------------------------------------------------
    # Collect input images
    # -----------------------------------------------------

    if args.image is not None:

        if not args.image.is_file():
            raise FileNotFoundError(
                f"Image not found: {args.image}"
            )

        image_paths = [args.image]

    else:

        if not args.input_dir.is_dir():
            raise NotADirectoryError(
                f"Input folder not found: {args.input_dir}"
            )

        image_paths = sorted(
            path for path in args.input_dir.iterdir()
            if (
                path.is_file()
                and path.suffix.lower() in IMAGE_EXTENSIONS
            )
        )

    if not image_paths:
        raise FileNotFoundError(
            "No supported images were found."
        )

    # -----------------------------------------------------
    # Prediction information
    # -----------------------------------------------------

    args.output.mkdir(parents=True, exist_ok=True)

    print("=" * 50)
    print("CARVANA IMAGE SEGMENTATION PREDICTION")
    print("=" * 50)

    print(f"Model:       {model_dir}")
    print(f"Class:       {model_class_name}")
    print(f"Device:      {device}")
    print(f"Checkpoint:  {args.checkpoint}")
    print(f"Image size:  {image_size}")
    print(f"Threshold:   {args.threshold}")
    print(f"Images:      {len(image_paths)}")
    print(f"Output:      {args.output}")

    # -----------------------------------------------------
    # Run predictions
    # -----------------------------------------------------

    successful_predictions = 0

    with torch.no_grad():

        for image_path in image_paths:

            if not image_path.is_file():
                print(f"Skipping missing image: {image_path}")
                continue

            tensor, original_size = preprocess_image(
                image_path,
                image_size
            )

            tensor = tensor.to(device)

            # Get model predictions
            model_output = model(tensor)

            logits = extract_logits(model_output)

            probabilities = (
                torch.sigmoid(logits)[0, 0]
                .cpu()
                .numpy()
            )

            # Save mask and optional overlay
            mask_path, overlay_path = save_prediction(
                probabilities=probabilities,
                original_image_path=image_path,
                output_dir=args.output,
                original_size=original_size,
                threshold=args.threshold,
                save_overlay=args.save_overlay
            )

            print(f"\nImage: {image_path.name}")
            print(f"Saved mask: {mask_path}")

            if overlay_path is not None:
                print(f"Saved overlay: {overlay_path}")

            successful_predictions += 1

    # -----------------------------------------------------
    # Finished
    # -----------------------------------------------------

    print("\n" + "=" * 50)
    print("PREDICTION COMPLETE")
    print("=" * 50)

    print(f"Successful predictions: {successful_predictions}")
    print(f"Output folder: {args.output}")


if __name__ == "__main__":
    main()
