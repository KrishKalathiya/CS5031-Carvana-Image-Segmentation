"""Generate a Kaggle Carvana submission CSV from a trained segmentation model.

The required CSV columns are exactly:
    img,rle_mask

RLE is 1-indexed and traverses the binary mask in column-major order
(Fortran order), as required by the Carvana competition.
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

try:
    from .config import DEFAULT_CONFIG, IMAGE_EXTENSIONS
    from .dataset import preprocess_image
    from .utils import extract_logits, get_device, import_model_class, load_checkpoint, load_model_weights
except ImportError:
    from config import DEFAULT_CONFIG, IMAGE_EXTENSIONS
    from dataset import preprocess_image
    from utils import extract_logits, get_device, import_model_class, load_checkpoint, load_model_weights


def rle_encode(mask: np.ndarray) -> str:
    """Encode a 2-D binary mask as Carvana/Kaggle run-length encoding.

    Pixels are flattened column-first (``order='F'``), positions are 1-based,
    and the returned string alternates ``start length`` values. An all-zero
    mask is represented by an empty string.
    """
    mask = np.asarray(mask)
    if mask.ndim != 2:
        raise ValueError(f"RLE expects a 2-D mask; got shape {mask.shape}")

    pixels = (mask > 0).astype(np.uint8).flatten(order="F")
    padded = np.concatenate(([0], pixels, [0]))
    changes = np.flatnonzero(padded[1:] != padded[:-1]) + 1
    changes[1::2] -= changes[::2]
    return " ".join(str(value) for value in changes.tolist())


@torch.no_grad()
def predict_binary_mask(model: torch.nn.Module,
                        image_path: Path | str,
                        image_size: int,
                        device: str | torch.device,
                        threshold: float = 0.5) -> np.ndarray:
    """Predict one image and return a boolean mask at its original resolution."""
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be between 0 and 1")

    device = get_device(str(device))
    tensor, original_size = preprocess_image(image_path, image_size)
    tensor = tensor.to(device)

    model.eval()
    logits = extract_logits(model(tensor))
    probabilities = torch.sigmoid(logits)

    original_width, original_height = original_size
    probabilities = F.interpolate(
        probabilities,
        size=(original_height, original_width),
        mode="bilinear",
        align_corners=False,
    )
    return (probabilities[0, 0].cpu().numpy() >= threshold)


def generate_submission(model: torch.nn.Module,
                        image_dir: Path | str,
                        output_csv: Path | str,
                        image_size: int,
                        device: str | torch.device,
                        threshold: float = 0.5) -> Path:
    """Predict every image in ``image_dir`` and write the Kaggle CSV."""
    image_dir = Path(image_dir)
    output_csv = Path(output_csv)

    if not image_dir.is_dir():
        raise NotADirectoryError(f"Test image directory not found: {image_dir}")
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be between 0 and 1")

    image_paths = sorted(
        p for p in image_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not image_paths:
        raise FileNotFoundError(f"No test images found in: {image_dir}")

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    model.eval()

    with output_csv.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=["img", "rle_mask"])
        writer.writeheader()

        for index, image_path in enumerate(image_paths, start=1):
            mask = predict_binary_mask(
                model=model,
                image_path=image_path,
                image_size=image_size,
                device=device,
                threshold=threshold,
            )
            writer.writerow({"img": image_path.stem, "rle_mask": rle_encode(mask)})

            if index % 100 == 0 or index == len(image_paths):
                print(f"Predicted {index}/{len(image_paths)} test images")

    # Verify the file header and row count after writing.
    with output_csv.open("r", newline="", encoding="utf-8") as csv_file:
        reader = csv.reader(csv_file)
        header = next(reader, None)
        if header != ["img", "rle_mask"]:
            raise RuntimeError(f"Unexpected submission header: {header}")
        row_count = sum(1 for _ in reader)
    if row_count != len(image_paths):
        raise RuntimeError(
            f"Submission has {row_count} rows but {len(image_paths)} images were found"
        )

    print(f"Saved submission: {output_csv}")
    print(f"Rows: {row_count}; columns: img, rle_mask")
    return output_csv


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a Carvana Kaggle submission")
    parser.add_argument("--model-file", type=Path, required=True,
                        help="Path to the architecture's model.py")
    parser.add_argument("--model-class", type=str, required=True,
                        help="Class name, e.g. UNet or LinkNet")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--input-dir", type=Path, required=True,
                        help="Folder containing the competition test images")
    parser.add_argument("--output", type=Path, default=Path("submission.csv"))
    parser.add_argument("--image-size", type=int, default=None,
                        help="Defaults to image_size stored in the checkpoint")
    parser.add_argument("--threshold", type=float, default=DEFAULT_CONFIG.threshold)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--model-kwargs", type=str, default="{}",
                        help="Optional JSON kwargs passed to the model constructor")
    args = parser.parse_args()

    device = get_device(args.device)
    checkpoint = load_checkpoint(args.checkpoint, map_location="cpu")
    image_size = args.image_size or int(checkpoint.get("image_size", DEFAULT_CONFIG.image_size))

    try:
        model_kwargs = json.loads(args.model_kwargs)
    except json.JSONDecodeError as exc:
        parser.error(f"--model-kwargs must be valid JSON: {exc}")
    if not isinstance(model_kwargs, dict):
        parser.error("--model-kwargs must decode to a JSON object")

    model_class = import_model_class(args.model_file, args.model_class)
    model = model_class(**model_kwargs)
    load_model_weights(model, args.checkpoint, device=device)

    generate_submission(
        model=model,
        image_dir=args.input_dir,
        output_csv=args.output,
        image_size=image_size,
        device=device,
        threshold=args.threshold,
    )


if __name__ == "__main__":
    main()
