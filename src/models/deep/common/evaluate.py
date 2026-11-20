"""Evaluate a trained Carvana segmentation model on labelled image/mask pairs.

Example from the repository root:
    python evaluate.py \
        --model-file src/deep_learning/U_net/model.py --model-class UNet \
        --checkpoint outputs/best_unet_vehicle.pth \
        --images data/train --masks data/train_masks

The metric calculation averages Dice and IoU per image, matching the usual
Carvana competition-style aggregation rather than pooling all pixels globally.
"""

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

try:
    from .config import DEFAULT_CONFIG
    from .dataset import CarvanaDataset, find_image_mask_pairs
    from .metrics import BCEDiceLoss, dice_iou_per_image
    from .utils import extract_logits, get_device, import_model_class, load_checkpoint, load_model_weights
except ImportError:
    from config import DEFAULT_CONFIG
    from dataset import CarvanaDataset, find_image_mask_pairs
    from metrics import BCEDiceLoss, dice_iou_per_image
    from utils import extract_logits, get_device, import_model_class, load_checkpoint, load_model_weights


@torch.no_grad()
def evaluate_model(model: torch.nn.Module,
                   loader: DataLoader,
                   device: str | torch.device | None = None,
                   threshold: float = 0.5) -> dict[str, float | int]:
    """Evaluate a model and return image-averaged loss, Dice, and IoU."""
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be between 0 and 1")

    device = get_device(str(device) if device is not None else None)
    model = model.to(device)
    model.eval()

    criterion = BCEDiceLoss()
    total_loss = 0.0
    total_dice = 0.0
    total_iou = 0.0
    image_count = 0

    for images, masks, _image_ids in loader:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        logits = extract_logits(model(images))
        if logits.shape != masks.shape:
            raise ValueError(
                f"Model output shape {tuple(logits.shape)} does not match "
                f"mask shape {tuple(masks.shape)}"
            )

        batch_size = int(images.shape[0])
        loss = criterion(logits, masks)
        dice, iou = dice_iou_per_image(
            logits, masks, threshold=threshold, from_logits=True
        )

        total_loss += float(loss.item()) * batch_size
        total_dice += float(dice.sum().item())
        total_iou += float(iou.sum().item())
        image_count += batch_size

    if image_count == 0:
        raise ValueError("The evaluation DataLoader is empty")

    return {
        "images": image_count,
        "loss": total_loss / image_count,
        "dice": total_dice / image_count,
        "iou": total_iou / image_count,
        "threshold": float(threshold),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a Carvana segmentation checkpoint")
    parser.add_argument("--model-file", type=Path, required=True,
                        help="Path to the architecture's model.py")
    parser.add_argument("--model-class", type=str, required=True,
                        help="Class name, e.g. UNet or LinkNet")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--images", type=Path, default=DEFAULT_CONFIG.train_images)
    parser.add_argument("--masks", type=Path, default=DEFAULT_CONFIG.train_masks)
    parser.add_argument("--image-size", type=int, default=None,
                        help="Defaults to the image_size saved in the checkpoint")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_CONFIG.batch_size)
    parser.add_argument("--num-workers", type=int, default=DEFAULT_CONFIG.num_workers)
    parser.add_argument("--threshold", type=float, default=DEFAULT_CONFIG.threshold)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--model-kwargs", type=str, default="{}",
                        help="Optional JSON kwargs passed to the model constructor")
    parser.add_argument("--output-json", type=Path, default=None,
                        help="Optional path for saving the metric summary")
    args = parser.parse_args()

    if args.image_size is not None and args.image_size <= 0:
        parser.error("--image-size must be positive")
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")
    if args.num_workers < 0:
        parser.error("--num-workers cannot be negative")

    device = get_device(args.device)
    checkpoint_preview = load_checkpoint(args.checkpoint, map_location="cpu")

    # Verify that the selected model matches the checkpoint architecture.
    saved_model_class = checkpoint_preview.get("model_class")
    if saved_model_class is not None and saved_model_class != args.model_class:
        parser.error(
            f"Checkpoint was trained with {saved_model_class}, "
            f"but you selected {args.model_class}."
        )

    # Use the specified image size or the size saved in the checkpoint.
    image_size = (
        args.image_size
        if args.image_size is not None
        else int(checkpoint_preview.get("image_size", DEFAULT_CONFIG.image_size))
    )

    try:
        model_kwargs = json.loads(args.model_kwargs)
    except json.JSONDecodeError as exc:
        parser.error(f"--model-kwargs must be valid JSON: {exc}")
    if not isinstance(model_kwargs, dict):
        parser.error("--model-kwargs must decode to a JSON object")

    pairs = find_image_mask_pairs(args.images, args.masks)
    dataset = CarvanaDataset(pairs, image_size=image_size)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    model_class = import_model_class(args.model_file, args.model_class)
    model = model_class(**model_kwargs)
    checkpoint = load_model_weights(model, args.checkpoint, device=device)

    results = evaluate_model(model, loader, device=device, threshold=args.threshold)
    results["model"] = args.model_class
    results["checkpoint"] = str(args.checkpoint)
    results["image_size"] = image_size
    results["checkpoint_epoch"] = int(checkpoint.get("epoch", -1))

    print(f"Model: {results['model']}")
    print(f"Images evaluated: {results['images']}")
    print(f"Image size: {results['image_size']} x {results['image_size']}")
    print(f"Loss: {results['loss']:.6f}")
    print(f"Dice: {results['dice']:.6f}")
    print(f"IoU:  {results['iou']:.6f}")

    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"Saved metrics: {args.output_json}")


if __name__ == "__main__":
    main()
