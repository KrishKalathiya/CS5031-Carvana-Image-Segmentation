
"""
train.py
--------
Train either a U-Net or LinkNet model on the Kaggle Carvana dataset.

Expected structure:

deep_learning/
    train.py
    predict.py
    common/
        config.py
        dataset.py
        metrics.py
        utils.py
    U_net/
        model.py
    Link_net/
        model.py

Run from the deep_learning directory:

    python train.py --model-dir U_net --model-class UNet
    python train.py --model-dir Link_net --model-class LinkNet
"""

import argparse
import csv
import importlib.util
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

# ---------------------------------------------------------
# Make the deep_learning directory available for imports
# ---------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

# ---------------------------------------------------------
# Import shared project files
# ---------------------------------------------------------

from common.config import DEFAULT_CONFIG
from common.dataset import (
    CarvanaDataset,
    find_image_mask_pairs,
    split_pairs,
    vehicle_id_from_image_id,
)
from common.metrics import BCEDiceLoss, dice_iou_per_image
from common.utils import set_seed, get_device


# ---------------------------------------------------------
# Load the selected model architecture
# ---------------------------------------------------------

def load_model(model_dir, model_class_name):
    """
    Dynamically load the selected model from either U_net or Link_net.
    """

    model_path = BASE_DIR / model_dir / "model.py"

    if not model_path.is_file():
        raise FileNotFoundError(
            f"Could not find model file: {model_path}"
        )

    spec = importlib.util.spec_from_file_location(
        f"{model_dir}_model",
        model_path
    )

    if spec is None or spec.loader is None:
        raise ImportError(
            f"Could not load model from: {model_path}"
        )

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    if not hasattr(module, model_class_name):
        raise AttributeError(
            f"Model class '{model_class_name}' was not found "
            f"in {model_path}."
        )

    model_class = getattr(module, model_class_name)

    return model_class()


# ---------------------------------------------------------
# Evaluate model on validation data
# ---------------------------------------------------------

@torch.no_grad()
def evaluate(model, loader, device, criterion):
    model.eval()

    total_loss = 0.0
    total_dice = 0.0
    total_iou = 0.0
    total_images = 0

    for images, masks, _ in loader:

        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        logits = model(images)
        loss = criterion(logits, masks)

        dice, iou = dice_iou_per_image(
            logits,
            masks,
            threshold=0.5,
            from_logits=True
        )

        batch_size = images.size(0)

        total_loss += loss.item() * batch_size
        total_dice += dice.sum().item()
        total_iou += iou.sum().item()
        total_images += batch_size

    if total_images == 0:
        raise ValueError("Validation dataset is empty.")

    return (
        total_loss / total_images,
        total_dice / total_images,
        total_iou / total_images
    )


# ---------------------------------------------------------
# Main training function
# ---------------------------------------------------------

def main():

    parser = argparse.ArgumentParser(
        description="Train a Carvana segmentation model."
    )

    # Model selection
    parser.add_argument(
        "--model-dir",
        choices=("U_net", "Link_net"),
        default="U_net"
    )

    parser.add_argument(
        "--model-class",
        type=str,
        default="UNet"
    )

    # Dataset paths
    parser.add_argument(
        "--images",
        type=Path,
        default=DEFAULT_CONFIG.train_images
    )

    parser.add_argument(
        "--masks",
        type=Path,
        default=DEFAULT_CONFIG.train_masks
    )

    # Training settings
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_CONFIG.output_dir
    )

    parser.add_argument(
        "--image-size",
        type=int,
        default=DEFAULT_CONFIG.image_size
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=DEFAULT_CONFIG.epochs
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_CONFIG.batch_size
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=DEFAULT_CONFIG.learning_rate
    )

    parser.add_argument(
        "--val-fraction",
        type=float,
        default=DEFAULT_CONFIG.validation_fraction
    )

    parser.add_argument(
        "--split-mode",
        choices=("vehicle", "random"),
        default=DEFAULT_CONFIG.split_mode,
        help="vehicle = split by vehicle ID; random = split by image"
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_CONFIG.seed
    )

    parser.add_argument(
        "--num-workers",
        type=int,
        default=DEFAULT_CONFIG.num_workers
    )

    args = parser.parse_args()

    # -----------------------------------------------------
    # Validate settings
    # -----------------------------------------------------

    if args.image_size % 8 != 0:
        raise ValueError(
            "--image-size must be divisible by 8 (for example, 256)."
        )

    if not 0 < args.val_fraction < 1:
        raise ValueError("--val-fraction must be between 0 and 1.")

    if args.epochs < 1 or args.batch_size < 1:
        raise ValueError("Epochs and batch size must be positive.")

    # -----------------------------------------------------
    # Setup
    # -----------------------------------------------------

    set_seed(args.seed)

    args.output.mkdir(parents=True, exist_ok=True)

    device = get_device()

    print("=" * 55)
    print("CARVANA IMAGE SEGMENTATION TRAINING")
    print("=" * 55)

    print(f"Model:          {args.model_dir}")
    print(f"Model class:    {args.model_class}")
    print(f"Device:         {device}")
    print(f"Image size:     {args.image_size}")
    print(f"Epochs:         {args.epochs}")
    print(f"Batch size:     {args.batch_size}")
    print(f"Learning rate:  {args.learning_rate}")
    print(f"Split mode:     {args.split_mode}")

    # -----------------------------------------------------
    # Load and split dataset
    # -----------------------------------------------------

    pairs = find_image_mask_pairs(
        args.images,
        args.masks
    )

    train_pairs, val_pairs = split_pairs(
        pairs,
        args.val_fraction,
        args.split_mode,
        args.seed
    )

    print("\nDATASET")
    print("-" * 55)

    print(f"Total paired images: {len(pairs)}")
    print(f"Training images:     {len(train_pairs)}")
    print(f"Validation images:   {len(val_pairs)}")

    # Check for vehicle leakage
    if args.split_mode == "vehicle":

        train_vehicles = {
            vehicle_id_from_image_id(pair[2])
            for pair in train_pairs
        }

        val_vehicles = {
            vehicle_id_from_image_id(pair[2])
            for pair in val_pairs
        }

        overlap = train_vehicles.intersection(val_vehicles)

        print(f"Vehicle ID overlap:  {len(overlap)}")

        if overlap:
            raise RuntimeError(
                "Vehicle leakage detected between training "
                "and validation sets."
            )

    # -----------------------------------------------------
    # Save train/validation split manifest
    # -----------------------------------------------------

    split_file = args.output / (
        f"{args.model_dir}_{args.split_mode}_split_seed{args.seed}.csv"
    )

    with split_file.open(
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.writer(file)

        writer.writerow([
            "image_id",
            "vehicle_id",
            "split",
            "image_path",
            "mask_path"
        ])

        for split_name, split_data in (
            ("train", train_pairs),
            ("validation", val_pairs)
        ):

            for image_path, mask_path, image_id in split_data:

                writer.writerow([
                    image_id,
                    vehicle_id_from_image_id(image_id),
                    split_name,
                    str(image_path),
                    str(mask_path)
                ])

    # -----------------------------------------------------
    # Create datasets and dataloaders
    # -----------------------------------------------------

    train_dataset = CarvanaDataset(
        train_pairs,
        image_size=args.image_size
    )

    val_dataset = CarvanaDataset(
        val_pairs,
        image_size=args.image_size
    )

    pin_memory = torch.cuda.is_available()

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=pin_memory
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory
    )

    # -----------------------------------------------------
    # Initialize selected model
    # -----------------------------------------------------

    model = load_model(args.model_dir, args.model_class).to(device)

    criterion = BCEDiceLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    print("\nMODEL")
    print("-" * 55)
    print(f"Loaded {args.model_class} successfully.")

    # -----------------------------------------------------
    # Training output files
    # -----------------------------------------------------

    history_path = args.output / (f"history_{args.model_dir}_{args.split_mode}.csv")

    model_filename = ("unet" if args.model_dir == "U_net" else "linknet")

    checkpoint_path = args.output / (f"best_{model_filename}_{args.split_mode}.pth")

    best_val_dice = -1.0

    # -----------------------------------------------------
    # Training loop
    # -----------------------------------------------------

    with history_path.open("w", newline="", encoding="utf-8"
    )   as history_file:

        writer = csv.writer(history_file)

        writer.writerow([
            "epoch",
            "train_loss",
            "val_loss",
            "val_dice",
            "val_iou"
        ])

        for epoch in range(1, args.epochs + 1):

            model.train()

            running_loss = 0.0
            total_train_images = 0

            for batch_index, (images, masks, _) in enumerate(
                train_loader,
                start=1
            ):

                images = images.to(
                    device,
                    non_blocking=True
                )

                masks = masks.to(
                    device,
                    non_blocking=True
                )

                optimizer.zero_grad(set_to_none=True)

                logits = model(images)

                loss = criterion(logits, masks)

                loss.backward()

                optimizer.step()

                batch_size = images.size(0)

                running_loss += loss.item() * batch_size
                total_train_images += batch_size

                if batch_index % 20 == 0:

                    print(
                        f"Epoch {epoch}/{args.epochs} | "
                        f"Batch {batch_index}/{len(train_loader)} | "
                        f"Loss {loss.item():.4f}"
                    )

            # -------------------------------------------------
            # End of epoch: evaluate validation performance
            # -------------------------------------------------

            train_loss = (
                running_loss / max(total_train_images, 1)
            )

            val_loss, val_dice, val_iou = evaluate(
                model,
                val_loader,
                device,
                criterion
            )

            print(
                f"\nEpoch {epoch:02d}/{args.epochs} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Val Loss: {val_loss:.4f} | "
                f"Dice: {val_dice:.4f} | "
                f"IoU: {val_iou:.4f}"
            )

            writer.writerow([
                epoch,
                train_loss,
                val_loss,
                val_dice,
                val_iou
            ])

            history_file.flush()

            # -------------------------------------------------
            # Save best model checkpoint
            # -------------------------------------------------

            if val_dice > best_val_dice:

                best_val_dice = val_dice

                checkpoint = {
                    "model_state_dict": model.state_dict(),
                    "model_dir": args.model_dir,
                    "model_class": args.model_class,
                    "image_size": args.image_size,
                    "epoch": epoch,
                    "val_dice": val_dice,
                    "val_iou": val_iou,
                    "split_mode": args.split_mode,
                    "seed": args.seed,
                }

                torch.save(
                    checkpoint,
                    checkpoint_path
                )

                print(
                    f"Saved best checkpoint: {checkpoint_path}"
                )

    # -----------------------------------------------------
    # Training complete
    # -----------------------------------------------------

    print("\n" + "=" * 55)
    print("TRAINING COMPLETE")
    print("=" * 55)

    print(f"Model:                {args.model_dir}")
    print(f"Best validation Dice: {best_val_dice:.4f}")
    print(f"Best checkpoint:      {checkpoint_path}")
    print(f"Training history:     {history_path}")
    print(f"Split manifest:       {split_file}")


if __name__ == "__main__":
    main()
