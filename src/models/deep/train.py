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
from PIL import Image

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
        example = ", ".join(missing[:5])
        raise FileNotFoundError(
            f"Could not find masks for {len(missing)} images. "
            f"Example image IDs: {example}. Expected mask names like "
            f"'<image_id>_mask.gif' in {mask_dir}."
        )

    return pairs

class CarvanaDataset(Dataset):
    def __init__(self, pairs, image_size=256):
        self.pairs = pairs
        self.image_size = image_size

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        image_path, mask_path, image_id = self.pairs[index]

        # RGB image -> float tensor [3, H, W], scaled to [0, 1].
        image = Image.open(image_path).convert("RGB")
        image = image.resize(
            (self.image_size, self.image_size),
            resample=Image.Resampling.BILINEAR
        )
        image = np.asarray(image, dtype=np.float32) / 255.0
        image = torch.from_numpy(image).permute(2, 0, 1).contiguous()

        # Masks are categorical labels. Use nearest-neighbour resizing.
        # Carvana masks are commonly GIF files with foreground near 255.
        mask = Image.open(mask_path)
        try:
            mask.seek(0)  # Use first frame if the GIF has multiple frames.
        except EOFError:
            pass
        mask = mask.convert("L")
        mask = mask.resize(
            (self.image_size, self.image_size),
            resample=Image.Resampling.NEAREST
        )
        mask = np.asarray(mask, dtype=np.uint8)
        mask = (mask >= 128).astype(np.float32)
        mask = torch.from_numpy(mask).unsqueeze(0)

        return image, mask, image_id

def split_pairs(pairs, validation_fraction, split_mode, seed):
    """Create either a naive image-level split or a vehicle-group split."""
    rng = random.Random(seed)
    pairs = list(pairs)

    if split_mode == "random":
        rng.shuffle(pairs)
        n_val = max(1, int(round(len(pairs) * validation_fraction)))
        val_pairs = pairs[:n_val]
        train_pairs = pairs[n_val:]

    else:
        groups = {}
        for pair in pairs:
            vehicle_id = vehicle_id_from_image_id(pair[2])
            groups.setdefault(vehicle_id, []).append(pair)

        vehicle_ids = list(groups.keys())
        rng.shuffle(vehicle_ids)

        target_val_images = max(1, int(round(len(pairs) * validation_fraction)))
        val_pairs = []
        train_pairs = []

        for vehicle_id in vehicle_ids:
            group = groups[vehicle_id]
            if len(val_pairs) < target_val_images:
                val_pairs.extend(group)
            else:
                train_pairs.extend(group)

        # Avoid an empty training or validation set on very small datasets.
        if not train_pairs and len(vehicle_ids) > 1:
            moved_vehicle = vehicle_ids[-1]
            moved = groups[moved_vehicle]
            val_ids = {p[2] for p in val_pairs}
            val_pairs = [p for p in val_pairs if p[2] not in {x[2] for x in moved}]
            train_pairs.extend(moved)

    if not train_pairs or not val_pairs:
        raise ValueError(
            "The split produced an empty training or validation set. "
            "Check the dataset size and --val-fraction."
        )

    return train_pairs, val_pairs

class BCEDiceLoss(nn.Module):
    def __init__(self, bce_weight=0.5, smooth=1.0):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.bce_weight = bce_weight
        self.smooth = smooth

    def forward(self, logits, targets):
        bce_loss = self.bce(logits, targets)

        probabilities = torch.sigmoid(logits)
        probabilities = probabilities.flatten(start_dim=1)
        targets = targets.flatten(start_dim=1)

        intersection = (probabilities * targets).sum(dim=1)
        denominator = probabilities.sum(dim=1) + targets.sum(dim=1)
        dice = (2.0 * intersection + self.smooth) / (
            denominator + self.smooth
        )
        dice_loss = 1.0 - dice.mean()

        return self.bce_weight * bce_loss + (1.0 - self.bce_weight) * dice_loss

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total_loss = 0.0
    total_dice = 0.0
    total_iou = 0.0
    batches = 0
    criterion = BCEDiceLoss()

    for images, masks, _ in loader:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        logits = model(images)
        loss = criterion(logits, masks)
        predictions = (torch.sigmoid(logits) >= 0.5).float()

        dims = (1, 2, 3)
        intersection = (predictions * masks).sum(dim=dims)
        pred_sum = predictions.sum(dim=dims)
        mask_sum = masks.sum(dim=dims)
        union = pred_sum + mask_sum - intersection

        dice = (2.0 * intersection + 1e-7) / (pred_sum + mask_sum + 1e-7)
        iou = (intersection + 1e-7) / (union + 1e-7)

        total_loss += loss.item()
        total_dice += dice.mean().item()
        total_iou += iou.mean().item()
        batches += 1

        return(
            total_loss / max(batches, 1),
            total_dice / max(batches, 1),
            total_iou / max(batches, 1 ),
        )

def main():
    parser = argparse.ArgumentParser(description="Train Carvana U-Net.")
    parser.add_argument("--images", type=Path, default=Path("data/train"))
    parser.add_argument("--masks", type=Path, default=Path("data/train_masks"))
    parser.add_argument("--output", type=Path, default=Path("outputs"))
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument(
        "--split-mode", choices=("vehicle", "random"), default="vehicle",
        help="vehicle = principled Vehicle ID split; random = naive image split"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    args = parser.parse_args()

    if args.image_size % 8 != 0:
        raise ValueError("--image-size must be divisible by 8 (e.g. 256).")

    set_seed(args.seed)
    args.output.mkdir(parents=True, exist_ok=True)

    pairs = find_image_mask_pairs(args.images, args.masks)
    train_pairs, val_pairs = split_pairs(
        pairs, args.val_fraction, args.split_mode, args.seed
    )

    print(f"Total paired images: {len(pairs)}")
    print(f"Training images:     {len(train_pairs)}")
    print(f"Validation images:   {len(val_pairs)}")
    print(f"Split mode:          {args.split_mode}")

    if args.split_mode == "vehicle":
        train_vehicles = {vehicle_id_from_image_id(p[2]) for p in train_pairs}
        val_vehicles = {vehicle_id_from_image_id(p[2]) for p in val_pairs}
        overlap = train_vehicles.intersection(val_vehicles)
        print(f"Vehicle ID overlap:  {len(overlap)} (should be 0)")
        if overlap:
            raise RuntimeError("Vehicle leakage detected in vehicle split.")

    # Save the exact split for reproducibility.
    split_file = args.output / f"{args.split_mode}_split_seed{args.seed}.csv"
    with split_file.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["image_id", "vehicle_id", "split", "image_path", "mask_path"])
        for split_name, split_data in (("train", train_pairs), ("validation", val_pairs)):
            for image_path, mask_path, image_id in split_data:
                writer.writerow([
                    image_id, vehicle_id_from_image_id(image_id), split_name,
                    str(image_path), str(mask_path)
                ])

    train_dataset = CarvanaDataset(train_pairs, args.image_size)
    val_dataset = CarvanaDataset(val_pairs, args.image_size)

    pin_memory = torch.cuda.is_available()
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=pin_memory
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=pin_memory
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = UNet().to(device)
    criterion = BCEDiceLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    best_val_dice = -1.0
    history_path = args.output / f"history_{args.split_mode}.csv"

    with history_path.open("w", newline="", encoding="utf-8") as history_file:
        writer = csv.writer(history_file)
        writer.writerow([
            "epoch", "train_loss", "val_loss", "val_dice", "val_iou"
        ])

        for epoch in range(1, args.epochs + 1):
            model.train()
            running_loss = 0.0

            for batch_index, (images, masks, _) in enumerate(train_loader, start=1):
                images = images.to(device, non_blocking=True)
                masks = masks.to(device, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)
                logits = model(images)
                loss = criterion(logits, masks)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

                if batch_index % 20 == 0:
                    print(
                        f"Epoch {epoch}/{args.epochs} | "
                        f"Batch {batch_index}/{len(train_loader)} | "
                        f"Loss {loss.item():.4f}"
                    )

            train_loss = running_loss / max(len(train_loader), 1)
            val_loss, val_dice, val_iou = evaluate(model, val_loader, device)

            print(
                f"Epoch {epoch:02d}/{args.epochs} | "
                f"train loss: {train_loss:.4f} | "
                f"val loss: {val_loss:.4f} | "
                f"Dice: {val_dice:.4f} | IoU: {val_iou:.4f}"
            )

            writer.writerow([epoch, train_loss, val_loss, val_dice, val_iou])
            history_file.flush()

            if val_dice > best_val_dice:
                best_val_dice = val_dice
                checkpoint = {
                    "model_state_dict": model.state_dict(),
                    "image_size": args.image_size,
                    "epoch": epoch,
                    "val_dice": val_dice,
                    "val_iou": val_iou,
                    "split_mode": args.split_mode,
                    "seed": args.seed,
                }
                checkpoint_path = args.output / f"best_unet_{args.split_mode}.pth"
                torch.save(checkpoint, checkpoint_path)
                print(f"Saved best checkpoint: {checkpoint_path}")

    print(f"Training complete. Best validation Dice: {best_val_dice:.4f}")
    print(f"Split manifest: {split_file}")
    print(f"Training history: {history_path}")


if __name__ == "__main__":
    main()
