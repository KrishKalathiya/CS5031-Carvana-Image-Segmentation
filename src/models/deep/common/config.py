"""Shared configuration defaults for the Carvana segmentation project.

These defaults match the U-Net training script that was tested in Kaggle.
Override them from command-line arguments in train/evaluate/submission scripts.
"""

from dataclasses import dataclass
from pathlib import Path


IMAGE_EXTENSIONS = frozenset({".jpg", ".jpeg", ".png", ".bmp"})
MASK_EXTENSIONS = frozenset({".gif", ".png", ".jpg", ".jpeg", ".bmp"})


@dataclass(frozen=True)
class ProjectConfig:
    # Dataset paths. Change these to match your local/Kaggle folder layout.
    train_images: Path = Path("data/train")
    train_masks: Path = Path("data/train_masks")
    test_images: Path = Path("data/test")

    # Training defaults from the existing U-Net train.py.
    image_size: int = 256
    epochs: int = 20
    batch_size: int = 8
    learning_rate: float = 1e-3
    validation_fraction: float = 0.20
    split_mode: str = "vehicle"  # "vehicle" or "random"
    seed: int = 42
    num_workers: int = 0

    # Segmentation/submission defaults.
    threshold: float = 0.50
    output_dir: Path = Path("outputs")

    checkpoint_name: str = "best_unet_vehicle.pth"

    model_features: tuple = (32, 64, 128)
    checkpoint_name: str = "best_unet_vehicle.pth"




DEFAULT_CONFIG = ProjectConfig()
