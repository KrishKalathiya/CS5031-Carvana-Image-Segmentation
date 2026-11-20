from pathlib import Path

import numpy as np
import pandas as pd

from src.data.carvana import load_manifest, load_pair
from src.models.classical.features import extract_region_features
from src.models.classical.slic import slic_regions

# To run: python -m scripts.build_training_table


# ----------------------------------------
# Configuration
# ----------------------------------------

ROOT = Path(__file__).resolve().parents[1]

TRAIN_CSV = ROOT / "data" / "splits" / "train.csv"
TABLES_DIR = ROOT / "outputs" / "tables"

IMAGE_SIZE = (768, 512)
N_SEGMENTS = 800
COMPACTNESS = 10
SIGMA = 1
MAX_IMAGES = None  # None = all training images; use 4 for a quick check


# ----------------------------------------
# Create labels from a training mask
# ----------------------------------------

def create_region_labels(regions, true_mask):
    """
    Calculate q(S) and label each region using q(S) >= 0.5.
    Mixed regions are kept; exact 50/50 ties receive foreground.
    """
    region_ids, areas = np.unique(
        regions, return_counts=True
    )

    foreground_pixels = np.bincount(
        regions.ravel(), weights=true_mask.ravel()
    )[region_ids]

    fractions = foreground_pixels / areas

    return pd.DataFrame({
        "foreground_fraction": fractions,
        "label": (fractions >= 0.5).astype(np.uint8)
    }, index=region_ids)


# ----------------------------------------
# Build the training table
# ----------------------------------------

def build_training_table():
    train_df = load_manifest(TRAIN_CSV, ROOT)

    if MAX_IMAGES is not None:
        train_df = train_df.head(MAX_IMAGES)

    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    table_path = TABLES_DIR / "slic_train.csv"
    metadata_path = TABLES_DIR / "slic_train_metadata.csv"

    total_regions = 0
    mixed_regions = 0

    print(f"\nBuilding rows from {len(train_df)} training images...")

    for index, (_, record) in enumerate(train_df.iterrows()):
        image, true_mask = load_pair(
            record["image"],
            record["mask"],
            size=IMAGE_SIZE
        )

        regions = slic_regions(
            image,
            n_segments=N_SEGMENTS,
            compactness=COMPACTNESS,
            sigma=SIGMA
        )

        features = extract_region_features(image, regions)
        labels = create_region_labels(regions, true_mask)
        labels = labels.loc[features.index]

        # Use FEATURE_COLUMNS for model inputs; label is the training target.
        training_rows = features.copy()
        training_rows["label"] = labels["label"]

        metadata = pd.DataFrame({
            "vehicle_id": record["vehicle_id"],
            "image_id": Path(record["image"]).stem,
            "region_id": features.index.to_numpy(),
            "foreground_fraction": labels["foreground_fraction"].to_numpy()
        })

        # Write one image at a time to avoid holding the full table in memory.
        # Both files have matching row order; the first image starts a new run.
        mode = "w" if index == 0 else "a"

        training_rows.to_csv(
            table_path, mode=mode, header=(index == 0),
            index=False, float_format="%.9g"
        )
        metadata.to_csv(
            metadata_path, mode=mode, header=(index == 0),
            index=False, float_format="%.10g"
        )

        fractions = labels["foreground_fraction"]
        total_regions += len(features)
        mixed_regions += int(((fractions > 0) & (fractions < 1)).sum())

        if (index + 1) % 50 == 0:
            print(f"Processed {index + 1} images")

    print("\n========== TRAINING TABLE ==========")
    print(f"Images: {len(train_df)}")
    print(f"Regions: {total_regions}")
    print(f"Mixed regions kept: {mixed_regions}")
    print(f"Saved training table: {table_path}")
    print(f"Saved metadata: {metadata_path}")


if __name__ == "__main__":
    build_training_table()
