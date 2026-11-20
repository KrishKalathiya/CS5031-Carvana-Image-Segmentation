from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from time import perf_counter

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
NUM_WORKERS = 8    # Images processed concurrently; use 1 for a sequential run


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
# Process one training image
# ----------------------------------------

def process_image(record):
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

    return training_rows, metadata


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

    records = train_df.to_dict("records")
    total_regions = 0
    mixed_regions = 0
    start = perf_counter()

    print(f"\nBuilding rows from {len(records)} training images...")
    print(f"Using {NUM_WORKERS} image workers")

    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
        # map keeps manifest order even when images finish out of order.
        results = executor.map(process_image, records)

        for index, (training_rows, metadata) in enumerate(results):
            # Only the main thread writes, keeping both CSVs aligned.
            mode = "w" if index == 0 else "a"

            training_rows.to_csv(
                table_path, mode=mode, header=(index == 0),
                index=False, float_format="%.9g"
            )
            metadata.to_csv(
                metadata_path, mode=mode, header=(index == 0),
                index=False, float_format="%.10g"
            )

            fractions = metadata["foreground_fraction"]
            total_regions += len(training_rows)
            mixed_regions += int(((fractions > 0) & (fractions < 1)).sum())

            if (index + 1) % 50 == 0 or index + 1 == len(records):
                elapsed = perf_counter() - start
                rate = (index + 1) / elapsed
                remaining_minutes = (len(records) - index - 1) / rate / 60

                print(
                    f"Processed {index + 1}/{len(records)} images | "
                    f"{rate:.2f} images/s | ETA {remaining_minutes:.1f} min",
                    flush=True
                )

    print("\n========== TRAINING TABLE ==========")
    print(f"Images: {len(train_df)}")
    print(f"Regions: {total_regions}")
    print(f"Mixed regions kept: {mixed_regions}")
    print(f"Saved training table: {table_path}")
    print(f"Saved metadata: {metadata_path}")


if __name__ == "__main__":
    build_training_table()
