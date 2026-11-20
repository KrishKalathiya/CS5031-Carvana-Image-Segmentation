from pathlib import Path
from collections import Counter
import random
import pandas as pd
from PIL import Image

# Step 1: Configuration

SEED = 42

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "carvana-image-masking-challenge"

IMAGE_DIR = DATASET / "train"
MASK_DIR = DATASET / "train_masks"
SPLIT_DIR = ROOT / "data" / "splits"

SPLIT_DIR.mkdir(parents=True, exist_ok=True)



# Step 2: Find and verify image/mask pairs

records = []

image_files = sorted(IMAGE_DIR.glob("*.jpg"))

for image_path in image_files:

    # Example: abc123_01.jpg
    # vehicle_id = abc123, view = 01
    vehicle_id, view = image_path.stem.rsplit("_", 1)

    # Matching mask: abc123_01_mask.gif
    mask_path = MASK_DIR / f"{image_path.stem}_mask.gif"

    if not mask_path.exists():
        raise FileNotFoundError(
            f"Missing mask for {image_path.name}"
        )

    # Verify image/mask dimensions
    with Image.open(image_path) as image:
        with Image.open(mask_path) as mask:
            if image.size != mask.size:
                raise ValueError(
                    f"Size mismatch: {image_path.name}"
                )

    records.append({
        "image": image_path.relative_to(ROOT).as_posix(),
        "mask": mask_path.relative_to(ROOT).as_posix(),
        "vehicle_id": vehicle_id,
        "view": view
    })

df = pd.DataFrame(records)

if df.empty:
    raise ValueError("No training images found!")

mask_count = len(list(MASK_DIR.glob("*_mask.gif")))

assert len(df) == mask_count, (
    "Image and mask counts differ!"
)

print("Total images:", len(df))
print("Total vehicles:", df["vehicle_id"].nunique())
print("Views per vehicle:")
print(Counter(df["vehicle_id"]).most_common(3))



# Step 3: Split by vehicle ID

vehicle_ids = sorted(df["vehicle_id"].unique())

rng = random.Random(SEED)
rng.shuffle(vehicle_ids)

n_vehicles = len(vehicle_ids)

n_val = round(n_vehicles * 0.15)
n_test = round(n_vehicles * 0.15)
n_train = n_vehicles - n_val - n_test

train_ids = set(vehicle_ids[:n_train])
val_ids = set(vehicle_ids[n_train:n_train + n_val])
test_ids = set(vehicle_ids[n_train + n_val:])

# Ensure no vehicle appears in multiple sets
assert train_ids.isdisjoint(val_ids)
assert train_ids.isdisjoint(test_ids)
assert val_ids.isdisjoint(test_ids)

grouped_splits = {
    "train": df[df["vehicle_id"].isin(train_ids)],
    "val": df[df["vehicle_id"].isin(val_ids)],
    "test": df[df["vehicle_id"].isin(test_ids)]
}

print("\nVehicle-level split:")

for name, subset in grouped_splits.items():
    subset.to_csv(SPLIT_DIR / f"{name}.csv", index=False)

    print(
        f"{name}: {len(subset)} images, "
        f"{subset['vehicle_id'].nunique()} vehicles"
    )



# Step 4: Also create a naive image-level split
# This is needed for the leakage comparison

shuffled = df.sample(frac=1, random_state=SEED)

n_images = len(shuffled)
n_val_images = round(n_images * 0.15)
n_test_images = round(n_images * 0.15)
n_train_images = n_images - n_val_images - n_test_images

naive_splits = {
    "train": shuffled.iloc[:n_train_images],
    "val": shuffled.iloc[
        n_train_images:n_train_images + n_val_images
    ],
    "test": shuffled.iloc[n_train_images + n_val_images:]
}

naive_dir = SPLIT_DIR / "naive"
naive_dir.mkdir(exist_ok=True)

for name, subset in naive_splits.items():
    subset.to_csv(naive_dir / f"{name}.csv", index=False)

print("\nAll dataset manifests saved successfully!")
