from datetime import datetime
from uuid import uuid4

from itertools import zip_longest
from pathlib import Path
from time import perf_counter
import json

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier

from src.models.classical.features import FEATURE_COLUMNS

# To run this script, go to the project root and run: python -m scripts.train_classical


# ----------------------------------------
# Configuration
# ----------------------------------------

ROOT = Path(__file__).resolve().parents[1]

TABLE_PATH = ROOT / "outputs" / "tables" / "slic_train.csv"
METADATA_PATH = ROOT / "outputs" / "tables" / "slic_train_metadata.csv"
TRAIN_CSV = ROOT / "data" / "splits" / "train.csv"

SEED = 42

MAX_TRAINING_VEHICLES = 20
MAX_TRAINING_REGIONS = 100_000

CHUNK_SIZE = 50_000


# ----------------------------------------
# Load training data
# ----------------------------------------

def load_training_data():
    print("Loading training data...")

    # Select vehicles from the official training split.
    manifest = pd.read_csv(TRAIN_CSV)

    train_vehicle_ids = np.array(
        sorted(manifest["vehicle_id"].astype(str).unique())
    )

    rng = np.random.default_rng(SEED)

    selected_ids = rng.choice(
        train_vehicle_ids,
        size=min(MAX_TRAINING_VEHICLES, len(train_vehicle_ids)),
        replace=False
    )

    selected_ids = set(selected_ids)

    print(f"Selected {len(selected_ids)} training vehicles")

    feature_chunks = []
    label_chunks = []
    metadata_chunks = []

    # Read the two CSVs in matching chunks.
    table_reader = pd.read_csv(
        TABLE_PATH,
        chunksize=CHUNK_SIZE
    )

    metadata_reader = pd.read_csv(
        METADATA_PATH,
        chunksize=CHUNK_SIZE,
        dtype={
            "vehicle_id": str,
            "image_id": str
        }
    )

    expected_columns = FEATURE_COLUMNS + ["label"]

    total_rows = 0

    for table_chunk, metadata_chunk in zip_longest(
        table_reader, metadata_reader
    ):
        if table_chunk is None or metadata_chunk is None:
            raise ValueError(
                "Training table and metadata have "
                "different row counts."
            )

        if table_chunk.columns.tolist() != expected_columns:
            raise ValueError(
                "Unexpected training feature schema."
            )

        if len(table_chunk) != len(metadata_chunk):
            raise ValueError(
                "Training table and metadata are misaligned."
            )

        total_rows += len(table_chunk)

        # Keep only the selected vehicles.
        selected = metadata_chunk[
            "vehicle_id"
        ].isin(selected_ids).to_numpy()

        if not selected.any():
            continue

        feature_chunks.append(
            table_chunk.loc[selected, FEATURE_COLUMNS]
        )

        label_chunks.append(
            table_chunk.loc[selected, "label"]
        )

        metadata_chunks.append(
            metadata_chunk.loc[
                selected, ["vehicle_id", "image_id"]
            ]
        )

    print(f"Total table rows scanned: {total_rows:,}")

    if not feature_chunks:
        raise ValueError("No training rows were selected.")

    X_df = pd.concat(feature_chunks, ignore_index=True)
    y_series = pd.concat(label_chunks, ignore_index=True)
    metadata = pd.concat(metadata_chunks, ignore_index=True)

    # Sample regions reproducibly if necessary.
    if len(X_df) > MAX_TRAINING_REGIONS:
        indices = rng.choice(
            len(X_df),
            size=MAX_TRAINING_REGIONS,
            replace=False
        )

        X_df = X_df.iloc[indices]
        y_series = y_series.iloc[indices]
        metadata = metadata.iloc[indices]

    X = X_df.to_numpy(dtype=np.float32)
    y = y_series.to_numpy(dtype=np.uint8)

    if not np.isfinite(X).all():
        raise ValueError("Features contain NaN or infinity.")

    if not np.isin(y_series, [0, 1]).all():
        raise ValueError("Labels must be binary.")

    if len(np.unique(y)) != 2:
        raise ValueError("Both classes must be present.")

    print("\n========== TRAINING DATA ==========")
    print(f"Vehicles: {metadata.vehicle_id.nunique()}")
    print(f"Images: {metadata.image_id.nunique()}")
    print(f"Regions: {len(X):,}")
    print(f"Features: {X.shape[1]}")

    unique, counts = np.unique(y, return_counts=True)

    for label, count in zip(unique, counts):
        print(f"Class {label}: {count:,} regions")

    print("===================================")

    return X, y, sorted(selected_ids)


# ----------------------------------------
# Train Random Forest
# ----------------------------------------

def train():
    X, y, selected_ids = load_training_data()

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=20,
        min_samples_leaf=5,
        max_features="sqrt",
        random_state=SEED,
        n_jobs=-1
    )

    print("\nTraining Random Forest...")

    start = perf_counter()
    model.fit(X, y)
    training_time = perf_counter() - start

    print(f"Training completed in {training_time:.2f} seconds")

    # Training accuracy is only a sanity check.
    train_accuracy = model.score(X, y)

    print(f"Training region accuracy: {train_accuracy:.4f}")

    # ----------------------------------------
    # Create experiment directory
    # ----------------------------------------

    experiment_id = (
        f"rf_{len(selected_ids)}cars_{len(X)//1000}k_"
        f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_"
        f"{uuid4().hex[:6]}"
    )

    experiment_dir = (
        ROOT / "outputs" / "experiments" / experiment_id
    )
    experiment_dir.mkdir(parents=True, exist_ok=False)

    print(f"\nExperiment: {experiment_id}")


    # ----------------------------------------
    # Save model
    # ----------------------------------------

    model_path = experiment_dir / "model.joblib"

    joblib.dump(model, model_path)

    config = {
        "seed": SEED,
        "training_vehicles": selected_ids,
        "training_regions": len(X),
        "image_size": [768, 512],
        "n_segments": 800,
        "compactness": 10,
        "sigma": 1,
        "feature_columns": FEATURE_COLUMNS,
        "n_estimators": 200,
        "max_depth": 20,
        "min_samples_leaf": 5,
        "max_features": "sqrt",
        "training_seconds": training_time,
        "experiment_id": experiment_id,
        "max_training_regions": MAX_TRAINING_REGIONS,
        "sampling_strategy": "random regions from selected vehicles"
    }

    config_path = experiment_dir / "config.json"

    config_path.write_text(
        json.dumps(config, indent=2),
        encoding="utf-8"
    )

    # Verify saved model reproduces predictions.
    reloaded_model = joblib.load(model_path)

    np.testing.assert_array_equal(
        model.predict(X[:100]),
        reloaded_model.predict(X[:100])
    )

    print("\nModel saved:", model_path)
    print("Configuration saved:", config_path)
    print("Reload verification: PASSED")


if __name__ == "__main__":
    train()
