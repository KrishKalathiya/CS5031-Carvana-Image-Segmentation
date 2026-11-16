import json
from pathlib import Path
from time import perf_counter

import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from skimage.segmentation import find_boundaries, mark_boundaries

from src.data.carvana import load_manifest, load_pair
from src.evaluation.superpixels import majority_label_diagnostic
from src.models.classical.slic import slic_regions

# To run this script, go to the project root and run: python -m scripts.eval_slic


# ----------------------------------------
# Configuration
# ----------------------------------------

ROOT = Path(__file__).resolve().parents[1]

TRAIN_CSV = ROOT / "data" / "splits" / "train.csv"
VAL_CSV = ROOT / "data" / "splits" / "val.csv"

OUTPUT_DIR = ROOT / "outputs" / "slic"

IMAGE_SIZE = (768, 512)  # None = original resolution
VAL_SAMPLE_SIZE = 48    # None = full validation set
TRAIN_EXAMPLES = 6
VAL_EXAMPLES = 3
SEED = 42

# (requested regions, compactness)
SETTINGS = [
    (400, 10),
    (800, 10),
    (1200, 10),
    (800, 5),
    (800, 20)
]


# ----------------------------------------
# Select images across vehicles and views
# ----------------------------------------

def diverse_sample(frame, count, seed):
    """
    Round-robin shuffled vehicles and views, without consulting masks.
    Colors must be checked visually because the manifest has no color labels.
    """
    rng = np.random.default_rng(seed)
    vehicles = rng.permutation(sorted(frame.vehicle_id.unique()))
    views = sorted(frame.view.unique())

    groups = {}

    for offset, vehicle in enumerate(vehicles):
        group = frame[frame.vehicle_id == vehicle]
        order = np.roll(views, -(offset % len(views)))
        groups[vehicle] = [group[group.view == view] for view in order]

    rows = []

    for round_index in range(len(views)):
        for vehicle in vehicles:
            rows.extend(
                groups[vehicle][round_index].to_dict("records")
            )

            if len(rows) >= count:
                return pd.DataFrame(rows[:count])

    return pd.DataFrame(rows)


# ----------------------------------------
# Visualize boundaries and majority masks
# ----------------------------------------

def save_panel(image, truth, results, destination, name):
    """
    Save boundary overlays and error maps for mirror/tire inspection.
    """
    fig, axes = plt.subplots(
        2,
        len(results) + 1,
        figsize=(24, 8),
        squeeze=False
    )

    axes[0, 0].imshow(image)
    axes[0, 0].set_title(f"RGB: {name}")

    axes[1, 0].imshow(
        truth,
        cmap="gray",
        vmin=0,
        vmax=1
    )
    axes[1, 0].set_title("Ground truth")

    boundary = find_boundaries(truth, mode="inner")

    for col, (regions, reconstruction, metrics, n, compactness) in enumerate(
        results, 1
    ):
        overlay = mark_boundaries(
            image, regions, color=(1, 1, 0)
        )
        overlay[boundary] = (0, 1, 1)

        axes[0, col].imshow(overlay)
        axes[0, col].set_title(
            f"n={n}, c={compactness}; actual={metrics['actual_regions']}\n"
            f"crossing={metrics['crossing_regions']}"
        )

        error = np.repeat(
            (truth.astype(float) * 0.65)[..., None],
            3,
            axis=2
        )
        error[(truth == 1) & (reconstruction == 0)] = (1, 0, 0)
        error[(truth == 0) & (reconstruction == 1)] = (0, 0.5, 1)

        axes[1, col].imshow(error)
        axes[1, col].set_title(
            f"GT-majority Dice={metrics['dice']:.4f}"
        )

    for ax in axes.ravel():
        ax.axis("off")

    fig.suptitle(
        "Yellow: SLIC | cyan: GT boundary | gray: correct vehicle | "
        "red: missed vehicle | blue: extra foreground\n"
        "Ground-truth majority labels are a diagnostic, not a predictor or a Dice upper bound."
    )

    fig.tight_layout()
    fig.savefig(destination, dpi=160)
    plt.close(fig)


# ----------------------------------------
# Evaluate SLIC boundary preservation
# ----------------------------------------

def evaluate():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    figures = OUTPUT_DIR / "figures"
    figures.mkdir(exist_ok=True)

    train = load_manifest(TRAIN_CSV, ROOT)
    validation = load_manifest(VAL_CSV, ROOT)

    if set(train.vehicle_id) & set(validation.vehicle_id):
        raise ValueError("Training and validation vehicles overlap.")

    selected = diverse_sample(
        validation,
        VAL_SAMPLE_SIZE or len(validation),
        SEED
    )

    training_examples = (
        diverse_sample(train, TRAIN_EXAMPLES, SEED)
        if TRAIN_EXAMPLES else train.iloc[:0]
    )

    rows = []
    selections = []

    for split, records in [("train", training_examples), ("val", selected)]:
        for index, (_, record) in enumerate(records.iterrows()):
            image, truth = load_pair(
                record.image,
                record["mask"],
                size=IMAGE_SIZE
            )

            selections.append({
                "split": split,
                "image": str(record.image),
                "vehicle_id": record.vehicle_id,
                "view": int(record.view)
            })

            panels = []
            visualize = split == "train" or index < VAL_EXAMPLES

            for n_segments, compactness in SETTINGS:
                # Measure SLIC latency only.
                start = perf_counter()

                regions = slic_regions(
                    image,
                    n_segments=n_segments,
                    compactness=compactness
                )

                latency = (perf_counter() - start) * 1000

                reconstructed, metrics = majority_label_diagnostic(
                    regions, truth
                )

                rows.append({
                    "split": split,
                    "image": str(record.image),
                    "vehicle_id": record.vehicle_id,
                    "view": int(record.view),
                    "width": image.shape[1],
                    "height": image.shape[0],
                    "n_segments": n_segments,
                    "compactness": compactness,
                    "sigma": 1,
                    **metrics,
                    "slic_latency_ms": latency
                })

                if visualize:
                    panels.append((
                        regions,
                        reconstructed,
                        metrics,
                        n_segments,
                        compactness
                    ))

            if visualize:
                save_panel(
                    image,
                    truth,
                    panels,
                    figures / f"{split}_{record.image.stem}.png",
                    record.image.stem
                )

            print(
                f"{split}: {index + 1}/{len(records)} {record.image.name}",
                flush=True
            )

    # ----------------------------------------
    # Save results
    # ----------------------------------------

    frame = pd.DataFrame(rows)

    frame.to_csv(
        OUTPUT_DIR / "per_image.csv",
        index=False
    )
    pd.DataFrame(selections).to_csv(
        OUTPUT_DIR / "selected_images.csv",
        index=False
    )

    metric_columns = [
        "dice", "iou", "recall", "precision",
        "actual_regions", "crossing_regions", "crossing_fraction",
        "pixel_error_fraction", "slic_latency_ms"
    ]

    summary = frame.groupby(
        ["split", "n_segments", "compactness"]
    )[metric_columns].mean()

    summary.to_csv(OUTPUT_DIR / "summary.csv")

    metadata = {
        "settings": SETTINGS,
        "image_size": IMAGE_SIZE,
        "seed": SEED,
        "validation_sample_size": VAL_SAMPLE_SIZE,
        "sigma": 1,
        "convert2lab": True,
        "enforce_connectivity": True,
        "start_label": 0,
        "tie_label": 0
    }

    (OUTPUT_DIR / "run.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8"
    )

    print("\n========== SLIC RESULTS ==========")
    print(summary.to_string())
    print("==================================")


if __name__ == "__main__":
    evaluate()
