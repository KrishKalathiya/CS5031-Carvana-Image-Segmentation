from pathlib import Path
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.data.carvana import load_manifest, load_pair
from src.evaluation.metrics import binary_segmentation_metrics
from src.models.classical.otsu import otsu_segment

# To run this script, go to the project root and run: python -m scripts.eval_otsu


# ----------------------------------------
# Configuration
# ----------------------------------------

ROOT = Path(__file__).resolve().parents[1]

TRAIN_CSV = ROOT / "data" / "splits" / "train.csv"
VAL_CSV = ROOT / "data" / "splits" / "val.csv"

OUTPUT_DIR = ROOT / "outputs"
METRICS_DIR = OUTPUT_DIR / "metrics"
FIGURES_DIR = OUTPUT_DIR / "figures"

METRICS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

IMAGE_SIZE = (768, 512)
SEED = 42
POLARITY_SAMPLE_SIZE = 48
NUM_VISUALIZATIONS = 3


# ----------------------------------------
# Select polarity using training images
# ----------------------------------------

def choose_polarity():
    train_df = load_manifest(TRAIN_CSV, ROOT)

    samples = train_df.sample(
        n=min(POLARITY_SAMPLE_SIZE, len(train_df)),
        random_state=SEED
    )

    results = {}

    for invert in [False, True]:
        dice_scores = []

        for _, record in samples.iterrows():
            image, true_mask = load_pair(
                record["image"],
                record["mask"],
                size=IMAGE_SIZE
            )

            pred_mask = otsu_segment(
                image,
                invert=invert
            )

            metrics = binary_segmentation_metrics(
                pred_mask, true_mask
            )

            dice_scores.append(metrics["dice"])

        results[invert] = np.mean(dice_scores)

    print("\nTraining polarity selection:")
    print(f"Normal:   {results[False]:.4f}")
    print(f"Inverted: {results[True]:.4f}")

    # On a tie, prefer normal polarity
    best_invert = results[True] > results[False]

    print(f"Selected invert={best_invert}")

    return best_invert


# ----------------------------------------
# Visualize prediction examples
# ----------------------------------------

def save_visualizations(examples):
    if not examples:
        return

    fig, axes = plt.subplots(
        len(examples),
        3,
        figsize=(12, 3 * len(examples)),
        squeeze=False
    )

    for i, (image, true_mask, pred_mask, name) in enumerate(
        examples
    ):
        axes[i, 0].imshow(image)
        axes[i, 0].set_title(
            f"Original: {name}", fontsize=10
        )

        axes[i, 1].imshow(
            true_mask,
            cmap="gray",
            vmin=0,
            vmax=1
        )
        axes[i, 1].set_title(
            "Ground Truth", fontsize=10
        )

        axes[i, 2].imshow(
            pred_mask,
            cmap="gray",
            vmin=0,
            vmax=1
        )
        axes[i, 2].set_title(
            "Otsu Prediction", fontsize=10
        )

        for ax in axes[i]:
            ax.axis("off")

    plt.tight_layout(pad=2.0)

    save_path = FIGURES_DIR / "otsu_examples.png"

    plt.savefig(
        save_path,
        dpi=150,
        bbox_inches="tight"
    )

    print(f"\nSaved visualization: {save_path}")

    plt.show()


# ----------------------------------------
# Evaluate Otsu on validation images
# ----------------------------------------

def evaluate():
    invert = choose_polarity()

    val_df = load_manifest(VAL_CSV, ROOT)

    example_ids = set(
    val_df.drop_duplicates("vehicle_id")
    .sample(n=3, random_state=42)["image"]
)

    results = []
    examples = []

    print(f"\nEvaluating {len(val_df)} validation images...")

    for index, record in val_df.iterrows():

        image, true_mask = load_pair(
            record["image"],
            record["mask"],
            size=IMAGE_SIZE
        )

        # Measure prediction latency only
        start = perf_counter()

        pred_mask = otsu_segment(
            image,
            invert=invert
        )

        latency_ms = (perf_counter() - start) * 1000

        metrics = binary_segmentation_metrics(
            pred_mask,
            true_mask
        )

        results.append({
            "image": str(record["image"]),
            "vehicle_id": record["vehicle_id"],
            "view": record["view"],
            "dice": metrics["dice"],
            "iou": metrics["iou"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "latency_ms": latency_ms
        })

        if record["image"] in example_ids:
            examples.append((
                image,
                true_mask,
                pred_mask,
                Path(record["image"]).stem
            ))

        if len(results) % 50 == 0:
            print(f"Processed {len(results)} images")

    # ----------------------------------------
    # Save results
    # ----------------------------------------

    results_df = pd.DataFrame(results)

    results_df.to_csv(
        METRICS_DIR / "otsu_validation.csv",
        index=False
    )

    print("\n========== OTSU RESULTS ==========")

    for metric in [
        "dice", "iou",
        "precision", "recall", "latency_ms"
    ]:
        print(
            f"{metric}: "
            f"{results_df[metric].mean():.4f}"
        )

    print("==================================")

    save_visualizations(examples)


if __name__ == "__main__":
    evaluate()
