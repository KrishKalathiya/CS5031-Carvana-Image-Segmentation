import argparse
import json

from pathlib import Path
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.data.carvana import load_manifest, load_pair
from src.evaluation.metrics import binary_segmentation_metrics
from src.models.classical.superpixel_rf import SuperpixelRF

# To run this script, go to the project root and run:
# python -m scripts.eval_classical --experiment <experiment_id>
# <experiment_id> is the name of the experiment folder in outputs/experiments, found in the config.json

ROOT = Path(__file__).resolve().parents[1]

VAL_CSV = ROOT / "data" / "splits" / "val.csv"

MODEL_PATH = (
    ROOT / "outputs" / "checkpoints" / "superpixel_rf.joblib"
)
CONFIG_PATH = (
    ROOT / "outputs" / "checkpoints" / "superpixel_rf.json"
)

OUTPUT_DIR = ROOT / "outputs" / "rf_validation"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

THRESHOLD = 0.5

# Set to None for the complete validation set
MAX_IMAGES = None


def evaluate(experiment_id):
    experiment_dir = (
        ROOT / "outputs" / "experiments" / experiment_id
    )

    model_path = experiment_dir / "model.joblib"
    config_path = experiment_dir / "config.json"

    if not model_path.is_file() or not config_path.is_file():
        raise FileNotFoundError(
            f"Missing model or configuration for {experiment_id}"
        )

    model = SuperpixelRF(model_path, config_path)

    image_size = tuple(model.config["image_size"])

    df = load_manifest(VAL_CSV, ROOT)

    if MAX_IMAGES is not None:
        # One view per vehicle for the initial evaluation
        df = (
            df.sort_values(["vehicle_id", "view"])
            .drop_duplicates("vehicle_id")
            .head(MAX_IMAGES)
        )

    results = []
    examples = []

    print(f"Evaluating {len(df)} validation images...")

    for _, record in df.iterrows():

        image, true_mask = load_pair(
            record["image"],
            record["mask"],
            size=image_size
        )

        start = perf_counter()

        pred_mask, score_map = model.predict(
            image,
            threshold=THRESHOLD
        )

        latency_ms = (perf_counter() - start) * 1000

        metrics = binary_segmentation_metrics(
            pred_mask, true_mask
        )

        results.append({
            "image": Path(record["image"]).name,
            "vehicle_id": record["vehicle_id"],
            "view": record["view"],
            **metrics,
            "latency_ms": latency_ms
        })

        if len(examples) < 3:
            examples.append((
                image.copy(),
                true_mask.copy(),
                pred_mask.copy(),
                score_map.copy(),
                Path(record["image"]).stem
            ))

        if len(results) % 10 == 0:
            print(f"Processed {len(results)}/{len(df)}")

    # Summarize metrics
    results_df = pd.DataFrame(results)

    results_df.to_csv(
        experiment_dir / "metrics.csv",
        index=False
    )

    metric_columns = [
    "dice", "iou", "precision",
    "recall", "latency_ms"
    ]

    summary = {
        "experiment_id": experiment_id,
        "evaluation_split": "vehicle_level_validation",
        "validation_images": int(len(results_df)),
        "decision_threshold": THRESHOLD,
        "metrics": {
            column: float(results_df[column].mean())
            for column in metric_columns
        }
    }

    (experiment_dir / "summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8"
    )

    print("\n========== RANDOM FOREST RESULTS ==========")

    for metric in [
        "dice", "iou", "precision",
        "recall", "latency_ms"
    ]:
        print(
            f"{metric}: {results_df[metric].mean():.4f}"
        )

    print("===========================================")

    # Show example predictions
    fig, axes = plt.subplots(
        len(examples),
        4,
        figsize=(14, 3 * len(examples)),
        squeeze=False
    )

    for i, (image, truth, pred, scores, name) in enumerate(examples):

        panels = [
            (image, "Original", None),
            (truth, "Ground Truth", "gray"),
            (scores, "Foreground Probability", "viridis"),
            (pred, "RF Prediction", "gray")
        ]

        for j, (data, title, cmap) in enumerate(panels):
            axes[i, j].imshow(
                data,
                cmap=cmap,
                **(
                    {"vmin": 0, "vmax": 1}
                    if cmap is not None else {}
                )
            )

            axes[i, j].set_title(
                f"{title}\n{name}" if j == 0 else title,
                fontsize=9
            )
            axes[i, j].axis("off")

    plt.tight_layout(pad=2.0)
    plt.savefig(
        experiment_dir / "examples.png",
        dpi=150,
        bbox_inches="tight"
    )
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--experiment",
        required=True,
        help="ID of the trained experiment to evaluate"
    )

    args = parser.parse_args()

    evaluate(args.experiment)
