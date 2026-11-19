import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.carvana import load_manifest, load_pair
from src.evaluation.metrics import binary_segmentation_metrics
from src.models.classical.superpixel_rf import SuperpixelRF

ROOT = Path(__file__).resolve().parents[1]
VAL_CSV = ROOT / "data" / "splits" / "val.csv"

THRESHOLDS = np.round(np.arange(0.30, 0.701, 0.05), 2)


def tune(experiment_id):
    experiment_dir = ROOT / "outputs" / "experiments" / experiment_id

    model = SuperpixelRF(
        experiment_dir / "model.joblib",
        experiment_dir / "config.json"
    )

    image_size = tuple(model.config["image_size"])
    df = load_manifest(VAL_CSV, ROOT)

    results = []

    print(f"Evaluating {len(df)} images at {len(THRESHOLDS)} thresholds")

    for i, (_, record) in enumerate(df.iterrows(), 1):
        image, true_mask = load_pair(
            record["image"],
            record["mask"],
            size=image_size
        )

        # Run SLIC + RF only once per image
        _, score_map = model.predict(image, threshold=0.5)

        for threshold in THRESHOLDS:
            pred_mask = (score_map >= threshold).astype(np.uint8)

            metrics = binary_segmentation_metrics(
                pred_mask, true_mask
            )

            results.append({
                "image": Path(record["image"]).name,
                "vehicle_id": record["vehicle_id"],
                "view": record["view"],
                "threshold": float(threshold),
                **metrics
            })

        if i % 25 == 0:
            print(f"Processed {i}/{len(df)}")

    results_df = pd.DataFrame(results)

    summary = (
        results_df
        .groupby("threshold")[["dice", "iou", "precision", "recall"]]
        .mean()
        .reset_index()
    )

    best_row = summary.loc[summary["dice"].idxmax()]
    best_threshold = float(best_row["threshold"])

    results_df.to_csv(
        experiment_dir / "threshold_sweep_per_image.csv",
        index=False
    )

    summary.to_csv(
        experiment_dir / "threshold_sweep.csv",
        index=False
    )

    best_config = {
        "experiment_id": experiment_id,
        "selection_split": "vehicle_level_validation",
        "validation_images": len(df),
        "selection_metric": "mean_dice",
        "best_threshold": best_threshold,
        "metrics": {
            metric: float(best_row[metric])
            for metric in ["dice", "iou", "precision", "recall"]
        }
    }

    (experiment_dir / "threshold_selection.json").write_text(
        json.dumps(best_config, indent=2),
        encoding="utf-8"
    )

    print("\n===== THRESHOLD SWEEP =====")
    print(summary.to_string(index=False))
    print(f"\nBest threshold: {best_threshold:.2f}")
    print(f"Best Dice: {best_row['dice']:.6f}")
    print(f"Best IoU: {best_row['iou']:.6f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", required=True)
    args = parser.parse_args()

    tune(args.experiment)
