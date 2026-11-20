import argparse
from datetime import datetime
import json
from pathlib import Path
from time import perf_counter

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import pandas as pd
from PIL import Image

from src.data.carvana import load_manifest, load_pair
from src.evaluation.metrics import binary_segmentation_metrics
from src.models.classical.superpixel_rf import SuperpixelRF

# Run from the project root. Train a checkpoint if you need one.
#   python -m scripts.train_classical
#
# Use the trained folder name from outputs/experiments as EXPERIMENT_ID.
#   python -m scripts.eval_classical --experiment EXPERIMENT_ID
#   python -m scripts.eval_classical --experiment EXPERIMENT_ID --grabcut
# Add --max-images 24 to either evaluation command for a quick sample.

ROOT = Path(__file__).resolve().parents[1]
VAL_CSV = ROOT / "data" / "splits" / "val.csv"
THRESHOLD = 0.5
MAX_IMAGES = None
METRICS = ["dice", "iou", "precision", "recall", "latency_ms"]


def evaluate(experiment_id, grabcut=False, max_images=None, threshold=None):
    experiment_dir = ROOT / "outputs" / "experiments" / experiment_id
    model_path = experiment_dir / "model.joblib"
    config_path = experiment_dir / "config.json"
    if not model_path.is_file() or not config_path.is_file():
        raise FileNotFoundError(f"Missing model or configuration for {experiment_id}")
    model = SuperpixelRF(model_path, config_path)
    image_size = tuple(model.config["image_size"])
    df = load_manifest(VAL_CSV, ROOT)
    if max_images is not None:
        df = df.sample(n=min(max_images, len(df)), random_state=42).sort_index()
    elif MAX_IMAGES is not None:
        df = df.sort_values(["vehicle_id", "view"]).drop_duplicates("vehicle_id").head(MAX_IMAGES)

    output_dir = experiment_dir
    if grabcut:
        from src.models.classical.grabcut import refine_grabcut

        selection = experiment_dir / "threshold_selection.json"
        if threshold is None and selection.is_file():
            threshold = json.loads(selection.read_text(encoding="utf-8"))["best_threshold"]
    if grabcut or max_images is not None or threshold is not None:
        method = "grabcut" if grabcut else "forest"
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        output_dir = ROOT / "outputs" / "predictions" / experiment_dir.name / f"{method}_{stamp}"
        output_dir.mkdir(parents=True)
    threshold = THRESHOLD if threshold is None else threshold

    example_rows = df.sample(n=min(3, len(df)), random_state=42) if grabcut else df.head(3)
    example_ids = set(example_rows.image)
    results, examples = [], []
    print(f"Evaluating {len(df)} validation images at threshold {threshold:g}...", flush=True)
    for record in df.itertuples(index=False):
        image, truth = load_pair(record.image, record.mask, size=image_size)
        start = perf_counter()
        forest, scores = model.predict(image, threshold=threshold)
        forest_ms = (perf_counter() - start) * 1000
        row = {"image": record.image.name, "vehicle_id": record.vehicle_id, "view": record.view,
               **binary_segmentation_metrics(forest, truth), "latency_ms": forest_ms}

        if grabcut:
            start = perf_counter()
            refined, seeds, fallback = refine_grabcut(image, forest, scores)
            grabcut_ms = (perf_counter() - start) * 1000
            metrics = binary_segmentation_metrics(refined, truth)
            row.update({f"refined_{key}": value for key, value in metrics.items()})
            row.update(grabcut_latency_ms=grabcut_ms, refined_latency_ms=forest_ms + grabcut_ms,
                       dice_change=metrics["dice"] - row["dice"], fallback_reason=fallback)
            if fallback:
                print(f"{record.image.stem} fell back to the forest mask ({fallback})", flush=True)
            for suffix, mask in (("forest", forest * 255), ("refined", refined * 255), ("seeds", seeds)):
                Image.fromarray(mask).save(output_dir / f"{record.image.stem}_{suffix}.png")

        results.append(row)
        if record.image in example_ids:
            panels = [(image, f"Original\n{record.image.stem}", None, 1),
                      (truth, "Ground Truth", "gray", 1),
                      (scores, "Foreground Probability", "viridis", 1),
                      (forest, "RF Prediction", "gray", 1)]
            if grabcut:
                colors = ListedColormap(["#2755a0", "#167c4c", "#b7d5fb", "#b4e6a8"])
                panels.extend([(seeds, "Seeds", colors, 3), (refined, "GrabCut", "gray", 1)])
            examples.append(panels)
        if len(results) % 10 == 0:
            print(f"Processed {len(results)}/{len(df)}", flush=True)

    frame = pd.DataFrame(results)
    frame.to_csv(output_dir / "metrics.csv", index=False)
    summary = {
        "experiment_id": experiment_id, "evaluation_split": "vehicle_level_validation",
        "validation_images": len(frame), "decision_threshold": threshold,
        "metrics": frame[METRICS].mean().to_dict()
    }
    if grabcut:
        summary["grabcut"] = {
            "metrics": {key: float(frame[f"refined_{key}"].mean()) for key in METRICS},
            "mean_added_latency_ms": float(frame.grabcut_latency_ms.mean()),
            "mean_dice_change": float(frame.dice_change.mean()),
            "improved_images": int((frame.dice_change > 0).sum()),
            "worsened_images": int((frame.dice_change < 0).sum()),
            "fallbacks": int(frame.fallback_reason.ne("").sum())
        }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Saved evaluation to {output_dir}")

    columns = len(examples[0])
    fig, axes = plt.subplots(len(examples), columns, figsize=(3.5 * columns, 3 * len(examples)), squeeze=False)
    for axes_row, panels in zip(axes, examples):
        for ax, (data, title, cmap, vmax) in zip(axes_row, panels):
            ax.imshow(data, **({"cmap": cmap, "vmin": 0, "vmax": vmax} if cmap is not None else {}))
            ax.set_title(title, fontsize=9)
            ax.axis("off")
    if grabcut:
        fig.suptitle("Seeds · Blue background · Green foreground · Dark definite · Light probable")
    fig.tight_layout()
    fig.savefig(output_dir / "examples.png", dpi=150, bbox_inches="tight")
    if not grabcut:
        plt.show()
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", required=True, help="ID of the trained experiment to evaluate")
    parser.add_argument("--grabcut", action="store_true", help="Compare forest masks with GrabCut refinement")
    parser.add_argument("--max-images", type=int, help="Limit evaluation to a seeded validation sample")
    parser.add_argument("--threshold", type=float, help="Override the forest mask threshold")
    args = parser.parse_args()
    if args.max_images is not None and args.max_images < 1:
        parser.error("--max-images must be positive")
    evaluate(args.experiment, args.grabcut, args.max_images, args.threshold)
