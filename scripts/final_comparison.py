"""Run from the project root with a trained forest experiment.

python -m scripts.final_comparison --experiment EXPERIMENT_ID --workers 6
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from functools import partial
import json
from pathlib import Path
from time import perf_counter

from matplotlib.figure import Figure
import numpy as np
import pandas as pd
from PIL import Image
from threadpoolctl import threadpool_limits

from scripts.tune_classical_threshold import THRESHOLDS
from src.data.carvana import load_manifest, load_mask, load_pair
from src.evaluation.metrics import binary_segmentation_metrics
from src.models.classical.otsu import otsu_segment
from src.models.classical.superpixel_rf import SuperpixelRF

ROOT = Path(__file__).resolve().parents[1]
OTSU, FOREST = "Otsu", "SLIC + random forest"
METRICS = ["dice", "iou", "precision", "recall"]


def choose_polarity(train, size):
    scores = []
    for row in train.sample(n=min(48, len(train)), random_state=42).itertuples(index=False):
        image, truth = load_pair(row.image, row.mask, size=size)
        mask = otsu_segment(image)
        scores.append([binary_segmentation_metrics(p, truth)["dice"] for p in (mask, 1 - mask)])
    normal, inverted = np.mean(scores, axis=0)
    return bool(inverted > normal)


def validation_dice(row, model, size):
    image, truth = load_pair(row.image, row.mask, size=size)
    _, scores = model.predict(image)
    return [
        binary_segmentation_metrics(scores >= np.float32(t), truth)["dice"]
        for t in THRESHOLDS
    ]


def predict_native(path, methods, working_size):
    start = perf_counter()
    with Image.open(path) as source:
        native_size = source.size
        image = np.asarray(source.convert("RGB").resize(working_size, Image.Resampling.LANCZOS))
    preprocess_ms = (perf_counter() - start) * 1000

    outputs = []
    for name, predict in methods.items():
        start = perf_counter()
        mask = Image.fromarray(predict(image)).resize(native_size, Image.Resampling.NEAREST)
        outputs.append((name, np.asarray(mask), preprocess_ms + (perf_counter() - start) * 1000))
    return outputs


def score_test_image(row, methods, size, output):
    outputs = predict_native(row.image, methods, size)
    truth = load_mask(row.mask)
    results = []
    for name, mask, latency_ms in outputs:
        metrics = binary_segmentation_metrics(mask, truth)
        destination = output / "masks" / name / f"{row.image.stem}.png"
        Image.fromarray(mask * 255).save(destination, compress_level=1)
        results.append({
            "image_id": row.image.stem, "vehicle_id": row.vehicle_id, "view": row.view,
            "model": name, **metrics, "batch_latency_ms": latency_ms
        })
    return results


def save_examples(test, results, output):
    indexed = test.set_index(test.image.map(lambda path: path.stem))
    forest = results.loc[results.model == FOREST].set_index("image_id")
    worst = forest.dice.idxmin()
    fixed = indexed.drop_duplicates("vehicle_id").drop(index=worst, errors="ignore")
    examples = [*fixed.sample(n=min(2, len(fixed)), random_state=42).index, worst]
    fig = Figure(figsize=(13, 2.6 * len(examples)))
    axes = fig.subplots(len(examples), 4, squeeze=False)
    dice = results.pivot(index="image_id", columns="model", values="dice")
    for name, row_axes in zip(examples, axes):
        row = indexed.loc[name]
        with Image.open(row.image) as source:
            image = source.convert("RGB")
        panels = [(image, f"{name}\nLowest forest Dice" if name == worst else name, {})]
        panels.append((load_mask(row["mask"]), "Ground truth", {"cmap": "gray", "vmin": 0, "vmax": 1}))
        for method in (OTSU, FOREST):
            with Image.open(output / "masks" / method / f"{name}.png") as mask:
                panels.append((np.asarray(mask), f"{method}\nDice {dice.loc[name, method]:.4f}",
                               {"cmap": "gray", "vmin": 0, "vmax": 255}))
        for ax, (panel, title, style) in zip(row_axes, panels):
            ax.imshow(panel, **style)
            ax.set_title(title, fontsize=9)
            ax.axis("off")
    fig.tight_layout()
    fig.savefig(output / "examples.png", dpi=150, bbox_inches="tight")


def run(experiment, workers):
    folder = ROOT / "outputs" / "experiments" / experiment
    checkpoint = folder / "model.joblib"
    model = SuperpixelRF(checkpoint, folder / "config.json")
    # Use one tree worker per image so the image workers can share CPU cores.
    model.model.n_jobs = 1
    size = tuple(model.config["image_size"])

    train, val, test = [load_manifest(ROOT / "data" / "splits" / f"{split}.csv", ROOT)
                        for split in ("train", "val", "test")]
    train_ids, val_ids, test_ids = [set(frame.vehicle_id) for frame in (train, val, test)]
    if (train_ids & val_ids) or (train_ids & test_ids) or (val_ids & test_ids):
        raise ValueError("Vehicle IDs overlap across splits.")
    if not set(model.config["training_vehicles"]) <= train_ids:
        raise ValueError("Checkpoint training vehicles must belong to the training split.")
    if any(frame.empty or frame.image.duplicated().any() for frame in (train, val, test)):
        raise ValueError("Each split must contain unique images.")

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output = ROOT / "outputs" / "metrics" / "final_comparison" / stamp
    for method in (OTSU, FOREST):
        (output / "masks" / method).mkdir(parents=True)
    print(f"Writing results to {output}", flush=True)

    with threadpool_limits(limits=1), ThreadPoolExecutor(max_workers=workers) as pool:
        invert = choose_polarity(train, size)
        tune = partial(validation_dice, model=model, size=size)
        scores = []
        print(f"Selecting the forest threshold on {len(val)} validation images", flush=True)
        for index, result in enumerate(pool.map(tune, val.itertuples(index=False)), 1):
            scores.append(result)
            if index % 50 == 0 or index == len(val):
                print(f"Validation {index}/{len(val)}", flush=True)
        means = np.mean(scores, axis=0)
        selected = int(np.argmax(means))
        threshold = float(THRESHOLDS[selected])
        pd.DataFrame({"threshold": THRESHOLDS, "mean_dice_working_resolution": means}).to_csv(
            output / "validation_thresholds.csv", index=False
        )
        settings = {
            "experiment": experiment, "checkpoint": str(checkpoint), "workers": workers,
            "input_size_wh": size, "test_mask_resize": "nearest neighbor to original image size",
            "otsu_invert": invert, "otsu_polarity_split": "train", "forest_threshold": threshold,
            "threshold_selection_split": "val", "validation_images": len(val),
            "validation_forest_dice_working_resolution": float(means[selected]),
            "test_images": len(test), "test_vehicles": len(test_ids)
        }
        (output / "run.json").write_text(json.dumps(settings, indent=2), encoding="utf-8")

        methods = {
            OTSU: partial(otsu_segment, invert=invert),
            FOREST: lambda image: model.predict(image, threshold=threshold)[0]
        }
        timing_sample = val.sample(n=min(16, len(val)), random_state=42)
        timings = []
        # Warm up both models before measuring one image at a time.
        predict_native(timing_sample.iloc[0].image, methods, size)
        for path in timing_sample.image:
            timings.extend((name, elapsed) for name, _, elapsed in predict_native(path, methods, size))
        serial_ms = pd.DataFrame(timings, columns=["model", "serial_ms_per_image"]).groupby("model").mean()

        score = partial(score_test_image, methods=methods, size=size, output=output)
        rows = []
        start = perf_counter()
        print(f"Frozen threshold {threshold:g}. Scoring {len(test)} held-out images", flush=True)
        for index, pair in enumerate(pool.map(score, test.itertuples(index=False)), 1):
            rows.extend(pair)
            if index % 50 == 0 or index == len(test):
                print(f"Test {index}/{len(test)}", flush=True)
        test_minutes = (perf_counter() - start) / 60

    results = pd.DataFrame(rows)
    results.to_csv(output / "per_image.csv", index=False)
    groups = results.groupby("model", sort=False)
    summary = groups[METRICS].mean().join(groups.dice.median().rename("median_dice")).join(serial_ms)
    summary.to_csv(output / "summary.csv")
    save_examples(test, results, output)

    gain = summary.loc[FOREST, "dice"] - summary.loc[OTSU, "dice"]
    report = (
        f"Held-out comparison on {len(test)} images from {len(test_ids)} vehicles\n"
        "Masks scored at each image's original resolution. Scores are per-image averages.\n\n"
        f"{summary.to_string(float_format=lambda value: f'{value:.4f}')}\n\n"
        f"Random forest mean Dice difference vs. Otsu {gain:+.4f}.\n"
        f"Forest threshold {threshold:g} selected on {len(val)} validation images.\n"
        f"Otsu darker foreground polarity {invert}, selected on training images.\n"
        f"Serial latency measured on {len(timing_sample)} validation images and includes image loading, "
        "preprocessing, prediction, and resize to original dimensions.\n"
        f"Held-out scoring and mask saving took {test_minutes:.2f} minutes with {workers} workers.\n"
    )
    (output / "report.txt").write_text(report, encoding="utf-8")
    print(f"\n{report}\nSaved results to {output}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", required=True, help="Folder under outputs/experiments.")
    parser.add_argument("--workers", type=int, default=6, help="Images processed concurrently.")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    run(args.experiment, args.workers)
