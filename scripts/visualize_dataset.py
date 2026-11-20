from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.data.carvana import load_manifest, load_pair


# To run this code, go to the project root and run:
# python -m scripts.visualize_dataset


ROOT = Path(__file__).resolve().parents[1]
SPLIT_CSV = ROOT / "data" / "splits" / "train.csv"

# Start with three examples
NUM_EXAMPLES = 3

# Working resolution from our project plan
IMAGE_SIZE = (768, 512)


def visualize_examples():
    df = load_manifest(SPLIT_CSV, ROOT)

    # Fixed sampling for reproducibility
    samples = df.sample(
        n=min(NUM_EXAMPLES, len(df)),
        random_state=42
    )

    fig, axes = plt.subplots(
        len(samples),
        3,
        figsize=(9, 2 * len(samples)),
        squeeze=False
    )

    for row_index, (_, record) in enumerate(
        samples.iterrows()
    ):
        image, mask = load_pair(
            record["image"],
            record["mask"],
            size=IMAGE_SIZE
        )

        # Confirm binary-mask format
        assert set(np.unique(mask)).issubset({0, 1})

        # Generate colored foreground overlay
        overlay = image.copy().astype(np.float32)

        foreground = mask.astype(bool)

        overlay[foreground] = (
            0.6 * overlay[foreground]
            + 0.4 * np.array([0, 255, 0])
        )

        overlay = overlay.astype(np.uint8)

        # Original image
        axes[row_index, 0].imshow(image)
        axes[row_index, 0].set_title(
            f"Original: {record['vehicle_id']}"
        )

        # Ground-truth mask
        axes[row_index, 1].imshow(
            mask,
            cmap="gray",
            vmin=0,
            vmax=1
        )
        axes[row_index, 1].set_title("Ground Truth")

        # Mask overlay
        axes[row_index, 2].imshow(overlay)
        axes[row_index, 2].set_title("Mask Overlay")

        print(
            f"{record['vehicle_id']}: "
            f"image={image.shape}, "
            f"mask={mask.shape}, "
            f"unique mask values={np.unique(mask)}"
        )

        for ax in axes[row_index]:
            ax.axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    visualize_examples()
