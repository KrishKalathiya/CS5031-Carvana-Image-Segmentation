import json
from pathlib import Path

import joblib
import numpy as np

from src.models.classical.slic import slic_regions
from src.models.classical.features import (
    FEATURE_COLUMNS,
    extract_region_features
)


class SuperpixelRF:
    def __init__(self, model_path, config_path):
        self.model = joblib.load(model_path)

        self.config = json.loads(
            Path(config_path).read_text(encoding="utf-8")
        )

        if self.config["feature_columns"] != FEATURE_COLUMNS:
            raise ValueError("Feature schema mismatch.")

        if self.model.n_features_in_ != len(FEATURE_COLUMNS):
            raise ValueError("Model expects different features.")

        self.classes = self.model.classes_

        if 1 not in self.classes:
            raise ValueError("Model has no foreground class.")

        self.foreground_index = list(self.classes).index(1)

    def predict(self, image, threshold=0.5):
        """
        Predict a binary vehicle mask from an RGB image.

        Returns
        -------
        binary_mask : ndarray (H, W), uint8
        score_map   : ndarray (H, W), float32
        """
        expected_size = tuple(self.config["image_size"])

        if (image.shape[1], image.shape[0]) != expected_size:
            raise ValueError(
                f"Expected image size {expected_size}, "
                f"got {(image.shape[1], image.shape[0])}"
            )

        if not 0 <= threshold <= 1:
            raise ValueError("Threshold must be between 0 and 1.")

        # 1. Generate SLIC superpixels
        regions = slic_regions(
            image,
            n_segments=self.config["n_segments"],
            compactness=self.config["compactness"],
            sigma=self.config["sigma"]
        )

        # 2. Extract region features
        features = extract_region_features(image, regions)

        X = features[FEATURE_COLUMNS].to_numpy(
            dtype=np.float32
        )

        # 3. Predict foreground probability for each region
        probabilities = self.model.predict_proba(X)
        scores = probabilities[:, self.foreground_index]

        # 4. Reconstruct pixel-level probability map
        region_ids = features.index.to_numpy()

        # np.searchsorted maps each pixel's region ID
        # to its corresponding feature row.
        indices = np.searchsorted(region_ids, regions)

        if not np.array_equal(region_ids[indices], regions):
            raise ValueError("Unmatched superpixel IDs.")

        score_map = scores[indices].astype(np.float32)

        # 5. Generate binary mask
        binary_mask = (
            score_map >= threshold
        ).astype(np.uint8)

        return binary_mask, score_map
