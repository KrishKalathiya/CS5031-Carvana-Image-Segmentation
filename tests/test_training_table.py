import numpy as np
import pandas as pd
from PIL import Image

from scripts import build_training_table as builder
from src.models.classical.features import (
    FEATURE_COLUMNS,
    extract_region_features
)


def test_region_fractions_and_foreground_ties():
    regions = np.repeat(np.array([0, 2, 7, 10, 12])[:, None], 4, axis=1)
    truth = np.array([
        [0, 0, 0, 0],
        [1, 0, 0, 0],
        [1, 1, 0, 0],
        [1, 1, 1, 0],
        [1, 1, 1, 1]
    ])

    labels = builder.create_region_labels(regions, truth)

    np.testing.assert_array_equal(
        labels.foreground_fraction, [0, 0.25, 0.5, 0.75, 1]
    )
    np.testing.assert_array_equal(labels.label, [0, 0, 1, 1, 1])
    assert labels.index.tolist() == [0, 2, 7, 10, 12]


def test_features_and_whole_image_edge_context():
    image = np.zeros((4, 6, 3), dtype=np.uint8)
    image[:, 3:] = 255
    regions = np.tile([7, 7, 7, 12, 12, 12], (4, 1))

    features = extract_region_features(image, regions)

    assert features.columns.tolist() == FEATURE_COLUMNS
    assert features.index.tolist() == [7, 12]
    np.testing.assert_allclose(features.lab_l_mean, [0, 100], atol=1e-4)
    np.testing.assert_allclose(features.lab_l_std, 0, atol=1e-6)
    np.testing.assert_allclose(features.centroid_x, [0.2, 0.8])
    np.testing.assert_allclose(features.centroid_y, [0.5, 0.5])
    np.testing.assert_allclose(features.area_fraction, [0.5, 0.5])

    # Each region is constant, but the whole image has a shared black/white edge.
    np.testing.assert_allclose(features.sobel_mean, 1 / (3 * np.sqrt(2)), rtol=1e-6)
    np.testing.assert_allclose(features.sobel_std, 1 / 3, rtol=1e-6)


def test_saved_tables_keep_metadata_out_of_features(tmp_path, monkeypatch):
    records = []
    for name, mask_value in [("car_a", 0), ("car_b", 255)]:
        image_path = tmp_path / f"{name}_01.png"
        mask_path = tmp_path / f"{name}_01_mask.png"
        Image.fromarray(np.full((20, 24, 3), 100, dtype=np.uint8)).save(image_path)
        Image.fromarray(np.full((20, 24), mask_value, dtype=np.uint8)).save(mask_path)
        records.append({
            "image": image_path.name, "mask": mask_path.name,
            "vehicle_id": name, "view": 1
        })

    manifest_path = tmp_path / "train.csv"
    pd.DataFrame(records).to_csv(manifest_path, index=False)

    monkeypatch.setattr(builder, "ROOT", tmp_path)
    monkeypatch.setattr(builder, "TRAIN_CSV", manifest_path)
    monkeypatch.setattr(builder, "TABLES_DIR", tmp_path / "tables")
    monkeypatch.setattr(builder, "IMAGE_SIZE", None)
    monkeypatch.setattr(builder, "N_SEGMENTS", 4)
    builder.build_training_table()

    table = pd.read_csv(builder.TABLES_DIR / "slic_train.csv")
    metadata = pd.read_csv(builder.TABLES_DIR / "slic_train_metadata.csv")

    assert table.columns.tolist() == FEATURE_COLUMNS + ["label"]
    assert len(table) == len(metadata) > 0
    np.testing.assert_array_equal(
        table.label, (metadata.foreground_fraction >= 0.5).astype(int)
    )
    assert table.loc[metadata.vehicle_id == "car_a", "label"].eq(0).all()
    assert table.loc[metadata.vehicle_id == "car_b", "label"].eq(1).all()
    # Identical images with different masks must still have identical features.
    np.testing.assert_array_equal(
        table.loc[metadata.vehicle_id == "car_a", FEATURE_COLUMNS],
        table.loc[metadata.vehicle_id == "car_b", FEATURE_COLUMNS]
    )
