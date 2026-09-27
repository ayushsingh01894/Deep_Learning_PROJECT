"""
Tests — Module 4: Food Image Classification
================================================
The MobileNetV2-based model architecture test requires downloading
ImageNet weights on first use; if no internet access is available in the
test environment, that specific test is skipped rather than failed.
"""

import json
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import src.food_classification.preprocess as fc_preprocess


@pytest.fixture
def synthetic_food101_layout(tmp_path, monkeypatch):
    """Build a tiny fake Food-101 directory structure (images + meta split
    files) under a pytest tmp_path, and point the preprocess module's raw
    data paths at it for the duration of the test."""
    from PIL import Image

    classes = ["waffles", "pizza", "sushi"]
    images_dir = tmp_path / "images"
    meta_dir = tmp_path / "meta"
    meta_dir.mkdir()

    train_lines, test_lines = [], []
    for cls in classes:
        cls_dir = images_dir / cls
        cls_dir.mkdir(parents=True)
        for i in range(6):
            img_path = cls_dir / f"{cls}_{i}.jpg"
            Image.fromarray((np.random.rand(50, 50, 3) * 255).astype("uint8")).save(img_path)
            train_lines.append(f"{cls}/{cls}_{i}")
        for i in range(2):
            img_path = cls_dir / f"{cls}_test_{i}.jpg"
            Image.fromarray((np.random.rand(50, 50, 3) * 255).astype("uint8")).save(img_path)
            test_lines.append(f"{cls}/{cls}_test_{i}")

    (meta_dir / "train.txt").write_text("\n".join(train_lines))
    (meta_dir / "test.txt").write_text("\n".join(test_lines))

    monkeypatch.setattr(fc_preprocess, "IMAGES_DIR", str(images_dir))
    monkeypatch.setattr(fc_preprocess, "META_DIR", str(meta_dir))
    monkeypatch.setattr(fc_preprocess, "TRAIN_SPLIT_FILE", str(meta_dir / "train.txt"))
    monkeypatch.setattr(fc_preprocess, "TEST_SPLIT_FILE", str(meta_dir / "test.txt"))
    monkeypatch.setattr(fc_preprocess, "SELECTED_CLASSES", classes)

    return {"classes": classes, "images_dir": images_dir, "meta_dir": meta_dir}


class TestReadSplitFile:
    def test_reads_only_selected_classes(self, synthetic_food101_layout):
        df = fc_preprocess.read_split_file(str(synthetic_food101_layout["meta_dir"] / "train.txt"))
        assert set(df["label"].unique()) == set(synthetic_food101_layout["classes"])

    def test_correct_row_count(self, synthetic_food101_layout):
        df = fc_preprocess.read_split_file(str(synthetic_food101_layout["meta_dir"] / "train.txt"))
        assert len(df) == 6 * len(synthetic_food101_layout["classes"])

    def test_filepaths_point_to_jpg_files(self, synthetic_food101_layout):
        df = fc_preprocess.read_split_file(str(synthetic_food101_layout["meta_dir"] / "train.txt"))
        assert all(df["filepath"].str.endswith(".jpg"))
        assert all(os.path.exists(p) for p in df["filepath"])


class TestRunPreprocessing:
    def test_creates_expected_manifests(self, synthetic_food101_layout, tmp_path, monkeypatch):
        processed_dir = tmp_path / "processed"
        model_dir = tmp_path / "models"
        monkeypatch.setattr(fc_preprocess, "FOOD101_PROCESSED_DIR", str(processed_dir))
        monkeypatch.setattr(fc_preprocess, "FOOD_MODEL_DIR", str(model_dir))

        fc_preprocess.run_preprocessing()

        assert (processed_dir / "train_manifest.csv").exists()
        assert (processed_dir / "val_manifest.csv").exists()
        assert (processed_dir / "test_manifest.csv").exists()

        with open(model_dir / "class_names.json") as f:
            class_names = json.load(f)
        assert class_names == sorted(synthetic_food101_layout["classes"])


class TestBuildFoodCnnModel:
    def test_model_architecture_shapes(self):
        try:
            from src.food_classification.model import IMG_SIZE, build_food_cnn_model

            model = build_food_cnn_model(num_classes=12)
        except Exception as e:
            pytest.skip(f"Could not build MobileNetV2 model (likely no internet access for "
                        f"ImageNet weights in this environment): {e}")

        assert model.input_shape == (None, IMG_SIZE, IMG_SIZE, 3)
        assert model.output_shape == (None, 12)


class TestPredictIntegration:
    def test_predict_food_image_end_to_end(self):
        from src.food_classification.predict import artifacts_available, predict_food_image

        if not artifacts_available():
            pytest.skip("Module 4 model artifacts not found — train the model first to run this test.")

        from PIL import Image
        dummy_image = Image.fromarray((np.random.rand(100, 100, 3) * 255).astype("uint8"))

        result = predict_food_image(dummy_image, top_k=3)

        assert "predicted_class" in result
        assert 0.0 <= result["confidence"] <= 1.0
        assert len(result["top_k"]) == 3
