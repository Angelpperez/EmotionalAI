"""Synthetic fixtures test software only; no fixtures become experiment results."""
import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from config import CLASSES
from src.inspect_dataset import inspect_dataset
from src.metrics import classification_metrics, comparison_table
from src.openface_features import parse_openface_csv
from src.train_svc import internal_cv_splits
from src.utils import normalize_label
from src.vit_inference import verified_model_mapping


class ProtocolTests(unittest.TestCase):
    def test_model_mapping_is_not_positional(self):
        order = ["sad", "happy", "angry", "surprise", "fear", "disgust", "neutral"]
        mapping = verified_model_mapping(dict(enumerate(order)))
        self.assertEqual(mapping[0], "sadness")
        self.assertEqual(mapping[6], "neutral")
        with self.assertRaises(ValueError):
            verified_model_mapping({i: "neutral" for i in range(7)})
        with self.assertRaises(ValueError):
            verified_model_mapping({i: f"LABEL_{i}" for i in range(7)})

    def test_numeric_labels_require_explicit_verified_mapping(self):
        with self.assertRaises(ValueError):
            normalize_label("0")
        self.assertEqual(normalize_label("8", {"8": "happy"}), "happiness")
        self.assertIsNone(normalize_label("9", {"9": None}))

    def test_metrics_keep_seven_classes_and_correct_normalization(self):
        true = list(CLASSES)
        pred = list(CLASSES)
        pred[1] = "neutral"
        m = classification_metrics(true, pred)
        self.assertAlmostEqual(m["accuracy"], 6 / 7)
        self.assertAlmostEqual(m["balanced_accuracy"], 6 / 7)
        self.assertAlmostEqual(m["uar"], 6 / 7)
        self.assertEqual(m["confusion_matrix_normalized_true"][1][0], 1)
        self.assertEqual(m["per_class"]["fear"]["f1"], 0)
        self.assertEqual(len(m["confusion_matrix"]), 7)

    def test_missing_experiments_never_copy_paper_scores(self):
        with tempfile.TemporaryDirectory() as tmp:
            rows, _ = comparison_table({}, Path(tmp))
            self.assertEqual(len(rows), 5)
            self.assertTrue(all(r["OUR RESULT Accuracy (%)"] is None for r in rows))
            self.assertTrue(all(r["delta_accuracy (percentage points)"] is None for r in rows))

    def test_grouped_cv_has_no_duplicate_image_content(self):
        labels, hashes = [], []
        for c in CLASSES:
            for i in range(4):
                labels.extend([c, c])
                hashes.extend([f"{c}-{i}"] * 2)
        for fitting, tuning in internal_cv_splits(labels, hashes, 2):
            self.assertFalse({hashes[i] for i in fitting} & {hashes[i] for i in tuning})
            self.assertEqual({labels[i] for i in tuning}, set(CLASSES))

    def test_real_output_parser_rejects_failed_detection_and_selects_best_face(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "output.csv"
            path.write_text("face_id, confidence, success, AU01_r, AU01_c\n0,0.99,0,4,1\n1,0.7,1,2,1\n2,0.8,1,3,0\n", encoding="utf-8")
            values, info = parse_openface_csv(path)
            self.assertEqual(values["AU01_r"], 3)
            self.assertEqual(info["n_detected_faces"], 2)
            path.write_text("confidence, success, AU01_r\n0.9,0,0\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                parse_openface_csv(path)

    def test_openface_official_static_image_header_has_no_success(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "static.csv"
            path.write_text("face, confidence, AU01_r, AU01_c\n0,0.9,1.5,1\n", encoding="utf-8")
            values, info = parse_openface_csv(path)
            self.assertEqual(values["AU01_r"], 1.5)
            self.assertIn("landmark success unavailable", info["validity_rule"])
            path.write_text("face, confidence, AU01_r, AU01_c\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                parse_openface_csv(path)

    def inspect(self, root, **kwargs):
        with contextlib.redirect_stdout(io.StringIO()):
            return inspect_dataset(root, root / "audit", data_root=root / "AffectNet", **kwargs)

    def test_folder_inspection_records_source_ids_and_deviation(self):
        from PIL import Image
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for split, color in (("train", 10), ("validation", 20)):
                path = root / "AffectNet" / split / "happy" / "img.png"
                path.parent.mkdir(parents=True)
                Image.new("RGB", (4, 4), (color, 0, 0)).save(path)
            rows, report = self.inspect(root)
            self.assertEqual(report["status"], "completed")
            self.assertEqual(len(rows), 2)
            happy = next(r for r in report["counts"] if r["split"] == "validation" and r["class_name"] == "happiness")
            self.assertEqual(happy["class_id"], ["happy"])
            self.assertEqual(happy["n_images"], 1)
            self.assertEqual(report["validation_deltas"]["happiness"], -499)

    def test_content_leakage_blocks_model_execution(self):
        from PIL import Image
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for split in ("train", "validation"):
                path = root / "AffectNet" / split / "neutral" / "same.png"
                path.parent.mkdir(parents=True)
                Image.new("RGB", (4, 4), "red").save(path)
            rows, report = self.inspect(root)
            self.assertEqual(rows, [])
            self.assertEqual(report["status"], "blocked")
            self.assertIn("leakage", " ".join(report["errors"]))

    def test_unknown_numeric_folders_block_models(self):
        from PIL import Image
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / "AffectNet" / "validation" / "0" / "img.png"
            path.parent.mkdir(parents=True)
            Image.new("RGB", (4, 4)).save(path)
            rows, report = self.inspect(root)
            self.assertFalse(rows)
            self.assertIn("--label-map", " ".join(report["errors"]))

    def test_missing_dataset_has_unknown_counts(self):
        with tempfile.TemporaryDirectory() as tmp:
            rows, report = self.inspect(Path(tmp))
            self.assertEqual(rows, [])
            self.assertTrue(all(r["n_images"] is None for r in report["counts"]))


if __name__ == "__main__":
    unittest.main()
