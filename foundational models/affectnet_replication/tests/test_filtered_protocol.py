"""Software fixtures are temporary and never reported as experimental results."""
import contextlib
import csv
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image

from config import CLASSES, PROJECT_ROOT
from src.attention import attention_rollout
from src.error_analysis import confusion_rankings
from src.filtered_dataset import audit_image, inspect_filtered_dataset, select_evaluation
from src.filtered_experiment import assert_allowed_stage, load_predictions
from src.metrics import calibration_statistics, validate_prediction_arrays
from src.utils import archive_sources, read_json, sha256_file, write_csv


class FilteredProtocolTests(unittest.TestCase):
    def profile(self):
        return read_json(PROJECT_ROOT / "configs" / "affectnetfiltered_vit.json")

    def make_dataset(self, root):
        profile = self.profile()
        profile.pop("expected_evaluation_n", None)
        profile.pop("expected_evaluation_counts", None)
        for i, partition in enumerate(profile["partitions"]):
            for j, source in enumerate(profile["source_labels"]):
                path = root / partition / source / "example.png"
                path.parent.mkdir(parents=True)
                Image.new("RGB", (4, 4), (i*80+j*5, j, i)).save(path)
        return profile

    def test_observed_partition_names_and_contempt_exclusion(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "data"
            profile = self.make_dataset(root)
            with contextlib.redirect_stdout(io.StringIO()):
                selected, report = inspect_filtered_dataset(root, Path(tmp)/"out", profile, 2)
            self.assertEqual(report["n_images"], 16)
            self.assertEqual(len(selected), 7)
            self.assertEqual({r["true_label"] for r in selected}, set(CLASSES))
            self.assertNotIn("contempt", {r["source_label"] for r in selected})
            self.assertEqual(next(r for r in selected if r["source_label"] == "happy")["true_label"], "happiness")

    def test_pixel_duplicates_with_different_file_bytes_are_excluded(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            items = []
            for split, compression in (("train", 0), ("validation", 9)):
                path = root / split / "neutral" / "same.png"
                path.parent.mkdir(parents=True)
                Image.new("RGB", (8, 8), "red").save(path, compress_level=compression)
                items.append((path, root, split, split, "neutral", "neutral"))
            rows = [audit_image(item) for item in items]
            self.assertNotEqual(rows[0]["file_sha256"], rows[1]["file_sha256"])
            self.assertEqual(rows[0]["rgb_sha256"], rows[1]["rgb_sha256"])
            selected, exclusions, counts, summary = select_evaluation(rows)
            self.assertEqual(selected, [])
            self.assertEqual(summary["n_cross_partition_groups"], 1)
            self.assertEqual(counts["neutral"]["identical_content_in_train"], 1)

    def test_conflicts_with_excluded_contempt_still_quarantine_main_class(self):
        rows = [{"sample_id": "a", "relative_path": "train/contempt/a", "split": "train", "source_label": "contempt", "canonical_true_label": None,
                 "readable": True, "file_sha256": "same", "rgb_sha256": "pixels"},
                {"sample_id": "b", "relative_path": "val/happy/b", "split": "validation", "source_label": "happy", "canonical_true_label": "happiness",
                 "readable": True, "file_sha256": "same", "rgb_sha256": "pixels"}]
        selected, exclusions, counts, _ = select_evaluation(rows)
        self.assertFalse(selected)
        validation = next(r for r in exclusions if r["sample_id"] == "b")
        self.assertEqual(validation["primary_reason"], "conflicting_source_labels_all_eight_classes")
        self.assertIn("identical_content_in_train", validation["all_reasons"])
        self.assertEqual(counts["happiness"]["conflicting_source_labels_all_eight_classes"], 1)

    def test_same_split_duplicates_choose_lexicographic_representative(self):
        rows = [{"sample_id": x, "relative_path": f"val/neutral/{x}", "split": "validation", "source_label": "neutral", "canonical_true_label": "neutral",
                 "readable": True, "file_sha256": "same", "rgb_sha256": "pixels"} for x in ["z", "a"]]
        selected, exclusions, counts, _ = select_evaluation(rows)
        self.assertEqual([r["sample_id"] for r in selected], ["a"])
        self.assertEqual(counts["neutral"]["repeated_content_within_validation"], 1)

    def test_corruption_is_explicit_and_excluded(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / "bad.jpg"
            path.write_bytes(b"not a jpeg")
            row = audit_image((path, root, "val", "validation", "fear", "fear"))
            self.assertFalse(row["readable"])
            selected, failures, counts, _ = select_evaluation([row])
            self.assertEqual(selected, [])
            self.assertEqual(counts["fear"]["unreadable"], 1)
            self.assertTrue(failures[0]["error"])

    def test_openface_and_svc_are_forbidden_even_if_configuration_allows_them(self):
        profile = self.profile()
        profile["allowed_stages"] += ["openface", "svc"]
        profile["forbidden_stages"] = []
        for name in ("openface", "svc"):
            with self.assertRaisesRegex(ValueError, "forbidden"):
                assert_allowed_stage(profile, name)

    def test_cli_filtered_dispatch_never_calls_legacy_entry_points(self):
        from run_experiment import main
        with patch("src.filtered_experiment.run_filtered_experiment", return_value=0) as filtered:
            with patch("src.openface_features.extract_openface", side_effect=AssertionError("Forbidden OpenFace execution")) as of:
                with patch("src.train_svc.train_svcs", side_effect=AssertionError("Forbidden SVC execution")) as svc:
                    self.assertEqual(main(["--profile", "affectnetfiltered-vit", "--inspect-only"]), 0)
                    filtered.assert_called_once()
                    of.assert_not_called()
                    svc.assert_not_called()

    def test_calibration_bins_include_zero_one_and_exact_boundary(self):
        rows = [{"confidence": c, "true_label": "neutral", "predicted_label": p}
                for c, p in [(0, "neutral"), (.1, "fear"), (.2, "neutral"), (1, "neutral")]]
        result = calibration_statistics(rows)
        self.assertEqual([result["bins"][i]["n"] for i in [0, 1, 2, 9]], [1, 1, 1, 1])
        self.assertEqual(sum(b["n"] for b in result["bins"]), 4)
        self.assertAlmostEqual(result["ece"], (1+.1+.8)/4)
        self.assertIsNone(result["bins"][3]["accuracy"])

    def test_confusion_count_and_rate_rankings_are_distinct(self):
        count = np.zeros((7, 7), dtype=int)
        count[0, 0], count[0, 1], count[1, 1], count[1, 2] = 1000, 20, 1, 9
        rates = np.divide(count, count.sum(axis=1, keepdims=True), out=np.zeros((7,7), dtype=float), where=count.sum(axis=1,keepdims=True)>0)
        by_count, by_rate = confusion_rankings({"confusion_matrix": count.tolist(), "confusion_matrix_normalized_true": rates.tolist()})
        self.assertEqual(by_count[0]["true_label"], "neutral")
        self.assertEqual(by_rate[0]["true_label"], "fear")

    def test_stored_prediction_alignment_and_probability_validation(self):
        logits = np.zeros((1, 7), dtype=np.float32)
        probabilities = np.full((1, 7), 1/7, dtype=np.float32)
        row = {"sample_id": "a", "prediction_index": 0, "true_label": "fear", "canonical_true_label": "fear",
               "predicted_label": "neutral", "confidence": 1/7, "correct": False}
        validate_prediction_arrays([row], logits, probabilities)
        with self.assertRaises(ValueError):
            validate_prediction_arrays([{**row, "prediction_index": 1}], logits, probabilities)
        with self.assertRaises(ValueError):
            validate_prediction_arrays([row], logits, probabilities*.5)

    def test_prediction_csv_false_flag_is_parsed_as_boolean(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            row = {"prediction_index": 0, "confidence": .5, "correct": False, **{f"p_{c}": 1/7 for c in CLASSES}}
            write_csv(out/"predictions/predictions_vit.csv", [row], list(row))
            self.assertIs(load_predictions(out)[0]["correct"], False)

    def test_attention_rollout_residual_composition_and_invalid_weights(self):
        layer = np.full((1, 2, 5, 5), .2, dtype=np.float32)
        rollout = attention_rollout([layer, layer])
        self.assertEqual(rollout.shape, (2, 2))
        np.testing.assert_allclose(rollout, .15, atol=1e-7)
        with self.assertRaises(ValueError):
            attention_rollout([layer*.5])

    def test_dataset_distribution_drift_blocks_evaluation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)/"data"
            profile = self.make_dataset(root)
            profile["expected_evaluation_n"] = 8
            with contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaisesRegex(ValueError, "Dataset drift"):
                    inspect_filtered_dataset(root, Path(tmp)/"out", profile, 2)

    def test_frozen_source_archive_cannot_be_replaced_by_changed_code(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)/"project"; root.mkdir()
            source = root/"model.py"; source.write_text("original",encoding="utf-8")
            hashes = {"model.py": sha256_file(source)}
            archive_sources(root, Path(tmp)/"run", hashes)
            source.write_text("modified",encoding="utf-8")
            with self.assertRaises(ValueError):
                archive_sources(root, Path(tmp)/"run", {"model.py": sha256_file(source)})


if __name__ == "__main__":
    unittest.main()
