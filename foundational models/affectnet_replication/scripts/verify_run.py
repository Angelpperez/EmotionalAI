"""Validate saved research artifacts without running or training a model."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
from PIL import Image
from config import CLASSES
from src.filtered_experiment import load_predictions
from src.metrics import calibration_statistics, classification_metrics, validate_prediction_arrays
from src.utils import read_json, sha256_file, utc_now, write_json


def read_csv(path):
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def verify(output, verify_originals=True):
    output = Path(output).resolve()
    metadata = read_json(output / "experiment_metadata.json")
    inspection = read_json(output / "dataset_inspection.json")
    rows = load_predictions(output)
    frozen = read_csv(output / "manifests/evaluation_manifest.csv")
    all_images = read_csv(output / "manifests/all_images.csv")
    assert metadata["profile"] == "affectnetfiltered-vit"
    assert metadata["status"] == "completed"
    assert metadata["environment"]["python_311"]
    assert not any(k in metadata["stages"] for k in ("openface", "svc"))
    assert all(s["status"] == "completed" for s in metadata["stages"].values())
    assert len(rows) == inspection["evaluation_n"] == len(frozen)
    assert [r["sample_id"] for r in rows] == [r["sample_id"] for r in frozen]
    assert sha256_file(output / "manifests/evaluation_manifest.csv") == inspection["evaluation_manifest_sha256"]
    assert set(r["true_label"] for r in rows) == set(CLASSES)
    assert len({r["rgb_sha256"] for r in frozen}) == len(frozen)
    train_pixels = {r["rgb_sha256"] for r in all_images if r["split"] == "train" and r["rgb_sha256"]}
    train_files = {r["file_sha256"] for r in all_images if r["split"] == "train" and r["file_sha256"]}
    assert not train_pixels & {r["rgb_sha256"] for r in frozen}
    assert not train_files & {r["file_sha256"] for r in frozen}
    logits = np.load(output / "predictions/logits_vit.npy", allow_pickle=False)
    probs = np.load(output / "predictions/probabilities_vit.npy", allow_pickle=False)
    cls = np.load(output / "embeddings/vit_cls_embeddings.npy", allow_pickle=False)
    validate_prediction_arrays(rows, logits, probs, cls)
    parameters = np.load(output / "embeddings/classifier_parameters.npz", allow_pickle=False)
    reconstructed = cls @ parameters["weight"].T + parameters["bias"]
    assert np.allclose(reconstructed, logits, atol=1e-5, rtol=1e-4)
    cls_labels = read_csv(output / "embeddings/vit_cls_labels.csv")
    assert [r["sample_id"] for r in rows] == [r["sample_id"] for r in cls_labels]
    metrics = read_json(output / "metrics/metrics_vit.json")
    recalculated = classification_metrics([r["true_label"] for r in rows], [r["predicted_label"] for r in rows])
    for key in ("accuracy", "macro_f1", "balanced_accuracy", "precision_macro", "recall_macro", "uar"):
        assert abs(metrics[key]-recalculated[key]) < 1e-12, key
    assert np.array_equal(metrics["confusion_matrix"], recalculated["confusion_matrix"])
    assert int(np.asarray(metrics["confusion_matrix"]).sum()) == len(rows)
    assert dict(Counter(r["true_label"] for r in rows)) == inspection["evaluation_counts"]
    calibration = read_json(output / "metrics/calibration_vit.json")
    recomputed_calibration = calibration_statistics(rows)
    assert abs(calibration["ece"]-recomputed_calibration["ece"]) < 1e-12
    assert sum(b["n"] for b in calibration["bins"]) == len(rows)
    comparison = read_csv(output / "comparison.csv")
    assert len(comparison) == 2 and comparison[0]["Accuracy (%)"] == "43.9"
    assert all(comparison[0][k] == "N/R" for k in ("Macro-F1 (%)", "Balanced Accuracy (%)", "UAR (%)"))
    image_checks = []
    for entry in read_json(output / "figures/figure_manifest.json")["figures"]:
        path = output / "figures" / entry["file"]
        assert entry["status"] == "completed" and path.is_file(), entry["file"]
        with Image.open(path) as im:
            if path.suffix == ".png":
                assert all(abs(dpi-300) < .1 for dpi in im.info["dpi"])
                im.verify()
                image_checks.append({"file": path.name, "size": list(im.size), "dpi": 300})
            else:
                durations = []
                for index in range(im.n_frames):
                    im.seek(index); im.load(); durations.append(im.info.get("duration", 0))
                image_checks.append({"file": path.name, "size": list(im.size), "n_frames": im.n_frames, "durations_ms": durations})
                if path.name == "vit_inference_demo.gif":
                    assert im.size == (1280, 720) and all(d >= 3000 for d in durations)
    with Image.open(output / "figures/experiment_summary_16x9.png") as im:
        assert im.size == (3840, 2160)
    demo = read_json(output / "figures/vit_inference_demo.json")
    assert demo["n_correct"] >= 3 and demo["n_wrong"] >= 2
    predictions_by_id = {r["sample_id"]: r for r in rows}
    assert all(identifier in predictions_by_id for identifier in demo["sample_ids"])
    for selected in read_csv(output / "error_analysis/selected_examples.csv"):
        assert selected["sample_id"] in predictions_by_id
    for category in ("false_positives", "false_negatives", "correct_high_confidence"):
        for copied in (output / "error_analysis" / category).iterdir():
            assert sha256_file(copied) == predictions_by_id[copied.stem]["file_sha256"]
    model = read_json(output / "metrics/vit_model_metadata.json")
    for name, info in model["files"].items():
        assert sha256_file(Path(model["snapshot_path"])/name) == info["sha256"]
    source_root = Path(metadata.get("source_archive", "source_code"))
    if not source_root.is_absolute():
        source_root = output / source_root
    for name, digest in metadata["source_sha256"].items():
        assert sha256_file(source_root / name) == digest, name
    checked_originals = 0
    if verify_originals:
        for row in all_images:
            assert sha256_file(row["filepath"]) == row["file_sha256"], row["filepath"]
            checked_originals += 1
    return {"status": "passed", "verified_at_utc": utc_now(), "output_dir": str(output), "n_evaluated": len(rows),
            "original_images_unchanged": checked_originals, "n_png": len(list((output/"figures").glob("*.png"))),
            "n_gif": len(list((output/"figures").glob("*.gif"))), "figures": image_checks,
            "metrics_recomputed_from_saved_predictions": True, "cls_reconstruction_max_abs_error": float(np.max(np.abs(reconstructed-logits))),
            "model_files_and_executed_sources_verified": True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--skip-originals", action="store_true")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    result = verify(args.run, not args.skip_originals)
    if args.report:
        write_json(args.report, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
