"""Orchestration of the ViT-only protocol. No OpenFace/SVC modules are imported here."""
from __future__ import annotations

import contextlib
import csv
import os
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from config import CLASSES, PROJECT_ROOT, SEED
from src.utils import (archive_sources, artifact_manifest, environment_metadata, freeze_environment, read_json,
                       markdown_local_link, seed_everything, sha256_file, source_snapshot, utc_now, write_csv, write_json)

SAFE_STAGES = {"inspect", "vit", "metrics", "figures", "embeddings", "attention", "calibration"}


def assert_allowed_stage(profile, stage):
    if stage not in SAFE_STAGES or stage not in profile["allowed_stages"] or stage in profile["forbidden_stages"]:
        raise ValueError(f"Stage {stage!r} is forbidden by profile affectnetfiltered-vit.")


def load_predictions(output):
    with (output / "predictions" / "predictions_vit.csv").open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        row["prediction_index"] = int(row["prediction_index"])
        row["confidence"] = float(row["confidence"])
        row["correct"] = row["correct"] == "True"
        for c in CLASSES:
            row[f"p_{c}"] = float(row[f"p_{c}"])
    return rows


class Tee:
    def __init__(self, stream, log):
        self.stream, self.log = stream, log
    def write(self, value):
        self.stream.write(value); self.log.write(value)
        return len(value)
    def flush(self):
        self.stream.flush(); self.log.flush()
    def isatty(self):
        return False


def write_comparison(metrics, output):
    fields = ["Experiment", "Dataset", "Model", "N", "Accuracy (%)", "Macro-F1 (%)", "Balanced Accuracy (%)", "UAR (%)", "delta_accuracy_pp"]
    rows = [{"Experiment": "PAPER REFERENCE: Schuller et al.", "Dataset": "Original AffectNet, paper subset", "Model": "ViT-FER", "N": 3500,
             "Accuracy (%)": 43.9, "Macro-F1 (%)": "N/R", "Balanced Accuracy (%)": "N/R", "UAR (%)": "N/R", "delta_accuracy_pp": ""},
            {"Experiment": "OUR RESULT: Technical validation", "Dataset": "AffectNetFiltered, provided validation after exclusions", "Model": "ViT-FER",
             "N": metrics["n_images"], "Accuracy (%)": 100*metrics["accuracy"], "Macro-F1 (%)": 100*metrics["macro_f1"],
             "Balanced Accuracy (%)": 100*metrics["balanced_accuracy"], "UAR (%)": 100*metrics["uar"],
             "delta_accuracy_pp": metrics["delta_accuracy_pp"]}]
    write_csv(output / "comparison.csv", rows, fields)
    return rows


def report_results(output, metadata, metrics, summary):
    inspection = read_json(output / "dataset_inspection.json")
    info = read_json(output / "metrics" / "vit_model_metadata.json")
    raw_validation = {r["canonical_label"]: r["n_images"] for r in inspection["counts"]
                      if r["split"] == "validation" and r["canonical_label"] is not None}
    lines = ["# Technical validation on AffectNetFiltered", "", f"Run: `{output.name}`", "",
             "1. Dataset path", "", inspection["dataset_root"], "", "2. Class distribution", "",
             "Canonical label | Source folder | Provided validation | Evaluated", "--- | --- | ---: | ---:"]
    for c in CLASSES:
        source = next(k for k, v in inspection["source_label_mapping"].items() if v == c)
        lines.append(f"{c} | {source} | {raw_validation[c]} | {inspection['evaluation_counts'][c]}")
    lines += ["", f"3. Evaluated N: **{metrics['n_images']}**. Unreadable in full audit: {inspection['n_unreadable']}.", "",
              "4. Model class mapping", "", str(info["id2label"]), "", str(info["canonical_mapping"]), "",
              "5. ViT-FER", "", "Metric | OUR RESULT", "--- | ---:"]
    for name, key in (("Accuracy", "accuracy"), ("Macro-F1", "macro_f1"), ("Balanced Accuracy", "balanced_accuracy"), ("UAR", "uar")):
        lines.append(f"{name} | {100*metrics[key]:.4f}%")
    lines += ["", "6. Per-class F1", "", "Class | Precision | Recall | F1 | Support", "--- | ---: | ---: | ---: | ---:"]
    for c in CLASSES:
        m = metrics["per_class"][c]
        lines.append(f"{c} | {100*m['precision']:.3f}% | {100*m['recall']:.3f}% | {100*m['f1']:.3f}% | {m['support']}")
    lines += ["", "7. Largest confusions by count", ""]
    for p in summary["top_two_confusions_by_count"]:
        lines.append(f"- TRUE {p['true_label']} → PRED {p['predicted_label']}: {p['n']}/{p['support']} ({100*p['rate']:.3f}% of the true class)")
    lines += ["", "8. PAPER REFERENCE Accuracy: **43.9%**, original AffectNet subset; other comparison metrics N/R.",
              "", f"9. OUR Accuracy: **{100*metrics['accuracy']:.4f}%**, AffectNetFiltered.",
              "", f"10. Delta Accuracy: **{metrics['delta_accuracy_pp']:+.4f} percentage points**.",
              "", "Different dataset preprocessing; not a direct replication. Delta is descriptive and does not imply superiority or degradation of the method.",
              "", "11. Absolute artifact paths", ""]
    selected_paths = [output / "metrics" / "metrics_vit.json", output / "metrics" / "classification_report_vit.csv",
                      output / "predictions" / "predictions_vit.csv", output / "comparison.csv", output / "ppt_results.txt"]
    selected_paths += sorted((output / "figures").glob("*.png")) + sorted((output / "figures").glob("*.gif"))
    lines += [f"- {markdown_local_link(p.name, p, output)} — `{p.resolve()}`" for p in selected_paths if p.is_file()]
    warnings = ["Technical validation on preprocessed grayscale images; not original AffectNet evaluation.",
                "Provided validation partition is not established to be the original official split.",
                "Conflicting labels and overlapping/duplicate content were excluded before inference; selection changes the evaluated population.",
                "CLAHE/Gaussian Blur parameters, exact download version and subject independence are unverified.",
                "Softmax confidence is uncalibrated; representation/attention figures do not establish psychological state or causal explanations.",
                "OpenFace and SVC were neither installed nor executed by this profile; awaiting investigator review."]
    warnings += [f"{name}: {stage.get('reason')}" for name, stage in metadata["stages"].items() if stage["status"] in {"failed", "skipped"}]
    lines += ["", "12. Methodological warnings and errors", ""] + [f"- {warning}" for warning in warnings]
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    worst = summary["worst_two_f1"][0]
    best = ", ".join(summary["best_f1"])
    biggest = summary["top_two_confusions_by_count"][0] if summary["top_two_confusions_by_count"] else None
    ppt = ["Dataset: AffectNetFiltered (provided validation, audited exclusions)", "Model: trpakov/vit-face-expression",
           f"N: {metrics['n_images']}", "Classes: " + ", ".join(CLASSES), "Fine-tuning: None (off-the-shelf)",
           f"Accuracy: {100*metrics['accuracy']:.3f}%", f"Macro-F1: {100*metrics['macro_f1']:.3f}%",
           f"Balanced Accuracy: {100*metrics['balanced_accuracy']:.3f}%", f"UAR: {100*metrics['uar']:.3f}%",
           f"Worst class (F1): {worst}", f"Best class (F1): {best}",
           "Largest confusion (count): " + (f"{biggest['true_label']} -> {biggest['predicted_label']}, {biggest['n']} images, {100*biggest['rate']:.2f}% of true class" if biggest else "None"),
           "Paper reference: ViT-FER, original AffectNet subset, Accuracy=43.9%",
           f"Delta Accuracy: {metrics['delta_accuracy_pp']:+.3f} percentage points (descriptive only)",
           "Methodological limitation: Different dataset preprocessing; not a direct replication."]
    (output / "ppt_results.txt").write_text("\n".join(ppt) + "\n", encoding="utf-8")
    write_resources_index(output, metrics)
    return warnings


def write_resources_index(output, metrics):
    """Keep clickable relative links separate from absolute paths intended for copying."""
    resources = [("Informe completo", "report.md"), ("Texto para PowerPoint", "ppt_results.txt"),
                 ("Métricas JSON", "metrics/metrics_vit.json"),
                 ("Classification report CSV", "metrics/classification_report_vit.csv"),
                 ("Predicciones por imagen", "predictions/predictions_vit.csv"),
                 ("Comparación contextual", "comparison.csv"),
                 ("Inspección del dataset", "dataset_inspection.json"),
                 ("Embeddings CLS", "embeddings/vit_cls_embeddings.npy")]
    verification = output / "logs" / "verification.json"
    if verification.is_file():
        resources.append(("Verificación independiente", "logs/verification.json"))
    lines = ["# Recursos de presentación — AffectNetFiltered", "", f"Run: `{output.name}`", "",
             f"N={metrics['n_images']} | Accuracy={100*metrics['accuracy']:.4f}% | Macro-F1={100*metrics['macro_f1']:.4f}% | Balanced Accuracy/UAR={100*metrics['uar']:.4f}%.",
             "", "Technical validation — different dataset preprocessing; not a direct replication.", "",
             "Los enlaces son relativos a este documento. Las rutas absolutas se incluyen para copiar y pegar en Ctrl+P o el Explorador.",
             "", "## Resultados y evidencia", ""]
    for label, relative in resources:
        path = output / relative
        if path.is_file():
            lines.append(f"- {markdown_local_link(label, path, output)} — `{path.resolve()}`")
    lines += ["", "## Figuras y animaciones", "", "Archivo | Ruta absoluta para copiar", "--- | ---"]
    for path in sorted((output / "figures").iterdir()):
        if path.suffix.lower() in {".png", ".gif"}:
            lines.append(f"{markdown_local_link(path.name, path, output)} | `{path.resolve()}`")
    lines += ["", "PNG: 300 dpi. El resumen es 16:9. Los JSON y arrays conservan fracciones; las figuras indican porcentajes.",
              "", "La atención es interpretativa, no causal. La referencia del paper no es una comparación experimental directa."]
    (output / "RECURSOS_PRESENTACION.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def postprocess(output, profile, metadata, stage, runtime=None):
    import numpy as np
    from src.metrics import calibration_statistics, save_filtered_metrics, validate_prediction_arrays
    from src.error_analysis import analyze_errors
    from src.embeddings import project_embeddings
    from src.visualizations import (core_figure_tasks, confidence_figure, reliability_figure,
                                    embedding_figure, attention_figure, confusion_gif)
    from src.attention import extract_attention

    rows = load_predictions(output)
    inspection = read_json(output / "dataset_inspection.json")
    logits = np.load(output / "predictions" / "logits_vit.npy", allow_pickle=False)
    probabilities = np.load(output / "predictions" / "probabilities_vit.npy", allow_pickle=False)
    cls = np.load(output / "embeddings" / "vit_cls_embeddings.npy", allow_pickle=False)
    validate_prediction_arrays(rows, logits, probabilities, cls)
    with (output / "manifests" / "evaluation_manifest.csv").open(encoding="utf-8", newline="") as f:
        frozen = list(csv.DictReader(f))
    if [r["sample_id"] for r in rows] != [r["sample_id"] for r in frozen]:
        raise ValueError("Predictions do not cover the frozen manifest in order.")
    if len(rows) != inspection["evaluation_n"]:
        raise ValueError("Incomplete evaluation.")
    metrics = stage("metrics", lambda: save_filtered_metrics(rows, output, inspection), "metrics")
    panels, summary = stage("error_analysis", lambda: analyze_errors(rows, metrics, output), "metrics")
    stage("comparison", lambda: write_comparison(metrics, output), "metrics")
    figure_entries = []
    for name, callback in core_figure_tasks(rows, metrics, panels, output):
        stage(f"figure_{name}", callback, "figures", required=False)
        figure_entries.append({"file": name, "source": "frozen manifest/predictions; metrics/metrics_vit.json", "status": metadata["stages"][f"figure_{name}"]["status"]})
    calibration = stage("calibration", lambda: calibration_statistics(rows), "calibration")
    write_json(output / "metrics" / "calibration_vit.json", calibration)
    write_csv(output / "metrics" / "reliability_bins.csv", calibration["bins"], list(calibration["bins"][0]))
    for name, callback in (("confidence_correct_vs_wrong.png", lambda: confidence_figure(rows, calibration, output)),
                           ("reliability_diagram.png", lambda: reliability_figure(calibration, output))):
        stage(f"figure_{name}", callback, "figures", required=False)
        figure_entries.append({"file": name, "source": "metrics/calibration_vit.json", "status": metadata["stages"][f"figure_{name}"]["status"]})
    projection = stage("embedding_projection", lambda: project_embeddings(rows, output), "embeddings", required=False)
    if projection:
        coordinates, projection_info = projection
        stage("figure_vit_embedding_projection.png", lambda: embedding_figure(rows, coordinates, projection_info, output), "figures", required=False)
    figure_entries.append({"file": "vit_embedding_projection.png", "source": "embeddings/vit_cls_embeddings.npy", "status": metadata["stages"].get("figure_vit_embedding_projection.png", {"status": "skipped"})["status"]})

    def attention_task():
        if runtime is not None:
            return extract_attention(runtime, panels["attention"], output)
        maps = {}
        for row in panels["attention"]:
            path = output / "attention" / f"{row['sample_id']}.npy"
            if path.is_file():
                maps[row["sample_id"]] = np.load(path, allow_pickle=False)
        if len(maps) == len(panels["attention"]) and maps:
            return maps
        from transformers import AutoImageProcessor, AutoModelForImageClassification
        info = read_json(output / "metrics" / "vit_model_metadata.json")
        snapshot = info["snapshot_path"]
        processor = AutoImageProcessor.from_pretrained(snapshot, local_files_only=True, use_fast=False)
        model = AutoModelForImageClassification.from_pretrained(snapshot, local_files_only=True, use_safetensors=True, attn_implementation="eager")
        model.to(info["device"]).eval().requires_grad_(False)
        return extract_attention({"model": model, "processor": processor, "device": info["device"],
                                  "canonical_ids": info["model_ids_in_canonical_order"]}, panels["attention"], output)

    maps = stage("attention", attention_task, "attention", required=False)
    if maps:
        stage("figure_vit_attention_examples.png", lambda: attention_figure(panels["attention"], maps, output), "figures", required=False)
    figure_entries.append({"file": "vit_attention_examples.png", "source": "attention/*.npy; real model attention weights", "status": metadata["stages"].get("figure_vit_attention_examples.png", {"status": "skipped"})["status"]})
    stage("figure_confusion_matrix_build.gif", lambda: confusion_gif(rows, metrics, output), "figures", required=False)
    figure_entries.append({"file": "confusion_matrix_build.gif", "source": "predictions in frozen manifest order", "status": metadata["stages"]["figure_confusion_matrix_build.gif"]["status"]})
    write_json(output / "figures" / "figure_manifest.json", {"source_run": output.name, "class_order": list(CLASSES),
        "selection_manifest": str(output / "error_analysis" / "selected_examples.csv"), "png_dpi": 300, "figures": figure_entries})
    metadata["metrics"] = metrics
    metadata["error_summary"] = summary
    metadata["warnings"] = report_results(output, metadata, metrics, summary)
    return metrics


def run_filtered_experiment(args):
    config_path = args.config or PROJECT_ROOT / "configs" / "affectnetfiltered_vit.json"
    profile = read_json(config_path)
    if profile["profile"] != "affectnetfiltered-vit" or set(profile["allowed_stages"]) - SAFE_STAGES:
        raise ValueError("Invalid or unsafe profile stage list.")
    if args.data_root is not None:
        profile["dataset_root"] = str(args.data_root.resolve())
    if args.model_revision is not None and args.model_revision != profile["model_revision"]:
        raise ValueError("Model revision override requires a separate protocol profile.")
    if sys.version_info[:2] != (3, 11) and not args.inspect_only:
        raise RuntimeError("This profile requires Python 3.11. Use .venv311/Scripts/python.exe.")
    if args.render_run:
        output = args.render_run.resolve()
        metadata = read_json(output / "experiment_metadata.json")
        if metadata.get("profile") != "affectnetfiltered-vit":
            raise ValueError("Rendering requires an existing run of this profile.")
        # Rendering always uses the frozen profile, not potentially changed defaults.
        profile = read_json(output / "protocol_snapshot.json")["profile"]
        if read_json(output / "logs" / "progress.json")["status"] != "completed":
            raise ValueError("Cannot render an incomplete inference run.")
        current_hashes = source_snapshot(PROJECT_ROOT)
        subdirectory = "source_code/postprocess_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid4().hex[:8]
        archive_sources(PROJECT_ROOT, output, current_hashes, subdirectory)
        metadata.setdefault("postprocess_history", []).append({"started_at_utc": utc_now(), "source_sha256": current_hashes,
                                                               "source_archive": subdirectory})
        metadata.setdefault("source_archive", "source_code")
    else:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        output = args.output_root.resolve() / f"run_{stamp}_{uuid4().hex[:8]}"
        output.mkdir(parents=True, exist_ok=False)
        for name in ("manifests", "metrics", "predictions", "embeddings", "figures", "error_analysis", "logs"):
            (output / name).mkdir()
        metadata = {"profile": profile["profile"], "experiment": profile["experiment_name"], "status": "running",
                    "started_at_utc": utc_now(), "output_dir": str(output), "stages": {}, "seed": SEED,
                    "source_sha256": source_snapshot(PROJECT_ROOT), "command": sys.argv,
                    "openface": "not executed; forbidden by profile", "svc": "not executed; forbidden by profile"}
        metadata["source_archive"] = archive_sources(PROJECT_ROOT, output, metadata["source_sha256"])
        git_status = subprocess.run(["git", "status", "--short"], cwd=PROJECT_ROOT, capture_output=True, text=True, check=False)
        (output / "logs" / "git_status.txt").write_text(git_status.stdout, encoding="utf-8")
        write_json(args.output_root.resolve() / "latest_run.json", {"output_dir": str(output)})
    write_json(output / "experiment_metadata.json", metadata)

    def stage(name, callback, stage_type, required=True):
        assert_allowed_stage(profile, stage_type)
        started = time.perf_counter()
        metadata["stages"][name] = {"status": "running", "started_at_utc": utc_now()}
        write_json(output / "experiment_metadata.json", metadata)
        print(f"Stage: {name}", flush=True)
        try:
            result = callback()
            metadata["stages"][name].update(status="completed", seconds=time.perf_counter()-started)
            return result
        except Exception as exc:
            metadata["stages"][name].update(status="failed", reason=f"{type(exc).__name__}: {exc}", seconds=time.perf_counter()-started)
            (output / "logs" / f"error_{name}.txt").write_text(traceback.format_exc(), encoding="utf-8")
            print(f"Failed {name}: {type(exc).__name__}: {exc}", flush=True)
            if required:
                raise
            return None
        finally:
            write_json(output / "experiment_metadata.json", metadata)

    metadata["status"] = "running"
    with (output / "logs" / "execution.log").open("a", encoding="utf-8") as log:
        with contextlib.redirect_stdout(Tee(sys.stdout, log)), contextlib.redirect_stderr(Tee(sys.stderr, log)):
            print(f"Run: {output}", flush=True)
            try:
                if args.render_run:
                    import torch
                    seed_everything(profile["seed"])
                    torch.backends.cuda.matmul.allow_tf32 = False
                    torch.backends.cudnn.allow_tf32 = False
                    metadata["postprocess_history"][-1]["environment"] = environment_metadata()
                    metrics = postprocess(output, profile, metadata, stage)
                else:
                    from src.filtered_dataset import inspect_filtered_dataset
                    records, inspection = stage("inspection", lambda: inspect_filtered_dataset(profile["dataset_root"], output, profile, args.audit_workers), "inspect")
                    metadata["dataset"] = {"n_images": inspection["n_images"], "evaluation_n": inspection["evaluation_n"], "n_unreadable": inspection["n_unreadable"]}
                    write_json(output / "protocol_snapshot.json", {"profile": profile, "frozen_at_utc": utc_now(),
                        "evaluation_n": len(records), "evaluation_manifest_sha256": inspection["evaluation_manifest_sha256"],
                        "plan_sha256": sha256_file(PROJECT_ROOT / "PLAN_EXPERIMENTO_DOCTORAL.md")})
                    if args.inspect_only:
                        metadata["status"] = "inspection_completed"
                        metadata["environment"] = environment_metadata()
                        return 0
                    import torch
                    seed_everything(profile["seed"])
                    torch.backends.cuda.matmul.allow_tf32 = False
                    torch.backends.cudnn.allow_tf32 = False
                    metadata["environment"] = environment_metadata()
                    freeze_environment(output / "requirements_freeze.txt")
                    freeze_environment(PROJECT_ROOT / "requirements_freeze.txt")
                    from src.vit_inference import run_filtered_vit
                    vit, runtime = stage("vit", lambda: run_filtered_vit(records, output, profile, args.device, args.batch_size), "vit")
                    metadata["vit"] = vit
                    metrics = postprocess(output, profile, metadata, stage, runtime)
                required_figures = [name for name in metadata["stages"] if name.startswith("figure_") and not any(optional in name for optional in ("embedding", "attention", "confidence", "reliability", "confusion_matrix_build"))]
                incomplete = [name for name in required_figures if metadata["stages"][name]["status"] != "completed"]
                metadata["status"] = "partial" if incomplete else "completed"
                print(f"OUR RESULT: Accuracy={100*metrics['accuracy']:.4f}%, Macro-F1={100*metrics['macro_f1']:.4f}%, Balanced Accuracy={100*metrics['balanced_accuracy']:.4f}%, UAR={100*metrics['uar']:.4f}%", flush=True)
                print(f"Report: {output / 'report.md'}", flush=True)
                return 3 if incomplete else 0
            except Exception as exc:
                metadata["status"] = "failed"
                metadata["error"] = f"{type(exc).__name__}: {exc}"
                print(traceback.format_exc(), flush=True)
                return 3
            finally:
                metadata["finished_at_utc"] = utc_now()
                write_json(output / "experiment_metadata.json", metadata)
                log.flush()
                artifact_manifest(output)
