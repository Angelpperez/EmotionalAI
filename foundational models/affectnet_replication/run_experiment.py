"""Inspect first, then run only stages with real data and required dependencies."""
from __future__ import annotations

import argparse
import csv
import os
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

from config import CLASSES, MODEL_ID, PAPER_URL, PROJECT_ROOT, REPOSITORY_ROOT, SEED
from src.inspect_dataset import inspect_dataset
from src.metrics import comparison_table, plot_comparisons, plot_pipeline, save_evaluation
from src.utils import (environment_metadata, read_json, seed_everything, sha256_file,
                       utc_now, write_json)
from src.vit_inference import run_vit


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--profile", choices=("affectnetfiltered-vit", "legacy-comparison"), default="affectnetfiltered-vit")
    p.add_argument("--config", type=Path)
    p.add_argument("--render-run", type=Path, help="Rebuild figures and reports from a completed frozen inference")
    p.add_argument("--audit-workers", type=int, default=6)
    p.add_argument("--data-root", type=Path)
    p.add_argument("--repository-root", type=Path, default=REPOSITORY_ROOT)
    p.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "outputs")
    p.add_argument("--manifest", type=Path, help="CSV with verified file, label and split columns")
    p.add_argument("--train-csv", type=Path)
    p.add_argument("--validation-csv", type=Path)
    p.add_argument("--label-map", type=Path, help="JSON: source label -> canonical label or null")
    p.add_argument("--filepath-column")
    p.add_argument("--label-column")
    p.add_argument("--split-column")
    p.add_argument("--openface-exe", type=Path)
    p.add_argument("--openface-timeout", type=int, default=120)
    p.add_argument("--batch-size", type=int)
    p.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    p.add_argument("--model-revision", help="Frozen commit required for the current profile")
    p.add_argument("--cv-folds", type=int, default=5)
    p.add_argument("--n-jobs", type=int, default=1)
    p.add_argument("--scale-features", action="store_true", help="Fit StandardScaler inside CV; default uses raw AUs")
    p.add_argument("--inspect-only", action="store_true")
    return p


def blocked(reason):
    return {"status": "blocked", "reason": reason, "metrics": None}


def safe_stage(name, output, function):
    try:
        return function()
    except Exception as exc:
        (output / f"error_{name}.txt").write_text(traceback.format_exc(), encoding="utf-8")
        return {"status": "failed", "reason": f"{type(exc).__name__}: {exc}", "metrics": None}


def common_subset_vit(output, results):
    from src.train_svc import SVC_SPECS
    if results["ViT-FER"]["status"] != "completed" or not (output / "openface_features.csv").is_file():
        return None
    with (output / "openface_features.csv").open(encoding="utf-8", newline="") as f:
        eligible = {r["filepath"] for r in csv.DictReader(f) if r["split"] == "validation"}
    with (output / "predictions" / "predictions_vit.csv").open(encoding="utf-8", newline="") as f:
        rows = [dict(r, confidence=float(r["confidence"])) for r in csv.DictReader(f) if r["filepath"] in eligible]
    if not rows:
        return None
    paired_paths = {r["filepath"] for r in rows}
    result = {"ViT-FER": {"status": "completed", "metrics": save_evaluation(rows, output, "vit_common", {
        "evaluation_population": "intersection of successful ViT and successful OpenFace validation images"})}}
    for name, slug, _, _ in SVC_SPECS:
        path = output / "predictions" / f"predictions_{slug}.csv"
        if results.get(name, {}).get("status") == "completed" and path.is_file():
            with path.open(encoding="utf-8", newline="") as f:
                paired = [dict(r, confidence=None) for r in csv.DictReader(f) if r["filepath"] in paired_paths]
            result[name] = {"status": "completed", "metrics": save_evaluation(paired, output, f"{slug}_common", {
                "evaluation_population": "intersection of successful ViT and successful OpenFace validation images"})}
    common = output / "common_subset"
    common.mkdir()
    comparison_table(result, common)
    return {"n_images": len(rows), "comparison": str(common / "comparison.csv")}


def final_report(output, inspection, results, openface, warnings):
    from src.train_svc import SVC_SPECS
    _, table = comparison_table(results, output)
    def metric_line(name):
        r = results[name]
        if not r.get("metrics"):
            return f"{name}: N/D ({r.get('reason', r['status'])})"
        m = r["metrics"]
        extra = f", best C={m['best_C']}" if "best_C" in m else ""
        return (f"{name}: Accuracy={100*m['accuracy']:.3f}%, Macro-F1={100*m['macro_f1']:.3f}%, "
                f"Balanced Accuracy={100*m['balanced_accuracy']:.3f}%, UAR={100*m['uar']:.3f}%{extra}")
    distribution = (output / "inspection" / "dataset_report.md").read_text(encoding="utf-8")
    figures = sorted(str(p.relative_to(output)) for p in (output / "figures").glob("*.png"))
    text = ("1. Dataset\n\n" + f"Root: {inspection.get('dataset_root') or 'AffectNet not found'}\n\n" + distribution +
            "\n2. ViT-FER\n\n" + metric_line("ViT-FER") +
            f"\n\n3. OpenFace\n\nStatus: {openface['status']}; attempted={openface.get('n_attempted', 0)}, "
            f"valid={openface.get('n_valid', 0)}, failed={openface.get('n_failed', 0)}.\n" +
            (openface.get("reason", "") + "\n") +
            "\n4. SVC\n\n" + "\n".join(metric_line(spec[0]) for spec in SVC_SPECS) +
            "\n\n5. OUR RESULT vs PAPER RESULT\n\n" + table +
            "\n6. Figures (PNG, 300 dpi)\n\n" + ("\n".join(figures) or "None generated") +
            "\n\n7. Methodological warnings\n\n" + "\n".join(dict.fromkeys(warnings)) + "\n")
    (output / "report.md").write_text(text, encoding="utf-8")
    return text


def main(argv=None):
    args = parser().parse_args(argv)
    if (args.batch_size is not None and args.batch_size < 1) or args.cv_folds < 2 or args.openface_timeout < 1 or args.n_jobs == 0 or args.audit_workers < 1:
        parser().error("batch size/timeout must be positive, cv-folds >= 2 and n-jobs != 0")
    if args.profile == "affectnetfiltered-vit":
        from src.filtered_experiment import run_filtered_experiment
        return run_filtered_experiment(args)
    if args.render_run:
        parser().error("--render-run is only available for affectnetfiltered-vit")
    args.batch_size = args.batch_size or 32
    args.model_revision = args.model_revision or "main"
    from src.openface_features import detect_openface, extract_openface
    from src.train_svc import SVC_SPECS, train_svcs
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = args.output_root.resolve() / f"run_{stamp}_{uuid4().hex[:8]}"
    output.mkdir(parents=True, exist_ok=False)
    for subdir in ("metrics", "figures", "predictions", "inspection"):
        (output / subdir).mkdir()
    metadata = {"started_at_utc": utc_now(), "status": "running", "seed": SEED,
                "arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                "reference_paper": PAPER_URL, "model_id": MODEL_ID,
                "scope": "conceptual replication; exact reproduction unverified",
                "output_dir": str(output)}
    write_json(output / "experiment_metadata.json", metadata)
    write_json(args.output_root.resolve() / "latest_run.json", {"output_dir": str(output)})
    print(f"Run: {output}", flush=True)
    # Persist the actual inspection and all deviations BEFORE calling a model.
    records, inspection = inspect_dataset(
        args.repository_root, output / "inspection", data_root=args.data_root, manifest=args.manifest,
        train_csv=args.train_csv, validation_csv=args.validation_csv, label_map_path=args.label_map,
        filepath_column=args.filepath_column, label_column=args.label_column, split_column=args.split_column)
    metadata["environment"] = environment_metadata()
    warnings = inspection["errors"] + inspection["warnings"]
    if not metadata["environment"]["python_311"]:
        warnings.append("Runtime is not Python 3.11; actual Python/package versions are recorded. Target requirements have not been validated in this runtime.")
    try:
        seed_everything()
    except (ImportError, OSError, RuntimeError) as exc:
        warnings.append(f"Could not initialize all requested random seeds: {exc}")
    metadata["source_sha256"] = {str(p.relative_to(PROJECT_ROOT)): sha256_file(p)
                                 for p in [PROJECT_ROOT / "config.py", PROJECT_ROOT / "run_experiment.py",
                                           PROJECT_ROOT / "requirements.txt", *sorted((PROJECT_ROOT / "src").glob("*.py"))]}
    inventory = read_json(output / "inspection" / "repository_inventory.json")
    detection = detect_openface(args.repository_root, args.openface_exe, inventory)
    write_json(output / "openface_detection.json", detection)
    metadata["dataset"] = inspection
    metadata["openface_detection"] = detection
    write_json(output / "experiment_metadata.json", metadata)
    reason = "Inspection-only run." if args.inspect_only else "; ".join(inspection["errors"])
    results = {name: blocked(reason or "Prerequisite unavailable.") for name in [s[0] for s in SVC_SPECS] + ["ViT-FER"]}
    figure_stage = safe_stage("pipeline_figure", output, lambda: (plot_pipeline(output / "figures" / "pipeline_comparison.png") or {"status": "completed"}))
    if figure_stage["status"] != "completed":
        warnings.append("Pipeline figure could not be generated: " + figure_stage["reason"])
    if records and not args.inspect_only:
        results["ViT-FER"] = safe_stage("vit", output, lambda: run_vit(
            records, output, args.batch_size, args.device, args.model_revision))
        openface = safe_stage("openface", output, lambda: extract_openface(records, output, detection, args.openface_timeout))
    else:
        openface = {**blocked(reason or "No verified dataset images."), "n_attempted": 0, "n_valid": 0, "n_failed": 0}
        if not detection["available"]:
            openface["reason"] += " " + detection["reason"]
    write_json(output / "metrics" / "openface_summary.json", openface)
    if openface["status"] == "completed" and not args.inspect_only:
        svc = safe_stage("svc", output, lambda: {"status": "completed", "models": train_svcs(
            output, openface["feature_columns"], args.cv_folds, args.n_jobs, args.scale_features)})
        if "models" in svc:
            results.update(svc["models"])
        else:
            results.update({s[0]: svc for s in SVC_SPECS})
    else:
        results.update({s[0]: blocked(openface.get("reason", "OpenFace did not complete.")) for s in SVC_SPECS})
    common = safe_stage("common_subset", output, lambda: {"status": "completed", "result": common_subset_vit(output, results)})
    if common["status"] != "completed":
        warnings.append("Could not evaluate common subset: " + common["reason"])
    rows, _ = comparison_table(results, output)
    comparisons = safe_stage("comparison_figures", output, lambda: {"status": "completed", "paths": plot_comparisons(rows, output)})
    if comparisons["status"] != "completed":
        warnings.append("Comparison figures could not be generated: " + comparisons["reason"])
    warnings.extend(openface.get("warnings", []))
    warnings.extend(f"{name}: {r['reason']}" for name, r in results.items() if r.get("reason"))
    warnings.extend(["Paper reference values never substitute missing experimental results.",
                     "No exact reproduction claimed: dataset identity, historical model revision, OpenFace version and preprocessing need verification.",
                     "Main table reports ViT on readable validation and SVC on OpenFace-success validation; use common_subset/comparison.csv for a paired comparison when available.",
                     "The paper does not fully specify scaling, RBF gamma or internal selection; this implementation records raw AU/default gamma=scale and train-only grouped CV."])
    completed = sum(r["status"] == "completed" for r in results.values())
    failed = any(r["status"] == "failed" for r in results.values()) or openface["status"] == "failed"
    status = "completed" if completed == 5 else "partial" if completed else "failed" if failed else "blocked"
    metadata.update(status=status, finished_at_utc=utc_now(), results=results, openface=openface,
                    warnings=list(dict.fromkeys(warnings)), common_subset=common,
                    figures={"pipeline": figure_stage, "comparisons": comparisons},
                    image_accounting={"selected": inspection["n_selected"], "valid": inspection["n_valid"],
                                      "failed_inspection": inspection["n_failed"],
                                      "vit_processed": results["ViT-FER"].get("n_processed", 0),
                                      "vit_failed": results["ViT-FER"].get("n_failed", 0),
                                      "openface_attempted": openface.get("n_attempted", 0),
                                      "openface_valid": openface.get("n_valid", 0),
                                      "openface_failed": openface.get("n_failed", 0)})
    write_json(output / "experiment_metadata.json", metadata)
    report = final_report(output, inspection, results, openface, warnings)
    print(report, flush=True)
    print(f"Artifacts: {output}", flush=True)
    return 0 if status == "completed" or args.inspect_only else 3 if failed else 2


if __name__ == "__main__":
    sys.exit(main())
