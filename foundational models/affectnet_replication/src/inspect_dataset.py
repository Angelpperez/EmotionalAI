"""Inspect actual files before any model load; ambiguous encodings fail closed."""
from __future__ import annotations

import csv
from collections import Counter, defaultdict
from pathlib import Path

from config import (CLASSES, IMAGE_EXTENSIONS, PAPER_TRAIN_COUNTS,
                    PAPER_VALIDATION_PER_CLASS, SPLIT_ALIASES)
from src.utils import (fingerprint_rows, normalize_label, read_json, sha256_file,
                       walk_files, write_csv, write_json)


def repository_inventory(root):
    root = Path(root).resolve()
    image_counts = Counter()
    candidates, executables, archives = set(), [], []
    for path in walk_files(root):
        relative = path.relative_to(root)
        if path.suffix.lower() in IMAGE_EXTENSIONS:
            image_counts[str(relative.parent)] += 1
            for parent in path.parents:
                if parent == root:
                    break
                if "affectnet" in parent.name.lower() and parent.name != "affectnet_replication":
                    candidates.add(str(parent))
        if path.name.lower() in {"facelandmarkimg.exe", "featureextraction.exe"}:
            executables.append(str(path))
        if path.suffix.lower() in {".zip", ".rar", ".7z", ".tar", ".gz"}:
            archives.append(str(relative))
    return {"repository_root": str(root), "image_directories": dict(sorted(image_counts.items())),
            "affectnet_candidates": sorted(candidates), "openface_executables": executables,
            "archives": archives, "scope": "repository, including ignored files; virtual environments/build trees excluded"}


def _choose_column(fieldnames, requested, candidates, role):
    if requested:
        if requested not in fieldnames:
            raise ValueError(f"{role} column {requested!r} not present: {fieldnames}")
        return requested
    found = [c for c in fieldnames if c.strip().lower() in candidates]
    if len(found) != 1:
        raise ValueError(f"Ambiguous {role} column; found {found} in {fieldnames}. Set its CLI argument.")
    return found[0]


def read_annotations(path, root, options, fixed_split=None):
    with Path(path).open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        names = reader.fieldnames or []
        filepath = _choose_column(names, options.get("filepath_column"),
                                  {"filepath", "path", "file", "subdirectory_filepath"}, "filepath")
        label = _choose_column(names, options.get("label_column"),
                              {"label", "expression", "class", "emotion"}, "label")
        split = None if fixed_split else _choose_column(names, options.get("split_column"),
                                                         {"split", "partition", "subset"}, "split")
        rows = []
        for number, row in enumerate(reader, 2):
            raw_path = row.get(filepath)
            raw_label = row.get(label)
            if not raw_path or raw_label is None:
                raise ValueError(f"Missing filepath or label at {path}:{number}")
            partition = fixed_split or SPLIT_ALIASES.get(str(row.get(split, "")).strip().lower())
            if partition is None:
                raise ValueError(f"Unknown split at {path}:{number}. Only train/validation are used.")
            image = Path(raw_path.strip().replace("\\", "/"))
            if not image.is_absolute():
                image = root / image
            rows.append({"filepath": str(image.resolve()), "raw_label": raw_label.strip(), "split": partition})
    return rows, {"source": str(Path(path).resolve()), "columns":
                  {"filepath": filepath, "label": label, "split": split}, "fixed_split": fixed_split,
                  "sha256": sha256_file(path)}


def read_folders(root):
    rows, split_dirs = [], {}
    for path in sorted(root.iterdir()):
        if path.is_dir() and path.name.lower() in SPLIT_ALIASES:
            split = SPLIT_ALIASES[path.name.lower()]
            if split in split_dirs:
                raise ValueError(f"Multiple directories resolve to {split}: {split_dirs[split]}, {path}")
            split_dirs[split] = str(path)
            for image in walk_files(path):
                if image.suffix.lower() not in IMAGE_EXTENSIONS:
                    continue
                relative = image.relative_to(path)
                if len(relative.parts) < 2:
                    raise ValueError(f"Unlabelled image {image}; use a verified CSV manifest.")
                rows.append({"filepath": str(image.resolve()), "raw_label": relative.parts[0], "split": split})
    if not rows:
        raise ValueError("No verified class-folder layout found. Inspect the structure and provide --manifest or paired CSVs.")
    return rows, {"layout": "class folders", "split_directories": split_dirs,
                  "label_source": "first directory under each detected split"}


def inspect_dataset(repository_root, output_dir, data_root=None, manifest=None,
                    train_csv=None, validation_csv=None, label_map_path=None, **options):
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    inventory = repository_inventory(repository_root)
    write_json(output / "repository_inventory.json", inventory)
    report = {"status": "blocked", "dataset_root": None, "canonical_class_order": list(CLASSES),
              "class_id_note": "canonical_class_id is local; class_id contains observed source values",
              "warnings": [], "errors": [], "counts": [], "n_selected": 0, "n_valid": 0,
              "n_failed": 0, "n_excluded": 0, "validation_equivalent_counts": False,
              "identity_with_paper_subset": "unverified"}
    root = Path(data_root).resolve() if data_root else None
    if root is None and len(inventory["affectnet_candidates"]) == 1:
        root = Path(inventory["affectnet_candidates"][0])
    if root is None:
        report["errors"].append("AffectNet not found unambiguously in the repository; supply --data-root.")
    elif not root.is_dir():
        report["errors"].append(f"Dataset root does not exist: {root}")
    else:
        report["dataset_root"] = str(root)
        report["structure_sample"] = [str(p.relative_to(root)) for p in list(root.iterdir())[:40]]
    if report["errors"]:
        report["counts"] = [{"class_name": c, "class_id": None, "canonical_class_id": i,
                             "split": "validation", "n_images": None, "n_valid": None}
                            for i, c in enumerate(CLASSES)]
        report["warnings"].append("No verified AffectNet images: the reference requires 500 per class, 3500 total. Counts are unavailable, not zero.")
        _save_inspection(output, report, [], [])
        return [], report

    rows, failures, exclusions = [], [], []
    try:
        label_map = read_json(label_map_path) if label_map_path else None
        if label_map is not None:
            if not isinstance(label_map, dict):
                raise ValueError("--label-map must contain a JSON object of source label -> canonical label/null.")
            report["explicit_label_map"] = label_map
            report["label_map_sha256"] = sha256_file(label_map_path)
        sources = []
        if manifest:
            if train_csv or validation_csv:
                raise ValueError("Use either --manifest or paired --train-csv/--validation-csv.")
            rows, source = read_annotations(manifest, root, options)
            sources.append(source)
        elif train_csv or validation_csv:
            for path, split in ((train_csv, "train"), (validation_csv, "validation")):
                if path:
                    part, source = read_annotations(path, root, options, split)
                    rows.extend(part)
                    sources.append(source)
        else:
            rows, source = read_folders(root)
            sources.append(source)
        report["annotation_sources"] = sources
        selected = []
        observed_ids = defaultdict(set)
        for row in rows:
            canonical = normalize_label(row["raw_label"], label_map)
            if canonical is None:
                exclusions.append({**row, "reason": "explicitly excluded by label map"})
                continue
            row["true_label"] = canonical
            row["class_id"] = CLASSES.index(canonical)
            selected.append(row)
            observed_ids[canonical].add(row["raw_label"])
        report["n_selected"] = len(selected)
        report["n_excluded"] = len(exclusions)
        raw_counts = Counter((r["split"], r["true_label"]) for r in selected)
        valid, seen_paths, content_split = [], set(), {}
        from PIL import Image
        for row in selected:
            path = Path(row["filepath"])
            if row["filepath"] in seen_paths:
                raise ValueError(f"Duplicate path in annotations: {path}")
            seen_paths.add(row["filepath"])
            try:
                with Image.open(path) as image:
                    image.verify()
                with Image.open(path) as image:
                    image.convert("RGB").load()
                digest = sha256_file(path)
            except (OSError, ValueError, Image.DecompressionBombError) as exc:
                failures.append({**row, "reason": f"unreadable image: {exc}"})
                continue
            if digest in content_split and content_split[digest] != row["split"]:
                raise ValueError(f"Train/validation content leakage detected by SHA256: {path}")
            content_split[digest] = row["split"]
            valid.append({**row, "sha256": digest})
        valid_counts = Counter((r["split"], r["true_label"]) for r in valid)
        report["counts"] = [
            {"class_name": c, "class_id": sorted(observed_ids[c]), "canonical_class_id": i,
             "split": split, "n_images": raw_counts[split, c], "n_valid": valid_counts[split, c],
             "paper_n_images": 500 if split == "validation" else PAPER_TRAIN_COUNTS[c],
             "delta_n_images": raw_counts[split, c] - (500 if split == "validation" else PAPER_TRAIN_COUNTS[c])}
            for split in ("train", "validation") for i, c in enumerate(CLASSES)]
        report["n_valid"], report["n_failed"] = len(valid), len(failures)
        report["validation_equivalent_counts"] = all(valid_counts["validation", c] == PAPER_VALIDATION_PER_CLASS for c in CLASSES)
        report["validation_deltas"] = {c: valid_counts["validation", c] - 500 for c in CLASSES}
        report["train_deltas"] = {c: valid_counts["train", c] - PAPER_TRAIN_COUNTS[c] for c in CLASSES}
        report["manifest_sha256"] = fingerprint_rows(valid)
        if not report["validation_equivalent_counts"]:
            report["warnings"].append(f"Validation differs from 500/class: {report['validation_deltas']} (valid images minus paper).")
        if any(report["train_deltas"].values()):
            report["warnings"].append(f"Training differs from paper Table 3: {report['train_deltas']}.")
        if failures:
            report["warnings"].append(f"Excluded {len(failures)} unreadable images; see dataset_failures.csv.")
        if not all(valid_counts["validation", c] for c in CLASSES):
            report["warnings"].append("Some validation classes have no support; balanced accuracy averages observed classes; UAR always uses all seven.")
        report["warnings"].append("Matching counts cannot establish identity with the paper subset; image IDs, OpenFace version and preprocessing remain unverified.")
        if valid:
            report["status"] = "completed"
        else:
            report["errors"].append("No valid images remain after verification.")
        rows = valid
    except (ValueError, OSError, KeyError, TypeError) as exc:
        report["errors"].append(str(exc))
        rows = []
    _save_inspection(output, report, rows, failures, exclusions)
    return rows, report


def _save_inspection(output, report, rows, failures, exclusions=()):
    write_json(output / "dataset_report.json", report)
    write_csv(output / "class_distribution.csv", report["counts"],
              ["split", "class_name", "class_id", "canonical_class_id", "n_images", "n_valid", "paper_n_images", "delta_n_images"])
    write_csv(output / "verified_manifest.csv", rows,
              ["filepath", "true_label", "class_id", "raw_label", "split", "sha256"])
    write_csv(output / "dataset_failures.csv", failures, ["filepath", "raw_label", "split", "reason"])
    write_csv(output / "excluded_labels.csv", exclusions, ["filepath", "raw_label", "split", "reason"])
    table = ["split | class_name | class_id (source) | n_images | n_valid", "--- | --- | --- | ---: | ---:"]
    for r in report["counts"]:
        table.append(f"{r['split']} | {r['class_name']} | {r['class_id']} | {r['n_images']} | {r['n_valid']}")
    text = "\n".join(table) + "\n\n" + "\n".join(report["errors"] + report["warnings"]) + "\n"
    (output / "dataset_report.md").write_text(text, encoding="utf-8")
    print(text, flush=True)
