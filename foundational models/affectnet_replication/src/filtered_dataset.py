"""Read-only audit and pre-inference selection for the explicitly configured dataset."""
from __future__ import annotations

import hashlib
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from config import CLASSES, IMAGE_EXTENSIONS
from src.utils import fingerprint_rows, sha256_file, utc_now, write_csv, write_json

MANIFEST_FIELDS = ["sample_id", "filepath", "relative_path", "split", "source_partition", "source_label",
                   "canonical_true_label", "true_label", "file_sha256", "rgb_sha256", "n_bytes",
                   "mtime_ns", "readable", "width", "height", "mode", "format", "error", "duplicate_group_id"]


def audit_image(item):
    from PIL import Image
    path, root, partition, split, label, canonical = item
    relative = path.relative_to(root).as_posix()
    row = {"sample_id": hashlib.sha256(relative.encode("utf-8")).hexdigest()[:24],
           "filepath": str(path.resolve()), "relative_path": relative, "split": split,
           "source_partition": partition, "source_label": label,
           "canonical_true_label": canonical, "true_label": canonical,
           "readable": False, "file_sha256": "", "rgb_sha256": "", "error": ""}
    try:
        stat = path.stat()
        row.update(n_bytes=stat.st_size, mtime_ns=stat.st_mtime_ns, file_sha256=sha256_file(path))
        with Image.open(path) as im:
            im.verify()
        with Image.open(path) as im:
            im.load()
            row.update(width=im.width, height=im.height, mode=im.mode, format=im.format)
            h = hashlib.sha256(f"RGB:{im.width}x{im.height}:".encode())
            h.update(im.convert("RGB").tobytes())
            row.update(readable=True, rgb_sha256=h.hexdigest())
    except (OSError, ValueError, Image.DecompressionBombError) as exc:
        row["error"] = f"{type(exc).__name__}: {exc}"
    return row


def select_evaluation(rows):
    """Group by the union of byte equality and decoded-pixel equality across ALL labels."""
    parents = list(range(len(rows)))

    def find(i):
        while parents[i] != i:
            parents[i] = parents[parents[i]]
            i = parents[i]
        return i

    seen_hashes = {}
    for i, row in enumerate(rows):
        for kind in ("file_sha256", "rgb_sha256"):
            digest = row.get(kind)
            if digest:
                key = (kind, digest)
                if key in seen_hashes:
                    parents[find(i)] = find(seen_hashes[key])
                else:
                    seen_hashes[key] = i
    groups = defaultdict(list)
    for i, row in enumerate(rows):
        groups[find(i)].append(row)
    group_info = {}
    for members in groups.values():
        identifier = min(r["sample_id"] for r in members)
        info = {"group_id": identifier, "members": [r["relative_path"] for r in members],
                "labels": sorted({r["source_label"] for r in members}),
                "splits": sorted({r["split"] for r in members})}
        group_info[identifier] = info
        for row in members:
            row["duplicate_group_id"] = identifier
    selected, exclusions, seen_validation = [], [], set()
    counts = defaultdict(Counter)
    for row in sorted(rows, key=lambda r: r["relative_path"]):
        reasons = []
        info = group_info[row["duplicate_group_id"]]
        if row["canonical_true_label"] is None:
            reasons.append("excluded_class_contempt")
        if not row["readable"]:
            reasons.append("unreadable")
        if len(info["labels"]) > 1:
            reasons.append("conflicting_source_labels_all_eight_classes")
        if row["split"] == "validation" and "train" in info["splits"]:
            reasons.append("identical_content_in_train")
        if row["split"] == "validation" and row["canonical_true_label"] is not None:
            counts[row["canonical_true_label"]]["candidates"] += 1
            if row["duplicate_group_id"] in seen_validation:
                reasons.append("repeated_content_within_validation")
            if not reasons:
                selected.append(row)
                seen_validation.add(row["duplicate_group_id"])
            counts[row["canonical_true_label"]][reasons[0] if reasons else "included"] += 1
        if reasons:
            exclusions.append({**row, "primary_reason": reasons[0], "all_reasons": ";".join(reasons)})
    duplicates = [info for info in group_info.values() if len(info["members"]) > 1]
    summary = {"n_groups": len(duplicates), "n_images_in_groups": sum(len(g["members"]) for g in duplicates),
               "n_cross_partition_groups": sum(len(g["splits"]) > 1 for g in duplicates),
               "n_conflicting_label_groups": sum(len(g["labels"]) > 1 for g in duplicates),
               "groups": duplicates}
    return selected, exclusions, {c: dict(counts[c]) for c in CLASSES}, summary


def inspect_filtered_dataset(root, output, profile, workers=6):
    root, output = Path(root).resolve(), Path(output)
    if not root.is_dir():
        raise FileNotFoundError(f"Dataset root not found: {root}")
    expected_partitions = profile["partitions"]
    actual_partitions = {p.name for p in root.iterdir() if p.is_dir()}
    if actual_partitions != set(expected_partitions):
        raise ValueError(f"Dataset partitions changed: actual={sorted(actual_partitions)}, configured={sorted(expected_partitions)}")
    mapping = profile["source_labels"]
    items, ignored = [], []
    for parent in [root, *(root / name for name in expected_partitions)]:
        for path in parent.iterdir():
            if path.is_file():
                if path.suffix.lower() in IMAGE_EXTENSIONS:
                    raise ValueError(f"Unlabelled image outside a configured class directory: {path}")
                ignored.append(str(path.relative_to(root)))
    for partition, split in expected_partitions.items():
        folder = root / partition
        actual_classes = {p.name for p in folder.iterdir() if p.is_dir()}
        if actual_classes != set(mapping):
            raise ValueError(f"Class directories changed in {folder}: {sorted(actual_classes)}")
        for label, canonical in mapping.items():
            if canonical is not None and canonical not in CLASSES:
                raise ValueError(f"Invalid canonical mapping: {label} -> {canonical}")
            for path in sorted((folder / label).rglob("*")):
                if path.is_file():
                    if path.suffix.lower() in IMAGE_EXTENSIONS:
                        items.append((path, root, partition, split, label, canonical))
                    else:
                        ignored.append(str(path.relative_to(root)))
    rows = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for row in pool.map(audit_image, items):
            rows.append(row)
            if len(rows) % 2000 == 0 or len(rows) == len(items):
                print(f"Audit: {len(rows)}/{len(items)} images; unreadable={sum(not r['readable'] for r in rows)}", flush=True)
    rows.sort(key=lambda r: r["relative_path"])
    if len({r["sample_id"] for r in rows}) != len(rows):
        raise ValueError("Duplicate sample IDs.")
    selected, exclusions, attrition, duplicates = select_evaluation(rows)
    manifest_dir = output / "manifests"
    write_csv(manifest_dir / "all_images.csv", rows, MANIFEST_FIELDS)
    write_csv(manifest_dir / "evaluation_manifest.csv", selected, MANIFEST_FIELDS)
    write_csv(manifest_dir / "exclusions.csv", exclusions, MANIFEST_FIELDS + ["primary_reason", "all_reasons"])
    write_json(manifest_dir / "duplicate_groups.json", duplicates)
    counts = Counter((r["source_partition"], r["source_label"]) for r in rows)
    readable = Counter((r["source_partition"], r["source_label"]) for r in rows if r["readable"])
    report = {"status": "completed", "timestamp_utc": utc_now(), "dataset_root": str(root),
              "dataset_name": profile["dataset_name"], "partitions": expected_partitions,
              "source_label_mapping": mapping, "canonical_class_order": list(CLASSES),
              "counts": [{"partition": p, "split": expected_partitions[p], "source_label": label,
                          "canonical_label": mapping[label], "n_images": n, "n_readable": readable[p, label]}
                         for (p, label), n in sorted(counts.items())],
              "n_images": len(rows), "n_readable": sum(r["readable"] for r in rows),
              "n_unreadable": sum(not r["readable"] for r in rows),
              "extensions": dict(Counter(Path(r["filepath"]).suffix.lower() for r in rows)),
              "modes": dict(Counter(r.get("mode", "unknown") for r in rows)),
              "dimensions": dict(Counter(f"{r.get('width')}x{r.get('height')}" for r in rows)),
              "evaluation_n": len(selected), "evaluation_counts": dict(Counter(r["true_label"] for r in selected)),
              "attrition_by_class": attrition,
              "duplicate_summary": {k: v for k, v in duplicates.items() if k != "groups"},
              "ignored_files": ignored, "selection_policy": "protocol 1.0; conflicts across all source classes; no train overlap; unique validation content",
              "validation_identity": "provided validation split; original AffectNet split membership unverified",
              "all_manifest_sha256": sha256_file(manifest_dir / "all_images.csv"),
              "evaluation_manifest_sha256": sha256_file(manifest_dir / "evaluation_manifest.csv"),
              "evaluation_content_fingerprint": fingerprint_rows([{k: r[k] for k in ("relative_path", "file_sha256", "rgb_sha256", "true_label")} for r in selected]),
              "failures": [{"filepath": r["filepath"], "error": r["error"]} for r in rows if not r["readable"]]}
    archive = profile.get("archive_path")
    if archive and Path(archive).is_file():
        report["archive"] = {"path": str(Path(archive).resolve()), "sha256": sha256_file(archive), "n_bytes": Path(archive).stat().st_size}
    write_json(output / "dataset_inspection.json", report)
    print(f"Dataset: {root}", flush=True)
    print(f"Partitions: {expected_partitions}", flush=True)
    print("source_partition | source_label | canonical_label | n_images | n_readable", flush=True)
    for count in report["counts"]:
        print(f"{count['partition']} | {count['source_label']} | {count['canonical_label']} | {count['n_images']} | {count['n_readable']}", flush=True)
    print(f"Extensions: {report['extensions']}; total={report['n_images']}; unreadable={report['n_unreadable']}", flush=True)
    print("canonical_label | source_folder | n_images (final evaluation)", flush=True)
    for canonical in CLASSES:
        source = next(k for k, v in mapping.items() if v == canonical)
        print(f"{canonical} | {source} | {report['evaluation_counts'].get(canonical, 0)}", flush=True)
    if set(report["evaluation_counts"]) != set(CLASSES):
        raise ValueError("The cleaned evaluation does not contain all seven classes.")
    if profile.get("expected_evaluation_n") is not None and len(selected) != profile["expected_evaluation_n"]:
        raise ValueError("Dataset drift: final N differs from the audited protocol. Review inspection before creating a new profile.")
    if profile.get("expected_evaluation_counts") is not None and report["evaluation_counts"] != profile["expected_evaluation_counts"]:
        raise ValueError("Dataset drift: final class distribution differs from the audited protocol.")
    return selected, report
