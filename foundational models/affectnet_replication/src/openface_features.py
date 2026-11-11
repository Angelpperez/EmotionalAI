"""Use actual OpenFace image output; never replace failed detections with zeros."""
from __future__ import annotations

import hashlib
import math
import os
import re
import shutil
import subprocess
from pathlib import Path

from src.utils import sha256_file, write_csv, write_json

AU_PATTERN = re.compile(r"^AU\d{2}_[rc]$")


def detect_openface(repository_root, explicit=None, inventory=None):
    candidates = []
    if explicit:
        candidate = Path(explicit).resolve()
        if not candidate.is_file() or candidate.name.lower() not in {"facelandmarkimg.exe", "facelandmarkimg"}:
            return {"available": False, "reason": f"Not a valid FaceLandmarkImg executable: {candidate}"}
        candidates = [candidate]
    else:
        if os.environ.get("OPENFACE_HOME"):
            candidates.append(Path(os.environ["OPENFACE_HOME"]) / "FaceLandmarkImg.exe")
        for name in ("FaceLandmarkImg.exe", "FaceLandmarkImg"):
            found = shutil.which(name)
            if found:
                candidates.append(Path(found))
        for path in (inventory or {}).get("openface_executables", []):
            if Path(path).name.lower() == "facelandmarkimg.exe":
                candidates.append(Path(path))
        for base in (Path(repository_root) / "external", Path("C:/OpenFace"), Path("C:/Program Files/OpenFace")):
            candidates.append(base / "FaceLandmarkImg.exe")
    for candidate in candidates:
        if candidate.is_file():
            return {"available": True, "executable": str(candidate.resolve()),
                    "executable_sha256": sha256_file(candidate),
                    "version": "unverified; executable hash recorded",
                    "detection_scope": "repository, PATH, OPENFACE_HOME, C:/OpenFace, C:/Program Files/OpenFace"}
    return {"available": False, "reason": "FaceLandmarkImg not found in repository, PATH, OPENFACE_HOME or standard OpenFace locations.",
            "installation_instructions": "README.md (OpenFace on Windows)"}


def parse_openface_csv(path):
    import pandas as pd
    frame = pd.read_csv(path, skipinitialspace=True)
    frame.columns = [str(c).strip() for c in frame.columns]
    columns = sorted(c for c in frame.columns if AU_PATTERN.fullmatch(c))
    if not columns:
        raise ValueError("No AUxx_r/AUxx_c columns in real OpenFace output.")
    if "success" in frame:
        frame = frame[pd.to_numeric(frame["success"], errors="coerce") == 1].copy()
        validity_rule = "success=1"
    elif "face" in frame:
        # Official FaceLandmarkImg 2.2.0 serializes `face, confidence`, WITHOUT
        # success. A row proves a face-detector output, not landmark success.
        face_ids = pd.to_numeric(frame["face"], errors="coerce")
        frame = frame[(face_ids >= 0) & (face_ids % 1 == 0)].copy()
        validity_rule = "detected face row; landmark success unavailable in static-image CSV"
    else:
        raise ValueError("Neither success nor the static-image face column is present; detection is unverified.")
    if frame.empty:
        raise ValueError("No detected face remains under the output's validity rule.")
    n_faces = len(frame)
    if "confidence" not in frame:
        raise ValueError("OpenFace output lacks confidence; face selection cannot be verified.")
    frame["confidence"] = pd.to_numeric(frame["confidence"], errors="coerce")
    frame = frame[frame["confidence"].map(lambda v: math.isfinite(v))]
    if frame.empty:
        raise ValueError("No face has finite confidence.")
    selected = frame.sort_values("confidence", ascending=False, kind="stable").iloc[0]
    values = {}
    for c in columns:
        value = float(selected[c])
        if not math.isfinite(value):
            raise ValueError(f"Non-finite AU value in {c}.")
        if c.endswith("_c") and value not in {0.0, 1.0}:
            raise ValueError(f"Non-binary AU presence: {c}={value}")
        if c.endswith("_r") and not 0 <= value <= 5:
            raise ValueError(f"AU intensity outside OpenFace [0,5]: {c}={value}")
        values[c] = value
    return values, {"n_detected_faces": n_faces, "confidence": float(selected["confidence"]),
                    "selected_face_id": str(selected.get("face_id", selected.get("face", "unavailable"))),
                    "validity_rule": validity_rule}


def extract_openface(records, output, detection, timeout=120):
    if not detection["available"]:
        return {"status": "blocked", "reason": detection["reason"], "n_attempted": 0,
                "n_valid": 0, "n_failed": 0, "metrics": None, "detection": detection}
    if not records:
        return {"status": "blocked", "reason": "No verified dataset images.", "n_attempted": 0,
                "n_valid": 0, "n_failed": 0, "metrics": None, "detection": detection}
    exe = Path(detection["executable"])
    good, failures, details, schema = [], [], [], None
    for index, row in enumerate(records):
        # Separate outputs prevent collisions when stems repeat in train and validation.
        identifier = hashlib.sha256(row["filepath"].encode()).hexdigest()[:24]
        raw_dir = output / "openface_raw" / identifier
        raw_dir.mkdir(parents=True, exist_ok=False)
        command = [str(exe), "-f", row["filepath"], "-out_dir", str(raw_dir), "-aus"]
        try:
            result = subprocess.run(command, cwd=exe.parent, capture_output=True, text=True,
                                    encoding="utf-8", errors="replace", timeout=timeout, check=False)
            (raw_dir / "process.log").write_text(result.stdout + "\n" + result.stderr, encoding="utf-8")
            if result.returncode:
                raise ValueError(f"OpenFace exited {result.returncode}; see {raw_dir / 'process.log'}")
            csvs = list(raw_dir.glob("*.csv"))
            if len(csvs) != 1:
                raise ValueError(f"Expected one output CSV; found {len(csvs)}. No successful face output.")
            values, face_info = parse_openface_csv(csvs[0])
            if schema is None:
                schema = sorted(values)
            elif sorted(values) != schema:
                raise RuntimeError("OpenFace AU schema changed between images; refusing mixed feature vectors.")
            good.append({**row, **values})
            details.append({"filepath": row["filepath"], "split": row["split"], **face_info})
        except (ValueError, OSError, subprocess.TimeoutExpired) as exc:
            failures.append({"filepath": row["filepath"], "split": row["split"], "true_label": row["true_label"],
                             "reason": str(exc)})
        if (index + 1) % 100 == 0 or index == len(records) - 1:
            print(f"OpenFace: {index+1}/{len(records)}, valid={len(good)}, failed={len(failures)}", flush=True)
    columns = schema or []
    write_csv(output / "openface_features.csv", good,
              ["filepath", "true_label", "class_id", "split", "sha256"] + columns)
    write_csv(output / "predictions" / "openface_failures.csv", failures,
              ["filepath", "split", "true_label", "reason"])
    write_csv(output / "predictions" / "openface_face_selection.csv", details,
              ["filepath", "split", "n_detected_faces", "confidence", "selected_face_id", "validity_rule"])
    summary = {"status": "completed" if good else "failed", "detection": detection,
               "n_attempted": len(records), "n_valid": len(good), "n_failed": len(failures),
               "n_features": len(columns), "feature_columns": columns,
               "matches_paper_35_features": len(columns) == 35,
               "face_policy": "success=1 when serialized, otherwise static-image detected face row; highest confidence, ties use first output row",
               "command_template": [str(exe), "-f", "<filepath>", "-out_dir", "<unique_dir>", "-aus"],
               "observed_validity_rules": sorted({d["validity_rule"] for d in details}),
               "warnings": ["OpenFace version and multiple-face policy of the paper are not specified; exact replication unverified."]}
    if any("unavailable" in d["validity_rule"] for d in details):
        summary["warnings"].append("FaceLandmarkImg static-image CSV does not serialize landmark success; face rows verify detector output only. No additional confidence threshold is assumed.")
    if len(columns) != 35:
        summary["warnings"].append(f"Observed {len(columns)} AU columns, versus the reference's 35; all observed AU columns are used.")
    if not good:
        summary["reason"] = "No successful OpenFace detections; no SVC will be trained. See openface_failures.csv."
    summary["per_split"] = {split: {"n_valid": sum(r["split"] == split for r in good),
                                    "n_failed": sum(r["split"] == split for r in failures)}
                            for split in ("train", "validation")}
    write_json(output / "metrics" / "openface_summary.json", summary)
    return summary
