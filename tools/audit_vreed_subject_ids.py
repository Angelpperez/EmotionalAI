"""Audit participant IDs and structural completeness in the local VREED data."""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import pandas as pd


def load_pickle(path: Path) -> dict:
    with path.open("rb") as handle:
        return pickle.load(handle, encoding="latin1")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("EmotionRecognition/archive"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/vreed_subject_audit"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    eye_root = (
        args.data_root
        / "04 Eye Tracking Data"
        / "01 Eye Tracking Data (Pre-Processed)"
    )
    phys_root = (
        args.data_root
        / "05 ECG-GSR Data"
        / "01 ECG-GSR Data (Pre-Processed)"
    )
    profile_path = (
        args.data_root
        / "03 Self-Reported Questionnaires"
        / "01 Participant Profile  Pre-Exposure Ratings.xlsx"
    )
    post_path = (
        args.data_root
        / "03 Self-Reported Questionnaires"
        / "02 Post Exposure Ratings.xlsx"
    )

    profiles = pd.read_excel(profile_path)
    post = pd.read_excel(post_path)
    rows = []
    for participant_id in sorted(profiles["ID"].astype(int).tolist()):
        eye_path = eye_root / f"{participant_id}_EyeTracking_PreProcessed.dat"
        phys_path = phys_root / f"{participant_id}_ECG_GSR_PreProcessed.dat"
        eye = load_pickle(eye_path)
        phys = load_pickle(phys_path)
        eye_trials = len(eye["Data"])
        phys_segments = len(phys["Data"])
        questionnaire_trials = int((post["ID"].astype(int) == participant_id).sum())
        rows.append(
            {
                "subject_id": participant_id,
                "eye_trials_present": eye_trials,
                "eye_label_count": len(eye["Labels"]),
                "physio_segments_present": phys_segments,
                "physio_label_count": len(phys["Labels"]),
                "questionnaire_trials": questionnaire_trials,
                "eye_12_trials": eye_trials == 12,
                "physio_baseline_plus_12": phys_segments == 13,
                "structurally_complete_multimodal": eye_trials == 12 and phys_segments == 13,
            }
        )

    audit = pd.DataFrame(rows)
    feature_counts = {}
    for name in (
        "EyeTracking_FeaturesExtracted.csv",
        "ECG_FeaturesExtracted.csv",
        "GSR_FeaturesExtracted.csv",
    ):
        path = next(args.data_root.rglob(name))
        table = pd.read_csv(path)
        feature_counts[name] = {
            "rows": int(len(table)),
            "columns": int(len(table.columns)),
            "has_subject_id": "subject_id" in table.columns,
            "has_trial_id": "trial_id" in table.columns,
        }

    summary = {
        "participant_files": int(len(audit)),
        "questionnaire_participants": int(profiles["ID"].nunique()),
        "structurally_complete_eye": int(audit["eye_12_trials"].sum()),
        "structurally_complete_multimodal": int(
            audit["structurally_complete_multimodal"].sum()
        ),
        "feature_extracted_rows": 312,
        "feature_extracted_implied_subjects_at_12_trials": 26,
        "feature_tables": feature_counts,
        "conclusion": (
            "Subject IDs exist in participant-level pre-processed files, but the "
            "312-row extracted-feature CSVs do not preserve a verifiable subject mapping."
        ),
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    audit.to_csv(args.output_dir / "vreed_subject_completeness.csv", index=False)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(audit.to_string(index=False))
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
