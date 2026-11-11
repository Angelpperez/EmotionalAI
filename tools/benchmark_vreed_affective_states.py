"""Benchmark VREED affective-state classifiers without modifying legacy files.

Outputs are written under ``outputs/emotion_benchmark/run_*`` by default.

Two targets are supported:

``quad4``
    Original VREED ``Quad_Cat`` labels.

``stress3``
    Operational closed-loop states derived from the VREED circumplex:
    Quad 1 (low arousal, positive valence) -> low
    Quad 3 (high arousal, negative valence) -> high
    Quad 0 and 2 -> medium

The 3-state mapping is a control-system label for the Blender demo, not a
clinical diagnosis.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.svm import SVC


LEGACY_EYE_FEATURES = [
    "Num_of_Fixations",
    "Mean_Fixation_Duration",
    "SD_Fixation_Duration",
    "Num_of_Saccade",
    "Mean_Saccade_Amplitude",
    "Mean_Saccade_Direction",
]

STRESS3_MAP = {
    0: "medium",  # high arousal, positive valence: engagement/excitement
    1: "low",     # low arousal, positive valence: calm/relaxed
    2: "medium",  # low arousal, negative valence: non-optimal, not acute overload
    3: "high",    # high arousal, negative valence: stress/overload
}

TARGET_LABELS = {
    "quad4": ["0", "1", "2", "3"],
    "stress3": ["low", "medium", "high"],
}


@dataclass(frozen=True)
class DatasetBundle:
    x: np.ndarray
    y: np.ndarray
    feature_names: list[str]
    modality: str


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def load_vreed_tables(data_root: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    eye = read_csv(
        data_root
        / "04 Eye Tracking Data"
        / "02 Eye Tracking Data (Features Extracted)"
        / "EyeTracking_FeaturesExtracted.csv"
    )
    ecg = read_csv(
        data_root
        / "05 ECG-GSR Data"
        / "02 ECG-GSR (Features Extracted)"
        / "ECG_FeaturesExtracted.csv"
    )
    gsr = read_csv(
        data_root
        / "05 ECG-GSR Data"
        / "02 ECG-GSR (Features Extracted)"
        / "GSR_FeaturesExtracted.csv"
    )

    if not (len(eye) == len(ecg) == len(gsr)):
        raise ValueError(f"Row-count mismatch: eye={len(eye)} ecg={len(ecg)} gsr={len(gsr)}")
    if not (eye["Quad_Cat"].equals(ecg["Quad_Cat"]) and eye["Quad_Cat"].equals(gsr["Quad_Cat"])):
        raise ValueError("Quad_Cat labels are not aligned row-wise across modalities")
    return eye, ecg, gsr


def make_target(quad: pd.Series, target: str) -> np.ndarray:
    quad_int = quad.astype(int)
    if target == "quad4":
        return quad_int.astype(str).to_numpy()
    if target == "stress3":
        return quad_int.map(STRESS3_MAP).to_numpy()
    raise ValueError(f"Unsupported target: {target}")


def prefixed_features(df: pd.DataFrame, prefix: str) -> tuple[pd.DataFrame, list[str]]:
    feature_df = df.drop(columns=["Quad_Cat"]).copy()
    feature_df = feature_df.add_prefix(prefix)
    return feature_df, feature_df.columns.tolist()


def build_datasets(eye: pd.DataFrame, ecg: pd.DataFrame, gsr: pd.DataFrame, target: str) -> list[DatasetBundle]:
    y = make_target(eye["Quad_Cat"], target)

    missing = [col for col in LEGACY_EYE_FEATURES if col not in eye.columns]
    if missing:
        raise ValueError(f"Missing legacy eye features: {missing}")

    eye_legacy = eye[LEGACY_EYE_FEATURES].copy()
    eye_all, eye_cols = prefixed_features(eye, "eye__")
    ecg_all, ecg_cols = prefixed_features(ecg, "ecg__")
    gsr_all, gsr_cols = prefixed_features(gsr, "gsr__")
    physio = pd.concat([ecg_all, gsr_all], axis=1)
    multimodal = pd.concat([eye_all, ecg_all, gsr_all], axis=1)

    return [
        DatasetBundle(eye_legacy.to_numpy(dtype=float), y, LEGACY_EYE_FEATURES, "eye_legacy6_live_compatible"),
        DatasetBundle(eye_all.to_numpy(dtype=float), y, eye_cols, "eye_all49"),
        DatasetBundle(ecg_all.to_numpy(dtype=float), y, ecg_cols, "ecg_all18"),
        DatasetBundle(gsr_all.to_numpy(dtype=float), y, gsr_cols, "gsr_all8"),
        DatasetBundle(physio.to_numpy(dtype=float), y, physio.columns.tolist(), "physio_ecg_gsr26"),
        DatasetBundle(multimodal.to_numpy(dtype=float), y, multimodal.columns.tolist(), "multimodal_eye_ecg_gsr75"),
    ]


def scaled(estimator) -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", RobustScaler()),
            ("model", estimator),
        ]
    )


def tree(estimator) -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("model", estimator),
        ]
    )


def model_registry(random_state: int) -> dict[str, Pipeline]:
    return {
        "LogReg_balanced": scaled(
            LogisticRegression(max_iter=3000, class_weight="balanced", random_state=random_state)
        ),
        "SVM_RBF_balanced": scaled(
            SVC(kernel="rbf", class_weight="balanced", probability=True, random_state=random_state)
        ),
        "RandomForest": tree(
            RandomForestClassifier(
                n_estimators=600,
                n_jobs=-1,
                random_state=random_state,
            )
        ),
        "ExtraTrees": tree(
            ExtraTreesClassifier(
                n_estimators=800,
                n_jobs=-1,
                random_state=random_state,
            )
        ),
        "HistGradientBoosting": tree(
            HistGradientBoostingClassifier(
                learning_rate=0.05,
                max_iter=250,
                l2_regularization=0.05,
                random_state=random_state,
            )
        ),
        "MLP_small": scaled(
            MLPClassifier(
                hidden_layer_sizes=(96, 48),
                activation="relu",
                alpha=1e-3,
                early_stopping=True,
                max_iter=1200,
                random_state=random_state,
            )
        ),
    }


def evaluate_dataset(
    bundle: DatasetBundle,
    models: dict[str, Pipeline],
    target: str,
    n_splits: int,
    n_repeats: int,
    random_state: int,
) -> tuple[list[dict[str, object]], dict[str, np.ndarray], dict[str, np.ndarray]]:
    splitter = RepeatedStratifiedKFold(
        n_splits=n_splits,
        n_repeats=n_repeats,
        random_state=random_state,
    )
    labels = TARGET_LABELS[target]
    rows: list[dict[str, object]] = []
    y_true_by_model: dict[str, list[str]] = {name: [] for name in models}
    y_pred_by_model: dict[str, list[str]] = {name: [] for name in models}

    for fold_idx, (train_idx, test_idx) in enumerate(splitter.split(bundle.x, bundle.y), start=1):
        x_train, x_test = bundle.x[train_idx], bundle.x[test_idx]
        y_train, y_test = bundle.y[train_idx], bundle.y[test_idx]

        for model_name, estimator in models.items():
            model = clone(estimator)
            model.fit(x_train, y_train)
            y_pred = model.predict(x_test)
            rows.append(
                {
                    "target": target,
                    "modality": bundle.modality,
                    "model": model_name,
                    "fold": fold_idx,
                    "accuracy": accuracy_score(y_test, y_pred),
                    "balanced_accuracy": balanced_accuracy_score(y_test, y_pred),
                    "f1_macro": f1_score(y_test, y_pred, average="macro"),
                    "n_train": len(train_idx),
                    "n_test": len(test_idx),
                    "n_features": len(bundle.feature_names),
                }
            )
            y_true_by_model[model_name].extend(y_test.tolist())
            y_pred_by_model[model_name].extend(y_pred.tolist())

    y_true_np = {name: np.asarray(values) for name, values in y_true_by_model.items()}
    y_pred_np = {name: np.asarray(values) for name, values in y_pred_by_model.items()}
    return rows, y_true_np, y_pred_np


def summarise(rows: list[dict[str, object]]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    grouped = (
        df.groupby(["target", "modality", "model"], as_index=False)
        .agg(
            accuracy_mean=("accuracy", "mean"),
            accuracy_std=("accuracy", "std"),
            balanced_accuracy_mean=("balanced_accuracy", "mean"),
            balanced_accuracy_std=("balanced_accuracy", "std"),
            f1_macro_mean=("f1_macro", "mean"),
            f1_macro_std=("f1_macro", "std"),
            n_features=("n_features", "first"),
        )
        .sort_values(["target", "f1_macro_mean"], ascending=[True, False])
    )
    return grouped


def save_confusion_plot(cm: np.ndarray, labels: list[str], title: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(5.5, 4.8))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_title(title)
    ax.set_xlabel("Predicho")
    ax.set_ylabel("Real")
    ax.set_xticks(range(len(labels)), labels)
    ax.set_yticks(range(len(labels)), labels)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center", color="black")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220)
    plt.close(fig)


def fit_and_save_best(
    out_dir: Path,
    datasets: list[DatasetBundle],
    models: dict[str, Pipeline],
    summary: pd.DataFrame,
    target: str,
    best_selector: Callable[[pd.DataFrame], pd.DataFrame],
) -> list[dict[str, object]]:
    artifacts: list[dict[str, object]] = []
    selected = best_selector(summary[summary["target"] == target])
    for _, row in selected.iterrows():
        modality = row["modality"]
        model_name = row["model"]
        bundle = next(item for item in datasets if item.modality == modality)
        model = clone(models[model_name])
        model.fit(bundle.x, bundle.y)
        artifact = {
            "pipeline": model,
            "target": target,
            "modality": modality,
            "model_name": model_name,
            "feature_names": bundle.feature_names,
            "label_order": TARGET_LABELS[target],
            "stress3_map": STRESS3_MAP,
            "cv_metrics": row.to_dict(),
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "note": "Generated by tools/benchmark_vreed_affective_states.py; legacy files were not modified.",
        }
        path = out_dir / "models" / target / f"{modality}__{model_name}.joblib"
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(artifact, path)
        meta = dict(artifact)
        meta.pop("pipeline")
        meta["artifact_path"] = str(path)
        artifacts.append(meta)
    return artifacts


def write_markdown_report(
    path: Path,
    summary: pd.DataFrame,
    artifacts: list[dict[str, object]],
    args: argparse.Namespace,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    top = summary.sort_values(["target", "f1_macro_mean"], ascending=[True, False])
    lines = [
        "# VREED Affective-State Benchmark",
        "",
        f"Generated: {datetime.now().isoformat(timespec='seconds')}",
        "",
        "## Protocol",
        "",
        f"- Data root: `{args.data_root}`",
        f"- Cross-validation: RepeatedStratifiedKFold, splits={args.n_splits}, repeats={args.n_repeats}",
        "- Metrics: accuracy, balanced accuracy, F1 macro.",
        "- Legacy scripts and legacy model artifacts were not modified.",
        "",
        "## Stress3 Mapping",
        "",
        "| Quad_Cat | Circumplex interpretation | Closed-loop state |",
        "|---:|---|---|",
        "| 0 | high arousal, positive valence | medium |",
        "| 1 | low arousal, positive valence | low |",
        "| 2 | low arousal, negative valence | medium |",
        "| 3 | high arousal, negative valence | high |",
        "",
        "This is an operational control-state mapping, not a clinical diagnosis.",
        "",
        "## Top Results",
        "",
        top.to_markdown(index=False),
        "",
        "## Saved Model Artifacts",
        "",
    ]
    for artifact in artifacts:
        lines.append(f"- `{artifact['artifact_path']}`")
        lines.append(
            f"  target={artifact['target']}, modality={artifact['modality']}, "
            f"model={artifact['model_name']}, F1={artifact['cv_metrics']['f1_macro_mean']:.3f}"
        )
    lines.extend(
        [
            "",
            "## Methodological Limitations",
            "",
            "- The extracted-feature CSVs do not include subject/trial identifiers, so this benchmark uses stratified repeated CV rather than leave-one-subject-out.",
            "- The live-compatible model can only use features computable from the current webcam prototype (`eye_legacy6_live_compatible`).",
            "- Multimodal ECG/GSR results are useful as an upper-bound reference unless equivalent real-time sensors are connected.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("EmotionRecognition/archive"))
    parser.add_argument("--output-root", type=Path, default=Path("outputs/emotion_benchmark"))
    parser.add_argument("--targets", nargs="+", choices=["quad4", "stress3"], default=["quad4", "stress3"])
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--n-repeats", type=int, default=5)
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_id = datetime.now().strftime("run_%Y%m%d_%H%M%S")
    out_dir = args.output_root / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    eye, ecg, gsr = load_vreed_tables(args.data_root)
    models = model_registry(args.random_state)
    all_rows: list[dict[str, object]] = []
    all_artifacts: list[dict[str, object]] = []

    for target in args.targets:
        datasets = build_datasets(eye, ecg, gsr, target)
        target_y_true: dict[tuple[str, str], np.ndarray] = {}
        target_y_pred: dict[tuple[str, str], np.ndarray] = {}

        for bundle in datasets:
            rows, y_true_by_model, y_pred_by_model = evaluate_dataset(
                bundle=bundle,
                models=models,
                target=target,
                n_splits=args.n_splits,
                n_repeats=args.n_repeats,
                random_state=args.random_state,
            )
            all_rows.extend(rows)
            for model_name in models:
                key = (bundle.modality, model_name)
                target_y_true[key] = y_true_by_model[model_name]
                target_y_pred[key] = y_pred_by_model[model_name]

        fold_df = pd.DataFrame(all_rows)
        fold_df.to_csv(out_dir / "fold_metrics.csv", index=False)
        summary = summarise(all_rows)
        summary.to_csv(out_dir / "summary_metrics.csv", index=False)

        best_row = summary[summary["target"] == target].sort_values("f1_macro_mean", ascending=False).iloc[0]
        best_key = (best_row["modality"], best_row["model"])
        labels = TARGET_LABELS[target]
        cm = confusion_matrix(target_y_true[best_key], target_y_pred[best_key], labels=labels)
        pd.DataFrame(cm, index=labels, columns=labels).to_csv(out_dir / f"confusion_{target}_best.csv")
        save_confusion_plot(
            cm,
            labels,
            f"{target}: {best_row['modality']} / {best_row['model']}",
            out_dir / f"confusion_{target}_best.png",
        )
        report = classification_report(
            target_y_true[best_key],
            target_y_pred[best_key],
            labels=labels,
            zero_division=0,
        )
        (out_dir / f"classification_report_{target}_best.txt").write_text(report, encoding="utf-8")

        def select_best_overall(df: pd.DataFrame) -> pd.DataFrame:
            return df.sort_values("f1_macro_mean", ascending=False).head(1)

        def select_live_compatible(df: pd.DataFrame) -> pd.DataFrame:
            return (
                df[df["modality"] == "eye_legacy6_live_compatible"]
                .sort_values("f1_macro_mean", ascending=False)
                .head(1)
            )

        all_artifacts.extend(fit_and_save_best(out_dir, datasets, models, summary, target, select_best_overall))
        if target == "stress3":
            all_artifacts.extend(fit_and_save_best(out_dir, datasets, models, summary, target, select_live_compatible))

    summary = summarise(all_rows)
    summary.to_csv(out_dir / "summary_metrics.csv", index=False)
    with (out_dir / "artifacts.json").open("w", encoding="utf-8") as f:
        json.dump(all_artifacts, f, indent=2, ensure_ascii=False)
    write_markdown_report(out_dir / "README.md", summary, all_artifacts, args)

    latest = args.output_root / "latest_run.txt"
    latest.parent.mkdir(parents=True, exist_ok=True)
    latest.write_text(str(out_dir.resolve()), encoding="utf-8")
    print(f"output_dir: {out_dir}")
    print(summary.sort_values(['target', 'f1_macro_mean'], ascending=[True, False]).head(12).to_string(index=False))


if __name__ == "__main__":
    main()
