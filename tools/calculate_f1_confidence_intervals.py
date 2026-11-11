"""Calculate descriptive confidence intervals from benchmark fold metrics.

The benchmark uses repeated stratified cross-validation. Individual folds
overlap in their training data and must not be treated as independent
replicates. The primary interval reported here is therefore a Student-t
interval over the mean F1 of each complete CV repetition.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


def t_interval(values: np.ndarray, confidence: float) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    if values.size < 2:
        return float("nan"), float("nan")
    mean = float(np.mean(values))
    sem = float(stats.sem(values))
    low, high = stats.t.interval(confidence, df=values.size - 1, loc=mean, scale=sem)
    return float(low), float(high)


def calculate_intervals(
    fold_df: pd.DataFrame,
    n_splits: int,
    confidence: float,
) -> pd.DataFrame:
    required = {"target", "modality", "model", "fold", "f1_macro"}
    missing = required.difference(fold_df.columns)
    if missing:
        raise ValueError(f"Missing columns in fold metrics: {sorted(missing)}")

    work = fold_df.copy()
    work["fold"] = pd.to_numeric(work["fold"], errors="raise").astype(int)
    work["f1_macro"] = pd.to_numeric(work["f1_macro"], errors="raise")
    work["repeat"] = ((work["fold"] - 1) // n_splits) + 1

    rows: list[dict[str, object]] = []
    group_cols = ["target", "modality", "model"]
    for keys, group in work.groupby(group_cols, sort=True):
        repeat_means = (
            group.groupby("repeat", sort=True)["f1_macro"]
            .mean()
            .to_numpy(dtype=float)
        )
        fold_values = group["f1_macro"].to_numpy(dtype=float)
        repeat_low, repeat_high = t_interval(repeat_means, confidence)
        naive_low, naive_high = t_interval(fold_values, confidence)
        rows.append(
            {
                "target": keys[0],
                "modality": keys[1],
                "model": keys[2],
                "n_folds": int(fold_values.size),
                "n_repeats": int(repeat_means.size),
                "n_splits": int(n_splits),
                "f1_macro_mean": float(np.mean(fold_values)),
                "f1_macro_fold_std": float(np.std(fold_values, ddof=1)),
                "f1_macro_repeat_std": float(np.std(repeat_means, ddof=1)),
                "ci_level": float(confidence),
                "ci_repeat_t_low": repeat_low,
                "ci_repeat_t_high": repeat_high,
                "ci_repeat_t_half_width": float((repeat_high - repeat_low) / 2.0),
                "ci_naive_fold_t_low": naive_low,
                "ci_naive_fold_t_high": naive_high,
                "fold_percentile_low": float(np.quantile(fold_values, (1.0 - confidence) / 2.0)),
                "fold_percentile_high": float(np.quantile(fold_values, 1.0 - (1.0 - confidence) / 2.0)),
            }
        )

    return (
        pd.DataFrame(rows)
        .sort_values(["target", "f1_macro_mean"], ascending=[True, False])
        .reset_index(drop=True)
    )


def select_report_rows(intervals: pd.DataFrame) -> pd.DataFrame:
    selected = []
    for target in sorted(intervals["target"].unique()):
        target_df = intervals[intervals["target"] == target]
        selected.append(target_df.sort_values("f1_macro_mean", ascending=False).head(1))
    live = intervals[
        (intervals["target"] == "stress3")
        & (intervals["modality"] == "eye_legacy6_live_compatible")
    ].sort_values("f1_macro_mean", ascending=False).head(1)
    if not live.empty:
        selected.append(live)
    return pd.concat(selected, ignore_index=True).drop_duplicates(
        subset=["target", "modality", "model"]
    )


def save_plot(selected: pd.DataFrame, path: Path) -> None:
    labels = [
        f"{row.target}\n{row.modality}\n{row.model}"
        for row in selected.itertuples(index=False)
    ]
    means = selected["f1_macro_mean"].to_numpy(dtype=float)
    lower = means - selected["ci_repeat_t_low"].to_numpy(dtype=float)
    upper = selected["ci_repeat_t_high"].to_numpy(dtype=float) - means

    fig, ax = plt.subplots(figsize=(10, 5.5))
    x = np.arange(len(selected))
    ax.errorbar(
        x,
        means,
        yerr=np.vstack([lower, upper]),
        fmt="o",
        capsize=6,
        linewidth=1.8,
    )
    ax.set_xticks(x, labels)
    ax.set_ylabel("F1 macro")
    ax.set_title("F1 macro medio e IC 95% sobre medias por repeticion")
    ax.set_ylim(0.0, 1.0)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220)
    plt.close(fig)


def write_report(
    selected: pd.DataFrame,
    output_path: Path,
    input_path: Path,
    confidence: float,
) -> None:
    pct = confidence * 100.0
    lines = [
        "# Intervalos de confianza de F1 macro",
        "",
        f"Fuente: `{input_path}`.",
        "",
        "## Metodo principal",
        "",
        "El benchmark usa validacion cruzada estratificada repetida. Los folds",
        "individuales comparten muestras de entrenamiento y no son replicas",
        "independientes. Por ello, el intervalo principal se calcula sobre las",
        "cinco medias por repeticion completa, usando una distribucion t de",
        f"Student y un nivel de confianza de {pct:.0f}%.",
        "",
        "Este intervalo cuantifica variabilidad entre repeticiones del protocolo",
        "actual; no reemplaza un intervalo por participantes ni una validacion",
        "Leave-One-Subject-Out.",
        "",
        "## Resultados principales",
        "",
        "| Target | Modalidad | Modelo | F1 medio | IC bajo | IC alto | Repeticiones |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for row in selected.itertuples(index=False):
        lines.append(
            f"| {row.target} | {row.modality} | {row.model} | "
            f"{row.f1_macro_mean:.6f} | {row.ci_repeat_t_low:.6f} | "
            f"{row.ci_repeat_t_high:.6f} | {row.n_repeats} |"
        )
    lines.extend(
        [
            "",
            "## Advertencia",
            "",
            "El CSV de salida tambien incluye un intervalo t calculado directamente",
            "sobre los 25 folds (`ci_naive_fold_t_*`) solo como referencia. Ese",
            "intervalo es optimista porque los folds repetidos no son independientes",
            "y no debe usarse como resultado doctoral principal.",
        ]
    )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fold-metrics",
        type=Path,
        default=Path("outputs/emotion_benchmark/run_20260616_034800/fold_metrics.csv"),
    )
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/emotion_benchmark/run_20260616_034800"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    fold_df = pd.read_csv(args.fold_metrics)
    intervals = calculate_intervals(fold_df, args.n_splits, args.confidence)
    selected = select_report_rows(intervals)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    intervals.to_csv(args.output_dir / "f1_confidence_intervals.csv", index=False)
    selected.to_csv(args.output_dir / "f1_confidence_intervals_selected.csv", index=False)
    write_report(
        selected,
        args.output_dir / "f1_confidence_intervals.md",
        args.fold_metrics,
        args.confidence,
    )
    save_plot(selected, args.output_dir / "f1_confidence_intervals.png")
    print(selected.to_string(index=False))


if __name__ == "__main__":
    main()
