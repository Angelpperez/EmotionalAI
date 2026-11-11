"""Auditoria y benchmark reproducible del conjunto VREED."""

from __future__ import annotations

import argparse
from pathlib import Path

from _common import (
    REPO_ROOT,
    python_executable,
    require,
    resolve_path,
    run,
    timestamp,
    write_manifest,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="EmotionRecognition/archive")
    parser.add_argument("--output-root", default="outputs/experiments/exp03_vreed_benchmark")
    parser.add_argument("--python", default="D:/emotion3dgs_env/Scripts/python.exe")
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--n-repeats", type=int, default=5)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--skip-audit", action="store_true")
    parser.add_argument("--skip-confidence-intervals", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = resolve_path(args.output_root) / f"run_{timestamp()}"
    audit_dir = output_dir / "subject_audit"
    benchmark_dir = output_dir / "benchmark"
    data_root = require(resolve_path(args.data_root), "Dataset VREED")
    python = python_executable(args.python)
    commands: list[list[str]] = []

    if not args.skip_audit:
        audit = [
            python,
            REPO_ROOT / "tools/audit_vreed_subject_ids.py",
            "--data-root",
            data_root,
            "--output-dir",
            audit_dir,
        ]
        commands.append([str(item) for item in audit])
        run(audit, dry_run=args.dry_run)

    benchmark = [
        python,
        REPO_ROOT / "tools/benchmark_vreed_affective_states.py",
        "--data-root",
        data_root,
        "--output-root",
        benchmark_dir,
        "--n-splits",
        str(args.n_splits),
        "--n-repeats",
        str(args.n_repeats),
        "--random-state",
        str(args.random_state),
    ]
    commands.append([str(item) for item in benchmark])
    run(benchmark, dry_run=args.dry_run)

    if not args.skip_confidence_intervals:
        if args.dry_run:
            fold_metrics = benchmark_dir / "run_TIMESTAMP/fold_metrics.csv"
            interval_dir = benchmark_dir / "run_TIMESTAMP"
        else:
            latest = benchmark_dir / "latest_run.txt"
            require(latest, "Referencia a la ultima corrida")
            interval_dir = Path(latest.read_text(encoding="utf-8").strip())
            fold_metrics = interval_dir / "fold_metrics.csv"
        intervals = [
            python,
            REPO_ROOT / "tools/calculate_f1_confidence_intervals.py",
            "--fold-metrics",
            fold_metrics,
            "--n-splits",
            str(args.n_splits),
            "--output-dir",
            interval_dir,
        ]
        commands.append([str(item) for item in intervals])
        run(intervals, dry_run=args.dry_run)

    write_manifest(
        output_dir,
        experiment="exp03_vreed_benchmark",
        arguments=vars(args),
        commands=commands,
    )
    print(f"Salida: {output_dir}")


if __name__ == "__main__":
    main()
