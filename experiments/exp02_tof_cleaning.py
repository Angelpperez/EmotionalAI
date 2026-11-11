"""Limpieza, segmentacion y visualizacion de la secuencia ToF Basler."""

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
    parser.add_argument("--input-root", default="captures/ToF")
    parser.add_argument("--output-root", default="outputs/experiments/exp02_tof_cleaning")
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--python", default="D:/emotion3dgs_env/Scripts/python.exe")
    parser.add_argument("--blender", default="C:/Program Files/Blender Foundation/Blender 4.4/blender.exe")
    parser.add_argument("--skip-segmentation", action="store_true")
    parser.add_argument("--skip-blender", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = resolve_path(args.output_root) / f"run_{timestamp()}"
    clean_dir = output_dir / "canonical_clean"
    segment_dir = output_dir / "rocks_segmented"
    input_root = require(resolve_path(args.input_root), "Secuencia ToF")
    python = python_executable(args.python)
    blender = resolve_path(args.blender)
    commands: list[list[str]] = []

    clean_command = [
        python,
        REPO_ROOT / "tools/clean_tof_blaze_sequence.py",
        "--input-root",
        input_root,
        "--output-dir",
        clean_dir,
    ]
    if args.max_frames is not None:
        clean_command.extend(["--max-frames", str(args.max_frames)])
    commands.append([str(item) for item in clean_command])
    run(clean_command, dry_run=args.dry_run)

    clean_ply = clean_dir / "blaze101_tof_canonical_clean_conf_temporal_edges_sor20_std2.ply"
    if not args.skip_segmentation:
        segment_command = [
            python,
            REPO_ROOT / "tools/segment_tof_rocks_ransac_components.py",
            "--input",
            clean_ply,
            "--output-dir",
            segment_dir,
        ]
        commands.append([str(item) for item in segment_command])
        if not args.dry_run:
            require(clean_ply, "Nube ToF limpia")
        run(segment_command, dry_run=args.dry_run)

    if not args.skip_blender:
        require(blender, "Blender")
        source_ply = (
            segment_dir / "blaze101_tof_rocks_clean_combined.ply"
            if not args.skip_segmentation
            else clean_ply
        )
        blend = output_dir / "blaze101_tof_result.blend"
        render = output_dir / "blaze101_tof_result.png"
        blender_command = [
            blender,
            "--background",
            "--python",
            REPO_ROOT / "tools/create_blender_tof_clean_scene.py",
            "--",
            "--input",
            source_ply,
            "--output",
            blend,
            "--render",
            render,
        ]
        commands.append([str(item) for item in blender_command])
        if not args.dry_run:
            require(source_ply, "Nube ToF para Blender")
        run(blender_command, dry_run=args.dry_run)

    write_manifest(
        output_dir,
        experiment="exp02_tof_cleaning",
        arguments=vars(args),
        commands=commands,
    )
    print(f"Salida: {output_dir}")


if __name__ == "__main__":
    main()
