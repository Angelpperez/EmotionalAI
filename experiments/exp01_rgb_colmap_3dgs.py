"""Reconstruccion RGB con COLMAP, 3DGS y evaluacion hold-out."""

from __future__ import annotations

import argparse
import csv
import os
import shutil
from pathlib import Path

import cv2

from _common import (
    REPO_ROOT,
    child_env,
    python_executable,
    require,
    resolve_path,
    run,
    timestamp,
    write_manifest,
)


STAGES = (
    "frames",
    "colmap-sparse",
    "colmap-dense",
    "prepare-3dgs",
    "train-3dgs",
    "evaluate",
    "blender-points",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--level", default="level_01")
    parser.add_argument("--video", default="captures/1.mp4")
    parser.add_argument("--stages", nargs="+", choices=STAGES, default=list(STAGES))
    parser.add_argument("--output-root", default="outputs/experiments/exp01_rgb_colmap_3dgs")
    parser.add_argument("--external-root", default="D:/emotion3dgs_results")
    parser.add_argument("--frames", type=int, default=160)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--trim-start", type=float, default=2.0)
    parser.add_argument("--trim-end", type=float, default=2.0)
    parser.add_argument("--steps", type=int, default=7000)
    parser.add_argument("--test-every", type=int, default=8)
    parser.add_argument("--colmap", default="D:/colmap-x64-windows-cuda/bin/colmap.exe")
    parser.add_argument("--blender", default="C:/Program Files/Blender Foundation/Blender 4.4/blender.exe")
    parser.add_argument("--python", default="D:/emotion3dgs_env/Scripts/python.exe")
    parser.add_argument("--trainer", default="external/gsplat_v1_5_3/examples/simple_trainer.py")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def extract_frames(
    video: Path,
    image_dir: Path,
    manifest: Path,
    *,
    level: str,
    target_frames: int,
    target_width: int,
    trim_start: float,
    trim_end: float,
) -> None:
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"No se pudo abrir {video}")
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    start = int(trim_start * fps)
    end = frame_count - int(trim_end * fps) - 1
    if end <= start:
        raise ValueError("El recorte temporal no deja fotogramas utiles")
    step = (end - start) / max(1, target_frames - 1)
    indices = [round(start + index * step) for index in range(target_frames)]
    image_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for output_index, frame_index in enumerate(indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = cap.read()
        if not ok:
            raise RuntimeError(f"No se pudo leer el frame {frame_index}")
        scale = target_width / frame.shape[1]
        resized = cv2.resize(
            frame,
            (target_width, round(frame.shape[0] * scale)),
            interpolation=cv2.INTER_AREA,
        )
        output = image_dir / f"{level}_{output_index:04d}.jpg"
        cv2.imwrite(str(output), resized, [cv2.IMWRITE_JPEG_QUALITY, 95])
        rows.append(
            {
                "level": level,
                "source_video": str(video),
                "source_frame": frame_index,
                "time_seconds": frame_index / fps,
                "output": str(output),
            }
        )
    cap.release()
    manifest.parent.mkdir(parents=True, exist_ok=True)
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    run_dir = resolve_path(args.output_root) / f"run_{timestamp()}"
    level_root = run_dir / args.level
    image_dir = level_root / "images"
    sparse_root = level_root / "colmap_sparse"
    dense_root = level_root / "colmap_dense"
    gs_data = level_root / "gs_dataset"
    gs_result = resolve_path(args.external_root) / f"{args.level}_3dgs_{args.steps}"
    commands: list[list[str]] = []

    video = resolve_path(args.video)
    colmap = resolve_path(args.colmap)
    blender = resolve_path(args.blender)
    python = python_executable(args.python)
    trainer = resolve_path(args.trainer)
    database = sparse_root / "database.db"
    sparse_models = sparse_root / "sparse"

    if "frames" in args.stages:
        require(video, "Video")
        frame_command = [
            "internal:extract_frames",
            "--video",
            str(video),
            "--output",
            str(image_dir),
            "--frames",
            str(args.frames),
            "--width",
            str(args.width),
            "--trim-start",
            str(args.trim_start),
            "--trim-end",
            str(args.trim_end),
        ]
        commands.append(frame_command)
        if args.dry_run:
            print(f"$ extraer {args.frames} fotogramas de {video} -> {image_dir}")
        else:
            extract_frames(
                video,
                image_dir,
                run_dir / "frame_manifest.csv",
                level=args.level,
                target_frames=args.frames,
                target_width=args.width,
                trim_start=args.trim_start,
                trim_end=args.trim_end,
            )

    if "colmap-sparse" in args.stages:
        require(colmap, "COLMAP")
        if not args.dry_run:
            require(image_dir, "Fotogramas")
        sparse_models.mkdir(parents=True, exist_ok=True)
        sparse_commands = [
            [colmap, "feature_extractor", "--database_path", database, "--image_path", image_dir, "--ImageReader.single_camera", "1"],
            [colmap, "sequential_matcher", "--database_path", database],
            [colmap, "mapper", "--database_path", database, "--image_path", image_dir, "--output_path", sparse_models],
        ]
        for command in sparse_commands:
            commands.append([str(item) for item in command])
            run(command, dry_run=args.dry_run)

    sparse_model = sparse_models / "0"
    if "colmap-dense" in args.stages:
        require(colmap, "COLMAP")
        if not args.dry_run:
            require(sparse_model, "Modelo COLMAP")
        dense_commands = [
            [colmap, "image_undistorter", "--image_path", image_dir, "--input_path", sparse_model, "--output_path", dense_root, "--output_type", "COLMAP"],
            [colmap, "patch_match_stereo", "--workspace_path", dense_root, "--workspace_format", "COLMAP", "--PatchMatchStereo.max_image_size", "960", "--PatchMatchStereo.geom_consistency", "1", "--PatchMatchStereo.cache_size", "8"],
            [colmap, "stereo_fusion", "--workspace_path", dense_root, "--workspace_format", "COLMAP", "--input_type", "geometric", "--output_path", dense_root / "fused_geometric.ply"],
            [colmap, "poisson_mesher", "--input_path", dense_root / "fused_geometric.ply", "--output_path", dense_root / "meshed_poisson_depth11.ply", "--PoissonMeshing.depth", "11", "--PoissonMeshing.trim", "10"],
        ]
        for command in dense_commands:
            commands.append([str(item) for item in command])
            run(command, dry_run=args.dry_run)

    if "prepare-3dgs" in args.stages:
        if not args.dry_run:
            require(image_dir, "Fotogramas")
            require(sparse_model, "Modelo COLMAP")
        if args.dry_run:
            print(f"$ preparar dataset 3DGS {gs_data}")
        else:
            if gs_data.exists():
                raise FileExistsError(f"El dataset ya existe: {gs_data}")
            shutil.copytree(image_dir, gs_data / "images")
            shutil.copytree(sparse_model, gs_data / "sparse/0")

    if "train-3dgs" in args.stages:
        require(trainer, "Entrenador gsplat")
        if not args.dry_run:
            require(gs_data, "Dataset 3DGS")
        command = [
            python,
            trainer,
            "default",
            "--data-dir",
            gs_data,
            "--result-dir",
            gs_result,
            "--data-factor",
            "1",
            "--test-every",
            str(args.test_every),
            "--max-steps",
            str(args.steps),
            "--eval-steps",
            str(args.steps),
            "--save-steps",
            str(args.steps),
            "--ply-steps",
            str(args.steps),
            "--save-ply",
            "--disable-video",
            "--disable-viewer",
            "--packed",
        ]
        commands.append([str(item) for item in command])
        env = child_env()
        compatibility = REPO_ROOT / "tools/gsplat_compat"
        env["PYTHONPATH"] = os.pathsep.join(
            part
            for part in (str(compatibility), env.get("PYTHONPATH", ""))
            if part
        )
        run(command, dry_run=args.dry_run, env=env)

    ply = gs_result / "ply" / f"point_cloud_{args.steps - 1}.ply"
    if "evaluate" in args.stages:
        renders = gs_result / "renders"
        report = run_dir / "holdout_report"
        command = [
            python,
            REPO_ROOT / "tools/summarize_3dgs_holdout_metrics.py",
            "--renders-dir",
            renders,
            "--images-dir",
            image_dir,
            "--test-every",
            str(args.test_every),
            "--step",
            str(args.steps - 1),
            "--output-dir",
            report,
        ]
        commands.append([str(item) for item in command])
        if not args.dry_run:
            require(renders, "Renders 3DGS")
        run(command, dry_run=args.dry_run)

    if "blender-points" in args.stages:
        rgb_ply = run_dir / f"{args.level}_3dgs_rgb_points.ply"
        blend = run_dir / f"{args.level}_3dgs_rgb_points.blend"
        convert = [python, REPO_ROOT / "tools/convert_3dgs_ply_to_rgb_points.py", ply, rgb_ply]
        create = [
            blender,
            "--background",
            "--python",
            REPO_ROOT / "tools/create_blender_rgb_point_scene.py",
            "--",
            "--input",
            rgb_ply,
            "--output",
            blend,
            "--name",
            f"{args.level}_3dgs_rgb_points",
        ]
        for command in (convert, create):
            commands.append([str(item) for item in command])
            if not args.dry_run and command is convert:
                require(ply, "PLY 3DGS")
            run(command, dry_run=args.dry_run)

    write_manifest(
        run_dir,
        experiment="exp01_rgb_colmap_3dgs",
        arguments=vars(args),
        commands=commands,
    )
    print(f"Salida: {run_dir}")


if __name__ == "__main__":
    main()
