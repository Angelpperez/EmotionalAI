"""Lazo cerrado Android, MediaPipe, clasificador, UDP y Blender."""

from __future__ import annotations

import argparse
import socket
import subprocess
import time
from pathlib import Path

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--duration", type=int, default=60)
    parser.add_argument("--camera-source", default="1")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--rotation", type=int, choices=(0, 90, 180, 270), default=0)
    parser.add_argument("--eye-detector", choices=("mediapipe", "haar"), default="mediapipe")
    parser.add_argument("--port", type=int, default=5055)
    parser.add_argument("--scene", default="D:/emotion3dgs_results/closed_loop_e2e/level_01_closed_loop_difficulty_v4.blend")
    parser.add_argument("--model")
    parser.add_argument("--benchmark-root", default="outputs/emotion_benchmark")
    parser.add_argument("--output-root", default="outputs/experiments/exp04_closed_loop")
    parser.add_argument("--python-eye", default="D:/emotion_eye_env312/Scripts/python.exe")
    parser.add_argument("--python-base", default="D:/emotion3dgs_env/Scripts/python.exe")
    parser.add_argument("--blender", default="C:/Program Files/Blender Foundation/Blender 4.4/blender.exe")
    parser.add_argument("--mirror", action="store_true")
    parser.add_argument("--no-display", action="store_true")
    parser.add_argument("--keep-blender-open", action="store_true")
    parser.add_argument("--skip-camera-check", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def port_available(port: int) -> bool:
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.bind(("127.0.0.1", port))
        return True
    except OSError:
        return False
    finally:
        sock.close()


def main() -> None:
    args = parse_args()
    output_dir = resolve_path(args.output_root) / f"run_{timestamp()}"
    scene = require(resolve_path(args.scene), "Escena Blender")
    model = require(resolve_path(args.model), "Modelo stress3") if args.model else None
    benchmark_root = resolve_path(args.benchmark_root)
    blender = require(resolve_path(args.blender), "Blender")
    python = python_executable(
        args.python_eye if args.eye_detector == "mediapipe" else args.python_base
    )
    controller = REPO_ROOT / "tools/blender_udp_difficulty_controller_experiment.py"
    runtime = REPO_ROOT / "tools/live_closed_loop_end_to_end.py"
    camera_test = REPO_ROOT / "tools/evaluate_eye_detection_source.py"
    commands: list[list[str]] = []

    if not args.dry_run and not port_available(args.port):
        raise RuntimeError(f"El puerto UDP {args.port} esta ocupado")

    if not args.skip_camera_check:
        camera_dir = output_dir / "camera_validation"
        camera_command = [
            python,
            camera_test,
            "--source",
            args.camera_source,
            "--width",
            str(args.width),
            "--height",
            str(args.height),
            "--duration-seconds",
            "15",
            "--detector",
            args.eye_detector,
            "--output-dir",
            camera_dir,
        ]
        commands.append([str(item) for item in camera_command])
        run(camera_command, dry_run=args.dry_run)

    env = child_env(output_dir / "temp")
    env["CLOSED_LOOP_EXPERIMENT_LOG"] = str(
        output_dir / "blender_difficulty_events.csv"
    )
    env["CLOSED_LOOP_EXPERIMENT_PORT"] = str(args.port)

    blender_command = [blender, scene, "--python", controller]
    runtime_command = [
        python,
        runtime,
        "--camera-source",
        args.camera_source,
        "--width",
        str(args.width),
        "--height",
        str(args.height),
        "--rotation",
        str(args.rotation),
        "--eye-detector",
        args.eye_detector,
        "--port",
        str(args.port),
        "--duration-seconds",
        str(args.duration),
        "--output-dir",
        output_dir,
        "--benchmark-root",
        benchmark_root,
    ]
    if model is not None:
        runtime_command.extend(["--model", model])
    if not args.mirror:
        runtime_command.append("--no-mirror")
    if args.no_display:
        runtime_command.append("--no-display")

    commands.extend(
        [
            [str(item) for item in blender_command],
            [str(item) for item in runtime_command],
        ]
    )

    if args.dry_run:
        run(blender_command, dry_run=True, env=env)
        run(runtime_command, dry_run=True, env=env)
    else:
        output_dir.mkdir(parents=True, exist_ok=True)
        process = subprocess.Popen(
            [str(item) for item in blender_command],
            cwd=REPO_ROOT,
            env=env,
        )
        try:
            time.sleep(7)
            if process.poll() is not None:
                raise RuntimeError("Blender se cerro antes de iniciar el listener")
            run(runtime_command, dry_run=False, env=env)
        finally:
            if not args.keep_blender_open and process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()

    write_manifest(
        output_dir,
        experiment="exp04_closed_loop",
        arguments=vars(args),
        commands=commands,
    )
    print(f"Salida: {output_dir}")


if __name__ == "__main__":
    main()
