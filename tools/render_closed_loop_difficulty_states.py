"""Render easy, normal and hard control images from the experimental scene."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import bpy


STATES = [
    ("easy", 0, "ALTO -> FACIL"),
    ("normal", 1, "MEDIO -> NORMAL"),
    ("hard", 2, "BAJO -> DIFICIL"),
]


def parse_args() -> argparse.Namespace:
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args(argv)


def set_level(level: int, label: str) -> int:
    changed = 0
    for obj in bpy.context.scene.objects:
        if "closed_loop_min_difficulty" not in obj:
            continue
        minimum = int(obj.get("closed_loop_min_difficulty", 0))
        maximum = int(obj.get("closed_loop_max_difficulty", 2))
        hidden = not (minimum <= level <= maximum)
        obj.hide_viewport = hidden
        obj.hide_render = hidden
        if not hidden:
            changed += 1
    status = bpy.data.objects.get("ClosedLoop_Status_Text")
    if status is not None and status.type == "FONT":
        difficulty = label.split("->")[-1].strip()
        status.data.body = f"DIFICULTAD: {difficulty}"
    return changed


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    scene = bpy.context.scene
    if scene.camera is None:
        raise RuntimeError("The experimental scene has no active camera")
    for name, level, label in STATES:
        visible = set_level(level, label)
        output = args.output_dir / f"difficulty_{name}.png"
        scene.render.filepath = str(output)
        bpy.ops.render.render(write_still=True)
        print(f"[ClosedLoopRender] {name}: visible={visible} output={output}")


if __name__ == "__main__":
    main()
