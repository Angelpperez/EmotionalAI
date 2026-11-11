"""Add observable difficulty objects to a copy of an existing Blender scene."""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import bpy
from mathutils import Vector


COLLECTION_NAME = "ClosedLoop_Difficulty_Experiment"


def parse_args() -> argparse.Namespace:
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args(argv)


def scene_bounds() -> tuple[Vector, Vector, Vector, float]:
    points = []
    for obj in bpy.context.scene.objects:
        if obj.type != "MESH" or obj.name.startswith("ClosedLoop_"):
            continue
        points.extend(obj.matrix_world @ Vector(corner) for corner in obj.bound_box)
    if not points:
        return Vector((-1, -1, -1)), Vector((1, 1, 1)), Vector(), 3.0
    minimum = Vector(tuple(min(point[i] for point in points) for i in range(3)))
    maximum = Vector(tuple(max(point[i] for point in points) for i in range(3)))
    center = (minimum + maximum) / 2.0
    diagonal = max((maximum - minimum).length, 1.0)
    return minimum, maximum, center, diagonal


def material(name: str, color: tuple[float, float, float, float]):
    existing = bpy.data.materials.get(name)
    if existing:
        return existing
    created = bpy.data.materials.new(name)
    created.diffuse_color = color
    created.use_nodes = True
    bsdf = created.node_tree.nodes.get("Principled BSDF")
    if bsdf:
        bsdf.inputs["Base Color"].default_value = color
        bsdf.inputs["Roughness"].default_value = 0.4
        bsdf.inputs["Metallic"].default_value = 0.05
    return created


def move_to_collection(obj, collection) -> None:
    for current in list(obj.users_collection):
        current.objects.unlink(obj)
    collection.objects.link(obj)


def tag(obj, level: int) -> None:
    obj["closed_loop_min_difficulty"] = level
    obj["closed_loop_max_difficulty"] = level
    obj["closed_loop_experiment_object"] = True


def add_easy_guides(
    collection,
    center: Vector,
    diagonal: float,
    green,
) -> None:
    radius = diagonal * 0.025
    spacing = diagonal * 0.075
    for index in range(5):
        location = center + Vector(
            ((index - 2) * spacing, -0.12 * diagonal, 0.08 * diagonal)
        )
        bpy.ops.mesh.primitive_cone_add(
            vertices=24,
            radius1=radius,
            radius2=0.0,
            depth=radius * 2.5,
            location=location,
            rotation=(0.0, math.pi / 2.0, 0.0),
        )
        obj = bpy.context.object
        obj.name = f"ClosedLoop_Easy_Guide_{index + 1:02d}"
        obj.data.materials.append(green)
        tag(obj, 0)
        move_to_collection(obj, collection)


def add_normal_targets(
    collection,
    center: Vector,
    diagonal: float,
    yellow,
) -> None:
    radius = diagonal * 0.022
    for index in range(6):
        angle = 2.0 * math.pi * index / 6.0
        location = center + Vector(
            (
                math.cos(angle) * diagonal * 0.18,
                math.sin(angle) * diagonal * 0.12,
                diagonal * 0.10,
            )
        )
        bpy.ops.mesh.primitive_torus_add(
            major_radius=radius * 1.8,
            minor_radius=radius * 0.35,
            major_segments=32,
            minor_segments=12,
            location=location,
        )
        obj = bpy.context.object
        obj.name = f"ClosedLoop_Normal_Target_{index + 1:02d}"
        obj.data.materials.append(yellow)
        tag(obj, 1)
        move_to_collection(obj, collection)


def add_hard_obstacles(
    collection,
    center: Vector,
    diagonal: float,
    red,
) -> None:
    size = diagonal * 0.045
    positions = [
        (-0.20, -0.12),
        (-0.12, 0.02),
        (-0.04, -0.08),
        (0.04, 0.08),
        (0.12, -0.02),
        (0.20, 0.12),
        (-0.20, 0.12),
        (0.20, -0.12),
    ]
    for index, (x_factor, y_factor) in enumerate(positions):
        location = center + Vector(
            (x_factor * diagonal, y_factor * diagonal, 0.10 * diagonal)
        )
        bpy.ops.mesh.primitive_cube_add(
            size=size,
            location=location,
        )
        obj = bpy.context.object
        obj.name = f"ClosedLoop_Hard_Obstacle_{index + 1:02d}"
        obj.scale.z = 2.0
        obj.data.materials.append(red)
        tag(obj, 2)
        move_to_collection(obj, collection)


def add_status_text(
    collection,
    center: Vector,
    diagonal: float,
    white,
) -> None:
    location = center + Vector((0.0, 0.0, diagonal * 0.24))
    bpy.ops.object.text_add(location=location)
    obj = bpy.context.object
    obj.name = "ClosedLoop_Status_Text"
    obj.data.body = "MEDIO -> NORMAL"
    obj.data.align_x = "CENTER"
    obj.data.align_y = "CENTER"
    obj.data.size = diagonal * 0.045
    obj.data.extrude = diagonal * 0.001
    obj.data.materials.append(white)
    obj["closed_loop_status_display"] = True
    move_to_collection(obj, collection)


def add_experiment_camera_and_light(
    collection,
    center: Vector,
    diagonal: float,
) -> None:
    camera_target = center + Vector((0.0, 0.0, diagonal * 0.08))
    camera_location = center + Vector(
        (0.0, -diagonal * 0.62, diagonal * 0.36)
    )
    bpy.ops.object.camera_add(location=camera_location)
    camera = bpy.context.object
    camera.name = "ClosedLoop_Experiment_Camera"
    camera.rotation_euler = (
        camera_target - camera.location
    ).to_track_quat("-Z", "Y").to_euler()
    camera.data.lens = 48.0
    camera.data.clip_start = max(diagonal * 0.0001, 0.001)
    camera.data.clip_end = diagonal * 20.0
    move_to_collection(camera, collection)
    bpy.context.scene.camera = camera

    bpy.ops.object.light_add(
        type="AREA",
        location=center + Vector(
            (-diagonal * 0.25, -diagonal * 0.25, diagonal * 0.55)
        ),
    )
    light = bpy.context.object
    light.name = "ClosedLoop_Experiment_Light"
    light.data.energy = 1200.0
    light.data.shape = "DISK"
    light.data.size = diagonal * 0.35
    light.rotation_euler = (
        camera_target - light.location
    ).to_track_quat("-Z", "Y").to_euler()
    move_to_collection(light, collection)


def set_initial_visibility() -> None:
    for obj in bpy.context.scene.objects:
        if "closed_loop_min_difficulty" not in obj:
            continue
        level = int(obj["closed_loop_min_difficulty"])
        hidden = level != 1
        obj.hide_viewport = hidden
        obj.hide_render = hidden


def main() -> None:
    args = parse_args()
    old = bpy.data.collections.get(COLLECTION_NAME)
    if old is not None:
        for obj in list(old.objects):
            bpy.data.objects.remove(obj, do_unlink=True)
        bpy.data.collections.remove(old)

    collection = bpy.data.collections.new(COLLECTION_NAME)
    bpy.context.scene.collection.children.link(collection)
    _, _, center, diagonal = scene_bounds()
    green = material("ClosedLoop_Easy_Green", (0.08, 0.75, 0.18, 1.0))
    yellow = material("ClosedLoop_Normal_Yellow", (0.95, 0.65, 0.05, 1.0))
    red = material("ClosedLoop_Hard_Red", (0.85, 0.08, 0.04, 1.0))
    white = material("ClosedLoop_Status_White", (0.95, 0.95, 0.95, 1.0))

    add_easy_guides(collection, center, diagonal, green)
    add_normal_targets(collection, center, diagonal, yellow)
    add_hard_obstacles(collection, center, diagonal, red)
    add_status_text(collection, center, diagonal, white)
    add_experiment_camera_and_light(collection, center, diagonal)
    set_initial_visibility()

    scene = bpy.context.scene
    scene.render.engine = "BLENDER_EEVEE_NEXT"
    scene.render.resolution_x = 1280
    scene.render.resolution_y = 720
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene["closed_loop_experiment_ready"] = True
    scene["closed_loop_difficulty_levels"] = "easy=0, normal=1, hard=2"
    scene["closed_loop_base_scene"] = bpy.data.filepath
    scene["closed_loop_state"] = "medium"
    scene["closed_loop_difficulty"] = "normal"
    scene["closed_loop_difficulty_level"] = 1

    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=str(output))
    print(f"[ClosedLoopPrep] output={output}")
    print("[ClosedLoopPrep] easy_objects=5 normal_objects=6 hard_objects=8")


if __name__ == "__main__":
    main()
