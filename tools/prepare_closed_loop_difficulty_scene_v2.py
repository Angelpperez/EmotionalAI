"""Create a camera-aligned closed-loop demonstration scene.

The reconstruction is not rotated or modified. A compact set of inspection
markers is parented to the original scene camera, so it remains correctly
oriented regardless of the COLMAP world axes.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import bpy
from mathutils import Matrix, Quaternion, Vector


COLLECTION_NAME = "ClosedLoop_Difficulty_Experiment_V2"


def parse_args() -> argparse.Namespace:
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--cameras-txt", required=True, type=Path)
    parser.add_argument("--images-txt", required=True, type=Path)
    parser.add_argument("--image-name", default="level_01_0088.jpg")
    return parser.parse_args(argv)


def material(name: str, color: tuple[float, float, float, float]):
    found = bpy.data.materials.get(name)
    if found is not None:
        return found
    created = bpy.data.materials.new(name)
    created.use_nodes = True
    created.diffuse_color = color
    nodes = created.node_tree.nodes
    links = created.node_tree.links
    nodes.clear()
    output = nodes.new("ShaderNodeOutputMaterial")
    emission = nodes.new("ShaderNodeEmission")
    emission.inputs["Color"].default_value = color
    emission.inputs["Strength"].default_value = 2.0
    links.new(emission.outputs["Emission"], output.inputs["Surface"])
    return created


def move_to_collection(obj, collection) -> None:
    for current in list(obj.users_collection):
        current.objects.unlink(obj)
    collection.objects.link(obj)


def tag(obj, level: int) -> None:
    obj["closed_loop_min_difficulty"] = level
    obj["closed_loop_max_difficulty"] = level
    obj["closed_loop_experiment_object"] = True
    obj["closed_loop_role"] = "inspection_marker"


def add_marker(
    camera,
    collection,
    name: str,
    level: int,
    x: float,
    y: float,
    depth: float,
    radius: float,
    marker_material,
) -> None:
    bpy.ops.mesh.primitive_torus_add(
        major_radius=radius,
        minor_radius=radius * 0.17,
        major_segments=40,
        minor_segments=12,
    )
    marker = bpy.context.object
    marker.name = name
    marker.data.materials.append(marker_material)
    marker.parent = camera
    marker.matrix_parent_inverse.identity()
    marker.location = (x, y, -depth)
    marker.rotation_euler = (0.0, 0.0, 0.0)
    tag(marker, level)
    move_to_collection(marker, collection)


def add_status_text(camera, collection, white, depth: float, view_width: float):
    bpy.ops.object.text_add()
    text = bpy.context.object
    text.name = "ClosedLoop_Status_Text"
    text.data.body = "DIFICULTAD: NORMAL"
    text.data.align_x = "CENTER"
    text.data.align_y = "CENTER"
    text.data.size = view_width * 0.025
    text.data.extrude = view_width * 0.0004
    text.data.materials.append(white)
    text.parent = camera
    text.matrix_parent_inverse.identity()
    text.location = (0.0, view_width * 0.19, -depth)
    text.rotation_euler = (0.0, 0.0, 0.0)
    text["closed_loop_status_display"] = True
    text["closed_loop_role"] = "status_only"
    move_to_collection(text, collection)


def add_level_markers(
    camera,
    collection,
    level: int,
    count: int,
    depth: float,
    view_width: float,
    marker_material,
) -> None:
    spacing = view_width * 0.09
    radius = view_width * 0.018
    for index in range(count):
        x = (index - (count - 1) / 2.0) * spacing
        y = -view_width * 0.18
        add_marker(
            camera,
            collection,
            f"ClosedLoop_Level{level}_InspectionMarker_{index + 1:02d}",
            level,
            x,
            y,
            depth,
            radius,
            marker_material,
        )


def set_initial_visibility() -> None:
    for obj in bpy.context.scene.objects:
        if "closed_loop_min_difficulty" not in obj:
            continue
        hidden = int(obj["closed_loop_min_difficulty"]) != 1
        obj.hide_viewport = hidden
        obj.hide_render = hidden


def read_colmap_camera(path: Path) -> dict[str, float | int | str]:
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        model = parts[1]
        if model != "SIMPLE_RADIAL":
            raise ValueError(f"Unsupported COLMAP camera model: {model}")
        return {
            "camera_id": int(parts[0]),
            "model": model,
            "width": int(parts[2]),
            "height": int(parts[3]),
            "focal": float(parts[4]),
            "cx": float(parts[5]),
            "cy": float(parts[6]),
            "radial_k": float(parts[7]),
        }
    raise ValueError(f"No camera found in {path}")


def read_colmap_pose(path: Path, image_name: str) -> dict[str, object]:
    lines = path.read_text(encoding="utf-8").splitlines()
    for index in range(0, len(lines)):
        line = lines[index].strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) >= 10 and parts[9] == image_name:
            return {
                "image_id": int(parts[0]),
                "qvec": tuple(float(value) for value in parts[1:5]),
                "tvec": Vector(tuple(float(value) for value in parts[5:8])),
                "camera_id": int(parts[8]),
                "image_name": parts[9],
            }
    raise ValueError(f"Image pose not found: {image_name}")


def create_colmap_camera(
    camera_data: dict[str, float | int | str],
    pose: dict[str, object],
):
    existing = bpy.data.objects.get("ClosedLoop_COLMAP_Reference_Camera")
    if existing is not None:
        bpy.data.objects.remove(existing, do_unlink=True)
    data = bpy.data.cameras.new("ClosedLoop_COLMAP_Reference_Camera_Data")
    camera = bpy.data.objects.new("ClosedLoop_COLMAP_Reference_Camera", data)
    bpy.context.scene.collection.objects.link(camera)

    qw, qx, qy, qz = pose["qvec"]
    world_to_camera = Quaternion((qw, qx, qy, qz)).to_matrix()
    camera_to_world_rotation = world_to_camera.transposed()
    camera_center = -(camera_to_world_rotation @ pose["tvec"])
    colmap_to_blender_camera = Matrix.Diagonal((1.0, -1.0, -1.0))
    blender_rotation = camera_to_world_rotation @ colmap_to_blender_camera
    camera.matrix_world = (
        Matrix.Translation(camera_center) @ blender_rotation.to_4x4()
    )

    data.type = "PERSP"
    data.sensor_fit = "HORIZONTAL"
    data.sensor_width = 36.0
    data.lens = (
        float(camera_data["focal"])
        * data.sensor_width
        / int(camera_data["width"])
    )
    data.shift_x = (
        float(camera_data["cx"]) - int(camera_data["width"]) / 2.0
    ) / int(camera_data["width"])
    data.shift_y = -(
        float(camera_data["cy"]) - int(camera_data["height"]) / 2.0
    ) / int(camera_data["height"])
    data.clip_start = 0.01
    data.clip_end = 1000.0
    camera["colmap_image_name"] = pose["image_name"]
    camera["colmap_radial_k_not_rendered"] = camera_data["radial_k"]
    return camera


def main() -> None:
    args = parse_args()
    scene = bpy.context.scene
    camera_data = read_colmap_camera(args.cameras_txt)
    pose = read_colmap_pose(args.images_txt, args.image_name)
    if int(pose["camera_id"]) != int(camera_data["camera_id"]):
        raise ValueError("COLMAP image and camera IDs do not match")
    camera = create_colmap_camera(camera_data, pose)
    scene.camera = camera

    old = bpy.data.collections.get(COLLECTION_NAME)
    if old is not None:
        for obj in list(old.objects):
            bpy.data.objects.remove(obj, do_unlink=True)
        bpy.data.collections.remove(old)

    collection = bpy.data.collections.new(COLLECTION_NAME)
    scene.collection.children.link(collection)
    green = material("ClosedLoop_V2_Easy_Green", (0.05, 0.85, 0.20, 1.0))
    amber = material("ClosedLoop_V2_Normal_Amber", (1.0, 0.62, 0.03, 1.0))
    red = material("ClosedLoop_V2_Hard_Red", (0.95, 0.08, 0.04, 1.0))
    white = material("ClosedLoop_V2_Status_White", (1.0, 1.0, 1.0, 1.0))

    depth = max(camera.data.clip_start * 20.0, 0.5)
    view_width = 2.0 * depth * math.tan(camera.data.angle_x / 2.0)
    add_level_markers(camera, collection, 0, 3, depth, view_width, green)
    add_level_markers(camera, collection, 1, 5, depth, view_width, amber)
    add_level_markers(camera, collection, 2, 7, depth, view_width, red)
    add_status_text(camera, collection, white, depth, view_width)
    set_initial_visibility()

    scene.render.engine = "BLENDER_EEVEE_NEXT"
    scene.render.resolution_x = 1280
    scene.render.resolution_y = 720
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene["closed_loop_experiment_ready"] = True
    scene["closed_loop_experiment_version"] = 2
    scene["closed_loop_task_note"] = (
        "Markers visualize task load only; they are not calibration targets."
    )
    scene["closed_loop_reference_image"] = args.image_name
    scene["closed_loop_difficulty"] = "normal"
    scene["closed_loop_difficulty_level"] = 1

    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=str(output))
    print(f"[ClosedLoopPrepV2] camera={camera.name}")
    print(f"[ClosedLoopPrepV2] output={output}")
    print("[ClosedLoopPrepV2] markers easy=3 normal=5 hard=7")


if __name__ == "__main__":
    main()
