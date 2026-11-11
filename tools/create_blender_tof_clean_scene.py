"""Create a Blender inspection scene for the cleaned Basler blaze ToF cloud.

The vertex RGB channels in the cleaned PLY contain the Basler intensity image
in grayscale. They are not natural RGB colors and the scene is not a 3DGS
rasterization.

Example:

blender --background --python tools/create_blender_tof_clean_scene.py -- ^
  --input outputs/tof_cleaning/blaze101_canonical_270/blaze101_tof_canonical_clean_conf_temporal_edges_sor20_std2.ply ^
  --output outputs/tof_cleaning/blaze101_canonical_270/blaze101_tof_clean.blend ^
  --render outputs/tof_cleaning/blaze101_canonical_270/blaze101_tof_clean_preview.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import bpy
from mathutils import Vector


def parse_args() -> argparse.Namespace:
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--render", type=Path)
    parser.add_argument("--name", default="Basler_blaze101_ToF_clean")
    parser.add_argument(
        "--radius-scale",
        type=float,
        default=1800.0,
        help="Point radius is the bounding-box diagonal divided by this value",
    )
    return parser.parse_args(argv)


def reset_scene() -> None:
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()
    for datablocks in (
        bpy.data.meshes,
        bpy.data.curves,
        bpy.data.materials,
        bpy.data.cameras,
        bpy.data.lights,
    ):
        for datablock in list(datablocks):
            if datablock.users == 0:
                datablocks.remove(datablock)


def import_ply(path: Path, name: str) -> bpy.types.Object:
    if hasattr(bpy.ops.wm, "ply_import"):
        bpy.ops.wm.ply_import(filepath=str(path))
    else:
        bpy.ops.import_mesh.ply(filepath=str(path))
    obj = bpy.context.object
    obj.name = name
    obj.data.name = f"{name}_mesh"
    return obj


def bounds(obj: bpy.types.Object) -> tuple[Vector, Vector, Vector, float]:
    corners = [obj.matrix_world @ Vector(corner) for corner in obj.bound_box]
    minimum = Vector(tuple(min(point[i] for point in corners) for i in range(3)))
    maximum = Vector(tuple(max(point[i] for point in corners) for i in range(3)))
    center = (minimum + maximum) / 2.0
    diagonal = max((maximum - minimum).length, 1.0)
    return minimum, maximum, center, diagonal


def add_intensity_material(obj: bpy.types.Object) -> None:
    material = bpy.data.materials.new("Basler_ToF_intensity_grayscale")
    material.use_nodes = True
    nodes = material.node_tree.nodes
    links = material.node_tree.links
    nodes.clear()

    output = nodes.new("ShaderNodeOutputMaterial")
    emission = nodes.new("ShaderNodeEmission")
    attribute = nodes.new("ShaderNodeAttribute")
    rgb_to_bw = nodes.new("ShaderNodeRGBToBW")
    normalize = nodes.new("ShaderNodeMapRange")
    attribute.attribute_name = (
        obj.data.color_attributes[0].name
        if getattr(obj.data, "color_attributes", None)
        and obj.data.color_attributes
        else "Col"
    )
    normalize.inputs["From Min"].default_value = 0.05
    normalize.inputs["From Max"].default_value = 0.25
    normalize.inputs["To Min"].default_value = 0.08
    normalize.inputs["To Max"].default_value = 1.0
    normalize.clamp = True
    emission.inputs["Strength"].default_value = 1.25
    links.new(attribute.outputs["Color"], rgb_to_bw.inputs["Color"])
    links.new(rgb_to_bw.outputs["Val"], normalize.inputs["Value"])
    links.new(normalize.outputs["Result"], emission.inputs["Color"])
    links.new(emission.outputs["Emission"], output.inputs["Surface"])
    obj.data.materials.append(material)


def add_point_geometry(obj: bpy.types.Object, radius: float) -> None:
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    bpy.ops.node.new_geometry_nodes_modifier()
    modifier = obj.modifiers[-1]
    modifier.name = "ToF points with inspection radius"
    group = modifier.node_group
    group.name = "Basler_ToF_Point_Inspection"

    input_node = next(
        node for node in group.nodes if node.bl_idname == "NodeGroupInput"
    )
    output_node = next(
        node for node in group.nodes if node.bl_idname == "NodeGroupOutput"
    )
    for link in list(group.links):
        group.links.remove(link)

    mesh_to_points = group.nodes.new("GeometryNodeMeshToPoints")
    mesh_to_points.inputs["Radius"].default_value = radius
    group.links.new(input_node.outputs["Geometry"], mesh_to_points.inputs["Mesh"])
    group.links.new(mesh_to_points.outputs["Points"], output_node.inputs["Geometry"])


def add_camera(obj: bpy.types.Object) -> None:
    minimum, maximum, center, diagonal = bounds(obj)
    extent = maximum - minimum
    camera_position = center + Vector(
        (0.0, -0.20 * diagonal, 0.95 * diagonal)
    )
    bpy.ops.object.camera_add(location=camera_position)
    camera = bpy.context.object
    camera.name = "Basler_ToF_Inspection_Camera"
    camera.rotation_euler = (center - camera.location).to_track_quat(
        "-Z", "Y"
    ).to_euler()
    camera.data.type = "ORTHO"
    camera.data.ortho_scale = max(extent.x, extent.y) * 1.18
    camera.data.clip_start = 1.0
    camera.data.clip_end = diagonal * 10.0
    bpy.context.scene.camera = camera


def configure_scene(render_path: Path | None) -> None:
    scene = bpy.context.scene
    scene.render.engine = "BLENDER_EEVEE_NEXT"
    scene.render.resolution_x = 1280
    scene.render.resolution_y = 720
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.film_transparent = False
    scene.world.color = (0.015, 0.015, 0.015)
    scene.view_settings.view_transform = "Standard"
    scene.view_settings.look = "Medium High Contrast"
    if render_path is not None:
        render_path.parent.mkdir(parents=True, exist_ok=True)
        scene.render.filepath = str(render_path)


def main() -> None:
    args = parse_args()
    input_path = args.input.resolve()
    output_path = args.output.resolve()
    render_path = args.render.resolve() if args.render else None
    if not input_path.exists():
        raise FileNotFoundError(input_path)

    reset_scene()
    obj = import_ply(input_path, args.name)
    _, _, _, diagonal = bounds(obj)
    point_radius = max(diagonal / args.radius_scale, 0.1)
    add_intensity_material(obj)
    add_point_geometry(obj, point_radius)
    add_camera(obj)
    configure_scene(render_path)

    obj["source_sensor"] = "Basler blaze-101 ToF"
    obj["coordinate_units"] = "millimetres"
    obj["display_channels"] = "intensity grayscale, not natural RGB"
    obj["representation"] = "cleaned canonical point cloud, not 3DGS"
    obj["point_radius_mm"] = point_radius

    output_path.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=str(output_path))
    if render_path is not None:
        bpy.ops.render.render(write_still=True)

    print(f"vertices: {len(obj.data.vertices)}")
    print(f"point_radius_mm: {point_radius}")
    print(f"blend: {output_path}")
    if render_path is not None:
        print(f"render: {render_path}")


if __name__ == "__main__":
    main()
