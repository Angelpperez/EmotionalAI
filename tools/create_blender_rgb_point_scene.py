"""Create a Blender inspection scene from a standard RGB point PLY.

Run with Blender, for example:

blender --background --python tools/create_blender_rgb_point_scene.py -- ^
  --input point_cloud_rgb.ply --output scene.blend --name level_01_rgb
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import bpy
from mathutils import Vector


def parse_args() -> argparse.Namespace:
    if "--" in sys.argv:
        argv = sys.argv[sys.argv.index("--") + 1 :]
    else:
        argv = []

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="Standard RGB point PLY")
    parser.add_argument("--output", required=True, type=Path, help="Output .blend path")
    parser.add_argument("--name", default="rgb_point_cloud", help="Object name")
    parser.add_argument("--radius-scale", type=float, default=1200.0, help="Bounding-box diagonal divisor")
    return parser.parse_args(argv)


def reset_scene() -> None:
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()


def import_ply(path: Path, name: str) -> bpy.types.Object:
    if hasattr(bpy.ops.wm, "ply_import"):
        bpy.ops.wm.ply_import(filepath=str(path))
    else:
        bpy.ops.import_mesh.ply(filepath=str(path))

    obj = bpy.context.object
    obj.name = name
    obj.data.name = f"{name}_mesh"
    return obj


def add_vertex_color_material(obj: bpy.types.Object) -> None:
    material = bpy.data.materials.new("RGB_from_3DGS_DC")
    material.use_nodes = True
    material.diffuse_color = (0.85, 0.85, 0.85, 1.0)

    nodes = material.node_tree.nodes
    links = material.node_tree.links
    bsdf = nodes.get("Principled BSDF")
    if bsdf is None:
        obj.data.materials.append(material)
        return

    attr_name = "Col"
    if getattr(obj.data, "color_attributes", None) and obj.data.color_attributes:
        attr_name = obj.data.color_attributes[0].name

    attr = nodes.new(type="ShaderNodeAttribute")
    attr.attribute_name = attr_name
    links.new(attr.outputs["Color"], bsdf.inputs["Base Color"])
    if "Alpha" in bsdf.inputs:
        links.new(attr.outputs["Alpha"], bsdf.inputs["Alpha"])

    material.use_nodes = True
    obj.data.materials.append(material)


def compute_radius(obj: bpy.types.Object, radius_scale: float) -> float:
    bbox = [obj.matrix_world @ Vector(corner) for corner in obj.bound_box]
    mins = Vector((min(v.x for v in bbox), min(v.y for v in bbox), min(v.z for v in bbox)))
    maxs = Vector((max(v.x for v in bbox), max(v.y for v in bbox), max(v.z for v in bbox)))
    diagonal = (maxs - mins).length
    return max(diagonal / radius_scale, 0.0005)


def add_points_geometry_nodes(obj: bpy.types.Object, point_radius: float) -> None:
    bpy.ops.object.select_all(action="DESELECT")
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    bpy.ops.node.new_geometry_nodes_modifier()
    modifier = obj.modifiers[-1]
    modifier.name = "3DGS center points with radius"
    group = modifier.node_group
    group.name = "3DGS_RGB_Point_Render"

    input_node = next(node for node in group.nodes if node.bl_idname == "NodeGroupInput")
    output_node = next(node for node in group.nodes if node.bl_idname == "NodeGroupOutput")
    for link in list(group.links):
        group.links.remove(link)

    mesh_to_points = group.nodes.new(type="GeometryNodeMeshToPoints")
    mesh_to_points.location = (0, 0)
    mesh_to_points.inputs["Radius"].default_value = point_radius

    set_radius = group.nodes.new(type="GeometryNodeSetPointRadius")
    set_radius.location = (220, 0)
    set_radius.inputs["Radius"].default_value = point_radius

    group.links.new(input_node.outputs["Geometry"], mesh_to_points.inputs["Mesh"])
    group.links.new(mesh_to_points.outputs["Points"], set_radius.inputs["Points"])
    group.links.new(set_radius.outputs["Points"], output_node.inputs["Geometry"])


def add_camera_and_lighting(obj: bpy.types.Object) -> None:
    bpy.ops.object.light_add(type="AREA", location=(0.0, -4.0, 5.0))
    light = bpy.context.object
    light.name = "ClosedLoop_Key_Light"
    light.data.energy = 650.0
    light.data.size = 5.0

    bbox = [obj.matrix_world @ Vector(corner) for corner in obj.bound_box]
    center = sum(bbox, Vector()) / 8.0
    mins = Vector((min(v.x for v in bbox), min(v.y for v in bbox), min(v.z for v in bbox)))
    maxs = Vector((max(v.x for v in bbox), max(v.y for v in bbox), max(v.z for v in bbox)))
    diagonal = max((maxs - mins).length, 1.0)
    camera_location = center + Vector((0.0, -diagonal * 1.1, diagonal * 0.45))

    bpy.ops.object.camera_add(location=camera_location)
    camera = bpy.context.object
    camera.name = "ClosedLoop_Fog_Volume_Camera"
    direction = center - camera.location
    camera.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()
    camera.data.lens = 28.0
    camera.data.clip_end = max(diagonal * 10.0, 100.0)
    bpy.context.scene.camera = camera

    world = bpy.context.scene.world or bpy.data.worlds.new("ClosedLoop_World")
    bpy.context.scene.world = world
    world.color = (0.04, 0.04, 0.04)


def save_scene(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=str(path))


def main() -> None:
    args = parse_args()
    reset_scene()
    obj = import_ply(args.input, args.name)
    add_vertex_color_material(obj)
    radius = compute_radius(obj, args.radius_scale)
    add_points_geometry_nodes(obj, radius)
    add_camera_and_lighting(obj)
    save_scene(args.output)
    print(f"point_radius: {radius}")
    print(f"output: {args.output}")


if __name__ == "__main__":
    main()
