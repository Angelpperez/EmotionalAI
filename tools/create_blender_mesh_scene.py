"""Create a Blender inspection scene from a mesh PLY.

This is intended for COLMAP/OpenMVS mesh outputs, not native 3DGS PLY files.
It preserves vertex color attributes when Blender imports them.
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
    parser.add_argument("--input", required=True, type=Path, help="Input mesh PLY")
    parser.add_argument("--output", required=True, type=Path, help="Output .blend path")
    parser.add_argument("--name", default="colmap_dense_mesh", help="Mesh object name")
    return parser.parse_args(argv)


def reset_scene() -> None:
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()


def import_mesh(path: Path, name: str) -> bpy.types.Object:
    if hasattr(bpy.ops.wm, "ply_import"):
        bpy.ops.wm.ply_import(filepath=str(path))
    else:
        bpy.ops.import_mesh.ply(filepath=str(path))
    obj = bpy.context.object
    obj.name = name
    obj.data.name = f"{name}_mesh"
    return obj


def add_vertex_color_material(obj: bpy.types.Object) -> None:
    material = bpy.data.materials.new("COLMAP_Vertex_Color")
    material.use_nodes = True
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
    obj.data.materials.append(material)


def add_camera_and_lighting(obj: bpy.types.Object) -> None:
    bbox = [obj.matrix_world @ Vector(corner) for corner in obj.bound_box]
    center = sum(bbox, Vector()) / 8.0
    mins = Vector((min(v.x for v in bbox), min(v.y for v in bbox), min(v.z for v in bbox)))
    maxs = Vector((max(v.x for v in bbox), max(v.y for v in bbox), max(v.z for v in bbox)))
    diagonal = max((maxs - mins).length, 1.0)

    bpy.ops.object.light_add(type="AREA", location=center + Vector((0.0, -diagonal * 0.7, diagonal * 0.9)))
    light = bpy.context.object
    light.name = "ClosedLoop_Key_Light"
    light.data.energy = 700.0
    light.data.size = max(diagonal * 0.35, 1.0)

    camera_location = center + Vector((0.0, -diagonal * 1.2, diagonal * 0.55))
    bpy.ops.object.camera_add(location=camera_location)
    camera = bpy.context.object
    camera.name = "ClosedLoop_Mesh_Camera"
    direction = center - camera.location
    camera.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()
    camera.data.lens = 28.0
    camera.data.clip_end = max(diagonal * 10.0, 100.0)
    bpy.context.scene.camera = camera

    world = bpy.context.scene.world or bpy.data.worlds.new("ClosedLoop_World")
    bpy.context.scene.world = world
    world.color = (0.035, 0.035, 0.035)


def main() -> None:
    args = parse_args()
    reset_scene()
    obj = import_mesh(args.input, args.name)
    add_vertex_color_material(obj)
    add_camera_and_lighting(obj)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=str(args.output))
    print(f"vertices: {len(obj.data.vertices)}")
    print(f"polygons: {len(obj.data.polygons)}")
    print(f"output: {args.output}")


if __name__ == "__main__":
    main()
