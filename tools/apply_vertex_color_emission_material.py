"""Apply an emission material driven by the mesh color attribute.

This is for visual inspection only: it preserves the imported vertex colors but
renders them without dependence on scene lights. It is useful when a COLMAP mesh
looks gray/dark in Blender Solid mode or under the closed-loop fog lighting.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import bpy


def parse_args() -> argparse.Namespace:
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--strength", type=float, default=1.2)
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    meshes = [obj for obj in bpy.context.scene.objects if obj.type == "MESH"]
    if not meshes:
        raise SystemExit("No mesh object found")

    obj = max(meshes, key=lambda item: len(item.data.vertices))
    color_attr = "Col"
    if obj.data.color_attributes:
        color_attr = obj.data.color_attributes[0].name

    material = bpy.data.materials.new("COLMAP_Vertex_Color_Emission")
    material.use_nodes = True
    nodes = material.node_tree.nodes
    links = material.node_tree.links
    nodes.clear()

    output = nodes.new(type="ShaderNodeOutputMaterial")
    output.location = (320, 0)
    emission = nodes.new(type="ShaderNodeEmission")
    emission.location = (80, 0)
    emission.inputs["Strength"].default_value = args.strength
    attr = nodes.new(type="ShaderNodeAttribute")
    attr.location = (-180, 0)
    attr.attribute_name = color_attr

    links.new(attr.outputs["Color"], emission.inputs["Color"])
    links.new(emission.outputs["Emission"], output.inputs["Surface"])

    obj.data.materials.clear()
    obj.data.materials.append(material)
    bpy.ops.object.select_all(action="DESELECT")
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj

    # Keep color-management simple for inspection.
    bpy.context.scene.view_settings.view_transform = "Standard"
    bpy.context.scene.view_settings.look = "None"
    bpy.context.scene.view_settings.exposure = 0
    bpy.context.scene.view_settings.gamma = 1

    args.output.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=str(args.output))
    print(f"mesh: {obj.name}")
    print(f"color_attr: {color_attr}")
    print(f"output: {args.output}")


if __name__ == "__main__":
    main()
