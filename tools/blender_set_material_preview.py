"""Set Blender 3D viewports to Material Preview and select the main mesh."""

from __future__ import annotations

import bpy


def select_largest_mesh() -> None:
    meshes = [obj for obj in bpy.context.scene.objects if obj.type == "MESH"]
    if not meshes:
        return
    largest = max(meshes, key=lambda obj: len(obj.data.vertices))
    bpy.ops.object.select_all(action="DESELECT")
    largest.select_set(True)
    bpy.context.view_layer.objects.active = largest


def set_material_preview() -> None:
    screen = bpy.context.screen
    if screen is None:
        return
    for area in screen.areas:
        if area.type != "VIEW_3D":
            continue
        for space in area.spaces:
            if space.type == "VIEW_3D":
                space.shading.type = "MATERIAL"
                space.shading.use_scene_world_render = False
                space.shading.use_scene_lights_render = False


select_largest_mesh()
set_material_preview()
