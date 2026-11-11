"""Minimal Blender diagnostic script.

Run this from Blender's Scripting tab. It does not open sockets, does not
register timers, and does not modify mesh/material data.
"""

import bpy


print("[ClosedLoopDiagnostic] Script started")
print(f"[ClosedLoopDiagnostic] Blender version: {bpy.app.version_string}")
print(f"[ClosedLoopDiagnostic] Scene: {bpy.context.scene.name}")
print(f"[ClosedLoopDiagnostic] Objects: {len(bpy.context.scene.objects)}")
bpy.context.scene["closed_loop_diagnostic_ok"] = True
print("[ClosedLoopDiagnostic] Script finished OK")
