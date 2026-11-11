"""Blender UDP listener for closed-loop environment control.

Run this inside Blender with:
    blender path/to/scene.blend --python tools/blender_udp_environment_controller.py

The script listens on UDP 127.0.0.1:5055 and accepts JSON or plain-text states:
    {"state": "low"}
    {"state": "medium"}
    {"state": "high"}

Plain text aliases are also accepted: low, medium, high, 0, 1, 2.

The controller is intentionally conservative: it never edits rock mesh
materials, vertex colors, geometry, or modifiers. By default it changes only
scene lighting/exposure. Volumetric fog can be enabled with ENABLE_FOG_VOLUME
after the material/color pipeline is confirmed in Blender.
"""

from __future__ import annotations

import json
import socket
import time

import bpy
from mathutils import Vector


HOST = "127.0.0.1"
PORT = 5055
NAMESPACE_KEY = "emotion_recognition_udp_environment_controller"
TIMER_TOKEN_KEY = f"{NAMESPACE_KEY}_timer_token"

ENABLE_WORLD_COLOR = False
ENABLE_FOG_VOLUME = False

STATE_CONFIG = {
    "low": {
        "label": "Bajo Estres",
        "light_energy": 700.0,
        "world_color": (0.78, 0.82, 0.88),
        "volume_density": 0.0,
        "exposure": 0.0,
    },
    "medium": {
        "label": "Carga Media",
        "light_energy": 430.0,
        "world_color": (0.52, 0.55, 0.58),
        "volume_density": 0.006,
        "exposure": -0.2,
    },
    "high": {
        "label": "Alta Sobrecarga",
        "light_energy": 180.0,
        "world_color": (0.25, 0.27, 0.30),
        "volume_density": 0.025,
        "exposure": -0.55,
    },
}

ALIASES = {
    "0": "low",
    "1": "medium",
    "2": "high",
    "bajo": "low",
    "bajo estres": "low",
    "low": "low",
    "low_stress": "low",
    "medio": "medium",
    "media": "medium",
    "carga media": "medium",
    "medium": "medium",
    "medium_load": "medium",
    "alto": "high",
    "alta": "high",
    "alta sobrecarga": "high",
    "high": "high",
    "high_overload": "high",
}


def _find_scene_bounds() -> tuple[tuple[float, float, float], float]:
    mesh_objects = [
        obj
        for obj in bpy.context.scene.objects
        if obj.type == "MESH" and not obj.name.startswith("ClosedLoop_Fog_Volume")
    ]
    if not mesh_objects:
        return (0.0, 0.0, 0.0), 30.0

    xs: list[float] = []
    ys: list[float] = []
    zs: list[float] = []
    for obj in mesh_objects:
        for corner in obj.bound_box:
            world = obj.matrix_world @ Vector(corner)
            xs.append(float(world.x))
            ys.append(float(world.y))
            zs.append(float(world.z))

    center = (
        (min(xs) + max(xs)) / 2.0,
        (min(ys) + max(ys)) / 2.0,
        (min(zs) + max(zs)) / 2.0,
    )
    span = max(max(xs) - min(xs), max(ys) - min(ys), max(zs) - min(zs), 1.0)
    return center, span * 2.5


def _ensure_light() -> bpy.types.Object:
    light = bpy.data.objects.get("ClosedLoop_Key_Light")
    if light is None:
        data = bpy.data.lights.new("ClosedLoop_Key_Light", type="AREA")
        light = bpy.data.objects.new("ClosedLoop_Key_Light", data)
        bpy.context.collection.objects.link(light)
        light.location = (0.0, -4.0, 5.0)
        data.size = 5.0
    return light


def _ensure_fog_volume() -> bpy.types.Object | None:
    volume = bpy.data.objects.get("ClosedLoop_Fog_Volume")
    if volume is None:
        center, size = _find_scene_bounds()
        bpy.ops.mesh.primitive_cube_add(size=size, location=center)
        volume = bpy.context.object
        volume.name = "ClosedLoop_Fog_Volume"
        volume.display_type = "WIRE"
        volume.show_wire = True
    volume.hide_viewport = not ENABLE_FOG_VOLUME
    volume.hide_render = not ENABLE_FOG_VOLUME

    material = bpy.data.materials.get("ClosedLoop_Fog_Material")
    if material is None:
        material = bpy.data.materials.new("ClosedLoop_Fog_Material")
        material.use_nodes = True
        nodes = material.node_tree.nodes
        nodes.clear()
        output = nodes.new("ShaderNodeOutputMaterial")
        volume_node = nodes.new("ShaderNodeVolumePrincipled")
        volume_node.name = "ClosedLoop_Principled_Volume"
        volume_node.inputs["Color"].default_value = (0.72, 0.76, 0.78, 1.0)
        volume_node.inputs["Density"].default_value = 0.0
        material.node_tree.links.new(volume_node.outputs["Volume"], output.inputs["Volume"])

    volume.data.materials.clear()
    volume.data.materials.append(material)
    return volume


def _set_fog_density(density: float) -> None:
    volume = bpy.data.objects.get("ClosedLoop_Fog_Volume")
    if volume is not None:
        volume.hide_viewport = not ENABLE_FOG_VOLUME
        volume.hide_render = not ENABLE_FOG_VOLUME

    material = bpy.data.materials.get("ClosedLoop_Fog_Material")
    if material is None or not material.use_nodes:
        return
    node = material.node_tree.nodes.get("ClosedLoop_Principled_Volume")
    if node is not None and "Density" in node.inputs:
        node.inputs["Density"].default_value = float(density if ENABLE_FOG_VOLUME else 0.0)


def _parse_state(payload: bytes) -> str | None:
    text = payload.decode("utf-8", errors="ignore").strip()
    if not text:
        return None

    try:
        data = json.loads(text)
        if isinstance(data, dict):
            text = str(data.get("state", data.get("level", ""))).strip()
    except json.JSONDecodeError:
        pass

    key = text.lower().replace("-", "_")
    return ALIASES.get(key)


def apply_state(state: str) -> None:
    config = STATE_CONFIG[state]
    scene = bpy.context.scene

    if ENABLE_WORLD_COLOR:
        if scene.world is None:
            scene.world = bpy.data.worlds.new("ClosedLoop_World")
        scene.world.color = config["world_color"]
    scene.view_settings.exposure = config["exposure"]

    light = _ensure_light()
    light.data.energy = config["light_energy"]
    if ENABLE_FOG_VOLUME:
        _ensure_fog_volume()
    _set_fog_density(config["volume_density"])

    scene["closed_loop_state"] = state
    scene["closed_loop_label"] = config["label"]
    scene["closed_loop_last_update"] = time.time()
    print(
        "[ClosedLoop] "
        f"{config['label']} -> light={config['light_energy']} "
        f"exposure={config['exposure']} fog={'on' if ENABLE_FOG_VOLUME else 'off'}"
    )


def _poll_udp() -> float:
    state = bpy.app.driver_namespace.get(NAMESPACE_KEY)
    if not state:
        return None
    if state.get("timer_token") != bpy.app.driver_namespace.get(TIMER_TOKEN_KEY):
        return None

    sock = state["socket"]
    while True:
        try:
            payload, address = sock.recvfrom(4096)
        except BlockingIOError:
            break

        parsed = _parse_state(payload)
        if parsed is None:
            print(f"[ClosedLoop] Ignored UDP payload from {address}: {payload!r}")
            continue
        apply_state(parsed)

    return 0.1


def start_udp_listener(host: str = HOST, port: int = PORT) -> None:
    previous = bpy.app.driver_namespace.get(NAMESPACE_KEY)
    if previous and previous.get("socket"):
        try:
            previous["socket"].close()
        except OSError:
            pass

    token = time.time()

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.bind((host, port))
    except OSError as exc:
        sock.close()
        print(f"[ClosedLoop] UDP listener not started on {host}:{port}: {exc}")
        print("[ClosedLoop] Close other Blender listeners or change PORT in this script.")
        return
    sock.setblocking(False)

    bpy.app.driver_namespace[TIMER_TOKEN_KEY] = token
    bpy.app.driver_namespace[NAMESPACE_KEY] = {
        "socket": sock,
        "host": host,
        "port": port,
        "timer_token": token,
    }
    _ensure_light()
    _set_fog_density(0.0)
    apply_state("low")
    bpy.app.timers.register(_poll_udp, persistent=True)
    print(f"[ClosedLoop] UDP listener active on {host}:{port}")


start_udp_listener()
