"""Minimal Blender UDP light controller for closed-loop tests.

Run this inside Blender from the Scripting tab.

This script is deliberately small and conservative:
    - It does not touch rock mesh geometry.
    - It does not touch rock materials or vertex colors.
    - It does not create fog volumes or cameras.
    - It only adjusts one area light and scene exposure.

UDP examples accepted on 127.0.0.1:5055:
    low, medium, high
    {"state": "high"}
"""

from __future__ import annotations

import json
import socket
import time

import bpy


HOST = "127.0.0.1"
PORT = 5055
NAMESPACE_KEY = "emotion_recognition_udp_light_controller_minimal"
TIMER_TOKEN_KEY = f"{NAMESPACE_KEY}_timer_token"

STATE_CONFIG = {
    "low": {
        "label": "Bajo Estres",
        "light_energy": 700.0,
        "exposure": 0.0,
    },
    "medium": {
        "label": "Carga Media",
        "light_energy": 430.0,
        "exposure": -0.2,
    },
    "high": {
        "label": "Alta Sobrecarga",
        "light_energy": 180.0,
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
    "medio": "medium",
    "media": "medium",
    "carga media": "medium",
    "medium": "medium",
    "alto": "high",
    "alta": "high",
    "alta sobrecarga": "high",
    "high": "high",
}


def _close_previous_listener() -> None:
    for key in (
        NAMESPACE_KEY,
        "emotion_recognition_udp_environment_controller",
    ):
        previous = bpy.app.driver_namespace.get(key)
        if previous and previous.get("socket"):
            try:
                previous["socket"].close()
            except OSError:
                pass
            previous["socket"] = None


def _ensure_light() -> bpy.types.Object:
    light = bpy.data.objects.get("ClosedLoop_Key_Light")
    if light is not None and light.type == "LIGHT":
        return light

    data = bpy.data.lights.new("ClosedLoop_Key_Light", type="AREA")
    light = bpy.data.objects.new("ClosedLoop_Key_Light", data)
    bpy.context.collection.objects.link(light)
    light.location = (0.0, -4.0, 5.0)
    data.size = 5.0
    return light


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

    return ALIASES.get(text.lower().replace("-", "_"))


def apply_state(state: str) -> None:
    config = STATE_CONFIG[state]
    light = _ensure_light()
    light.data.energy = config["light_energy"]
    bpy.context.scene.view_settings.exposure = config["exposure"]
    bpy.context.scene["closed_loop_state"] = state
    bpy.context.scene["closed_loop_label"] = config["label"]
    bpy.context.scene["closed_loop_last_update"] = time.time()
    print(
        "[ClosedLoopMinimal] "
        f"{config['label']} -> light={config['light_energy']} "
        f"exposure={config['exposure']}"
    )


def _poll_udp() -> float | None:
    state = bpy.app.driver_namespace.get(NAMESPACE_KEY)
    if not state:
        return None
    if state.get("timer_token") != bpy.app.driver_namespace.get(TIMER_TOKEN_KEY):
        return None

    sock = state.get("socket")
    if sock is None:
        return None

    while True:
        try:
            payload, address = sock.recvfrom(4096)
        except BlockingIOError:
            break
        except OSError as exc:
            print(f"[ClosedLoopMinimal] Socket stopped: {exc}")
            return None

        parsed = _parse_state(payload)
        if parsed is None:
            print(f"[ClosedLoopMinimal] Ignored UDP payload from {address}: {payload!r}")
            continue
        apply_state(parsed)

    return 0.15


def start_udp_listener(host: str = HOST, port: int = PORT) -> None:
    _close_previous_listener()

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.bind((host, port))
    except OSError as exc:
        sock.close()
        print(f"[ClosedLoopMinimal] Could not listen on {host}:{port}: {exc}")
        print("[ClosedLoopMinimal] Close other Blender listeners or change PORT.")
        return

    sock.setblocking(False)
    token = time.time()
    bpy.app.driver_namespace[TIMER_TOKEN_KEY] = token
    bpy.app.driver_namespace[NAMESPACE_KEY] = {
        "socket": sock,
        "host": host,
        "port": port,
        "timer_token": token,
    }

    apply_state("low")
    bpy.app.timers.register(_poll_udp)
    print(f"[ClosedLoopMinimal] UDP listener active on {host}:{port}")


start_udp_listener()
