"""Blender UDP difficulty controller for the formal closed-loop experiment."""

from __future__ import annotations

import csv
import json
import os
import socket
import time
from pathlib import Path

import bpy


HOST = "127.0.0.1"
PORT = int(os.environ.get("CLOSED_LOOP_EXPERIMENT_PORT", "5055"))
NAMESPACE = "emotion_recognition_closed_loop_experiment"
TOKEN_KEY = f"{NAMESPACE}_token"

STATE_CONFIG = {
    "low": ("hard", 2, "BAJO -> DIFICIL"),
    "medium": ("normal", 1, "MEDIO -> NORMAL"),
    "high": ("easy", 0, "ALTO -> FACIL"),
}

ALIASES = {
    "0": "low",
    "1": "medium",
    "2": "high",
    "low": "low",
    "bajo": "low",
    "medium": "medium",
    "medio": "medium",
    "high": "high",
    "alto": "high",
}


def log_path() -> Path:
    configured = os.environ.get("CLOSED_LOOP_EXPERIMENT_LOG")
    if configured:
        return Path(configured)
    return Path.cwd() / "outputs" / "closed_loop" / "difficulty_events_e2e.csv"


def close_previous() -> None:
    for key in (
        NAMESPACE,
        "emotion_recognition_udp_difficulty_controller",
        "emotion_recognition_udp_environment_controller",
        "emotion_recognition_udp_light_controller_minimal",
    ):
        previous = bpy.app.driver_namespace.get(key)
        if previous and previous.get("socket"):
            try:
                previous["socket"].close()
            except OSError:
                pass
            previous["socket"] = None


def parse_payload(payload: bytes) -> tuple[str | None, dict]:
    text = payload.decode("utf-8", errors="ignore").strip()
    if not text:
        return None, {}
    data = {}
    try:
        decoded = json.loads(text)
        if isinstance(decoded, dict):
            data = decoded
            text = str(decoded.get("state", "")).strip()
    except json.JSONDecodeError:
        pass
    return ALIASES.get(text.lower()), data


def update_status_text(label: str) -> None:
    status = bpy.data.objects.get("ClosedLoop_Status_Text")
    if status is not None and status.type == "FONT":
        difficulty = label.split("->")[-1].strip()
        status.data.body = f"DIFICULTAD: {difficulty}"


def set_viewport_to_scene_camera() -> None:
    screen = bpy.context.screen
    if screen is None:
        return
    for area in screen.areas:
        if area.type != "VIEW_3D":
            continue
        space = area.spaces.active
        if getattr(space, "region_3d", None) is not None:
            space.region_3d.view_perspective = "CAMERA"


def update_tagged_objects(level: int) -> int:
    changed = 0
    for obj in bpy.context.scene.objects:
        if (
            "closed_loop_min_difficulty" not in obj
            and "closed_loop_max_difficulty" not in obj
        ):
            continue
        minimum = int(obj.get("closed_loop_min_difficulty", 0))
        maximum = int(obj.get("closed_loop_max_difficulty", 2))
        hidden = not (minimum <= level <= maximum)
        if obj.hide_viewport != hidden or obj.hide_render != hidden:
            changed += 1
        obj.hide_viewport = hidden
        obj.hide_render = hidden
    return changed


def apply_state(state: str) -> tuple[str, int, int]:
    difficulty, level, label = STATE_CONFIG[state]
    changed = update_tagged_objects(level)
    update_status_text(label)
    scene = bpy.context.scene
    scene["closed_loop_state"] = state
    scene["closed_loop_difficulty"] = difficulty
    scene["closed_loop_difficulty_level"] = level
    scene["closed_loop_changed_tagged_objects"] = changed
    scene["closed_loop_last_update"] = time.time()
    scene["closed_loop_event_count"] = (
        int(scene.get("closed_loop_event_count", 0)) + 1
    )
    print(
        "[ClosedLoopE2E] "
        f"state={state} difficulty={difficulty} level={level} changed={changed}"
    )
    return difficulty, level, changed


def append_log(row: dict) -> None:
    path = log_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "message_id",
        "state",
        "difficulty",
        "difficulty_level",
        "source",
        "sender_timestamp",
        "received_wall_ns",
        "received_perf_ns",
        "applied_perf_ns",
        "apply_ms",
        "changed_tagged_objects",
    ]
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        if not exists:
            writer.writeheader()
        writer.writerow({field: row.get(field) for field in fields})


def process_message(
    sock: socket.socket,
    payload: bytes,
    address: tuple[str, int],
) -> bool:
    received_wall_ns = time.time_ns()
    received_perf_ns = time.perf_counter_ns()
    state, data = parse_payload(payload)
    if state is None:
        return False
    difficulty, level, changed = apply_state(state)
    applied_perf_ns = time.perf_counter_ns()
    apply_ms = (applied_perf_ns - received_perf_ns) / 1_000_000.0
    append_log(
        {
            "message_id": data.get("message_id"),
            "state": state,
            "difficulty": difficulty,
            "difficulty_level": level,
            "source": data.get("source"),
            "sender_timestamp": data.get("timestamp"),
            "received_wall_ns": received_wall_ns,
            "received_perf_ns": received_perf_ns,
            "applied_perf_ns": applied_perf_ns,
            "apply_ms": apply_ms,
            "changed_tagged_objects": changed,
        }
    )
    ack = {
        "type": "closed_loop_ack",
        "message_id": data.get("message_id"),
        "state": state,
        "difficulty": difficulty,
        "difficulty_level": level,
        "sent_perf_ns": data.get("sent_perf_ns"),
        "received_perf_ns": received_perf_ns,
        "applied_perf_ns": applied_perf_ns,
        "ack_perf_ns": time.perf_counter_ns(),
        "apply_ms": apply_ms,
        "changed_tagged_objects": changed,
        "source": "blender_udp_difficulty_controller_experiment",
    }
    sock.sendto(
        json.dumps(ack, separators=(",", ":")).encode("utf-8"),
        address,
    )
    return True


def poll_udp() -> float | None:
    listener = bpy.app.driver_namespace.get(NAMESPACE)
    if not listener:
        return None
    if listener.get("token") != bpy.app.driver_namespace.get(TOKEN_KEY):
        return None
    sock = listener.get("socket")
    if sock is None:
        return None
    while True:
        try:
            payload, address = sock.recvfrom(8192)
        except BlockingIOError:
            break
        except OSError:
            return None
        try:
            process_message(sock, payload, address)
        except Exception as exc:
            print(f"[ClosedLoopE2E] message error: {exc}")
    return 0.01


def start() -> None:
    close_previous()
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((HOST, PORT))
    sock.setblocking(False)
    token = time.time_ns()
    bpy.app.driver_namespace[TOKEN_KEY] = token
    bpy.app.driver_namespace[NAMESPACE] = {
        "socket": sock,
        "token": token,
    }
    apply_state("medium")
    set_viewport_to_scene_camera()
    bpy.app.timers.register(poll_udp)
    print(
        f"[ClosedLoopE2E] listening on {HOST}:{PORT}; log={log_path()}"
    )


def run_synchronous_test(message_count: int, timeout_seconds: float = 30.0) -> None:
    close_previous()
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((HOST, PORT))
    sock.settimeout(timeout_seconds)
    processed = 0
    apply_state("medium")
    print(
        f"[ClosedLoopE2E] synchronous test waiting for "
        f"{message_count} messages"
    )
    try:
        while processed < message_count:
            try:
                payload, address = sock.recvfrom(8192)
            except socket.timeout:
                break
            if process_message(sock, payload, address):
                processed += 1
    finally:
        sock.close()
    print(
        f"[ClosedLoopE2E] synchronous test completed "
        f"{processed}/{message_count}"
    )


test_messages = int(os.environ.get("CLOSED_LOOP_SYNC_TEST_MESSAGES", "0"))
if test_messages > 0:
    run_synchronous_test(test_messages)
else:
    start()
