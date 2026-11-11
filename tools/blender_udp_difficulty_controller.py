"""Blender UDP controller for cognitive-load-aware task difficulty.

The controller does not edit rock geometry, materials, vertex colors, lights,
or world settings. It updates scene custom properties and only changes the
visibility of objects explicitly tagged with:

    object["closed_loop_min_difficulty"] = 0, 1, or 2
    object["closed_loop_max_difficulty"] = 0, 1, or 2

Difficulty mapping:
    low cognitive load  -> hard task   (level 2)
    medium load         -> normal task (level 1)
    high overload       -> easy task   (level 0)

Messages may contain ``message_id`` and ``sent_perf_ns``. The controller sends
an acknowledgement to the sender's UDP address so end-to-end latency can be
measured on localhost.
"""

from __future__ import annotations

import csv
import json
import os
import socket
import time
from pathlib import Path

import bpy


HOST = "127.0.0.1"
PORT = 5055
NAMESPACE_KEY = "emotion_recognition_udp_difficulty_controller"
TIMER_TOKEN_KEY = f"{NAMESPACE_KEY}_timer_token"

STATE_CONFIG = {
    "low": {
        "label": "Bajo Estres",
        "difficulty": "hard",
        "difficulty_level": 2,
        "difficulty_normalized": 1.0,
    },
    "medium": {
        "label": "Carga Media",
        "difficulty": "normal",
        "difficulty_level": 1,
        "difficulty_normalized": 0.5,
    },
    "high": {
        "label": "Alta Sobrecarga",
        "difficulty": "easy",
        "difficulty_level": 0,
        "difficulty_normalized": 0.0,
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


def _project_output_path() -> Path:
    try:
        root = Path(__file__).resolve().parents[1]
    except NameError:
        root = Path.cwd()
    return root / "outputs" / "closed_loop" / "difficulty_events.csv"


LOG_PATH = _project_output_path()


def _close_previous_listeners() -> None:
    for key in (
        NAMESPACE_KEY,
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


def _parse_payload(payload: bytes) -> tuple[str | None, dict]:
    text = payload.decode("utf-8", errors="ignore").strip()
    if not text:
        return None, {}

    data = {}
    try:
        decoded = json.loads(text)
        if isinstance(decoded, dict):
            data = decoded
            text = str(decoded.get("state", decoded.get("level", ""))).strip()
    except json.JSONDecodeError:
        pass

    state = ALIASES.get(text.lower().replace("-", "_"))
    return state, data


def _update_tagged_objects(difficulty_level: int) -> int:
    changed = 0
    for obj in bpy.context.scene.objects:
        has_min = "closed_loop_min_difficulty" in obj
        has_max = "closed_loop_max_difficulty" in obj
        if not has_min and not has_max:
            continue

        min_level = int(obj.get("closed_loop_min_difficulty", 0))
        max_level = int(obj.get("closed_loop_max_difficulty", 2))
        visible = min_level <= difficulty_level <= max_level
        hide = not visible
        if obj.hide_viewport != hide or obj.hide_render != hide:
            changed += 1
        obj.hide_viewport = hide
        obj.hide_render = hide
    return changed


def apply_state(state: str) -> tuple[dict, int]:
    config = STATE_CONFIG[state]
    scene = bpy.context.scene
    difficulty_level = int(config["difficulty_level"])
    changed_objects = _update_tagged_objects(difficulty_level)

    scene["closed_loop_state"] = state
    scene["closed_loop_label"] = config["label"]
    scene["closed_loop_difficulty"] = config["difficulty"]
    scene["closed_loop_difficulty_level"] = difficulty_level
    scene["closed_loop_difficulty_normalized"] = float(
        config["difficulty_normalized"]
    )
    scene["closed_loop_last_update"] = time.time()
    scene["closed_loop_changed_tagged_objects"] = changed_objects

    print(
        "[ClosedLoopDifficulty] "
        f"state={state} difficulty={config['difficulty']} "
        f"level={difficulty_level} tagged_objects_changed={changed_objects}"
    )
    return config, changed_objects


def _append_log(row: dict) -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "message_id",
        "state",
        "difficulty",
        "difficulty_level",
        "sender_timestamp",
        "received_wall_ns",
        "received_perf_ns",
        "applied_perf_ns",
        "apply_ms",
        "changed_tagged_objects",
        "source",
    ]
    exists = LOG_PATH.exists()
    with LOG_PATH.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerow({key: row.get(key) for key in fieldnames})


def _process_message(
    sock: socket.socket,
    payload: bytes,
    address: tuple[str, int],
) -> bool:
    received_wall_ns = time.time_ns()
    received_perf_ns = time.perf_counter_ns()
    state, data = _parse_payload(payload)
    if state is None:
        print(f"[ClosedLoopDifficulty] Ignored payload from {address}: {payload!r}")
        return False

    config, changed_objects = apply_state(state)
    applied_perf_ns = time.perf_counter_ns()
    apply_ms = (applied_perf_ns - received_perf_ns) / 1_000_000.0
    row = {
        "message_id": data.get("message_id"),
        "state": state,
        "difficulty": config["difficulty"],
        "difficulty_level": config["difficulty_level"],
        "sender_timestamp": data.get("timestamp"),
        "received_wall_ns": received_wall_ns,
        "received_perf_ns": received_perf_ns,
        "applied_perf_ns": applied_perf_ns,
        "apply_ms": apply_ms,
        "changed_tagged_objects": changed_objects,
        "source": data.get("source"),
    }
    _append_log(row)

    ack = {
        "type": "closed_loop_ack",
        "message_id": data.get("message_id"),
        "state": state,
        "difficulty": config["difficulty"],
        "difficulty_level": config["difficulty_level"],
        "sent_perf_ns": data.get("sent_perf_ns"),
        "received_perf_ns": received_perf_ns,
        "applied_perf_ns": applied_perf_ns,
        "ack_perf_ns": time.perf_counter_ns(),
        "apply_ms": apply_ms,
        "changed_tagged_objects": changed_objects,
        "source": "blender_udp_difficulty_controller",
    }
    try:
        sock.sendto(json.dumps(ack, separators=(",", ":")).encode("utf-8"), address)
    except OSError as exc:
        print(f"[ClosedLoopDifficulty] ACK failed for {address}: {exc}")
    return True


def _poll_udp() -> float | None:
    listener = bpy.app.driver_namespace.get(NAMESPACE_KEY)
    if not listener:
        return None
    if listener.get("timer_token") != bpy.app.driver_namespace.get(TIMER_TOKEN_KEY):
        return None

    sock = listener.get("socket")
    if sock is None:
        return None

    while True:
        try:
            payload, address = sock.recvfrom(4096)
        except BlockingIOError:
            break
        except OSError as exc:
            print(f"[ClosedLoopDifficulty] Socket stopped: {exc}")
            return None
        _process_message(sock, payload, address)
    return 0.02


def start_udp_listener(host: str = HOST, port: int = PORT) -> None:
    _close_previous_listeners()
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.bind((host, port))
    except OSError as exc:
        sock.close()
        print(f"[ClosedLoopDifficulty] Could not listen on {host}:{port}: {exc}")
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
    apply_state("medium")
    bpy.app.timers.register(_poll_udp)
    print(f"[ClosedLoopDifficulty] UDP listener active on {host}:{port}")


def run_synchronous_test(message_count: int, timeout_seconds: float = 30.0) -> None:
    _close_previous_listeners()
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((HOST, PORT))
    sock.settimeout(timeout_seconds)
    processed = 0
    print(
        f"[ClosedLoopDifficulty] Synchronous integration test waiting for "
        f"{message_count} messages on {HOST}:{PORT}"
    )
    try:
        while processed < message_count:
            try:
                payload, address = sock.recvfrom(4096)
            except socket.timeout:
                print(
                    "[ClosedLoopDifficulty] Synchronous test timed out: "
                    f"processed={processed}/{message_count}"
                )
                break
            if _process_message(sock, payload, address):
                processed += 1
    finally:
        sock.close()
    print(f"[ClosedLoopDifficulty] Synchronous test completed: {processed} messages")


test_messages = int(os.environ.get("CLOSED_LOOP_SYNC_TEST_MESSAGES", "0"))
if test_messages > 0:
    run_synchronous_test(test_messages)
else:
    start_udp_listener()
