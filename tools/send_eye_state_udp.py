"""Send cognitive-load states to the Blender UDP environment controller."""

from __future__ import annotations

import argparse
import json
import socket
import time


STATE_MAP = {
    "low": "low",
    "bajo": "low",
    "medium": "medium",
    "medio": "medium",
    "high": "high",
    "alto": "high",
}


def send_state(sock: socket.socket, host: str, port: int, state: str, score: float | None) -> None:
    payload = {
        "state": state,
        "score": score,
        "timestamp": time.time(),
        "source": "send_eye_state_udp",
    }
    raw = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    sock.sendto(raw, (host, port))
    print(f"sent {payload}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Send UDP eye-state messages to Blender.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5055)
    parser.add_argument("--state", choices=sorted(STATE_MAP), default="low")
    parser.add_argument("--score", type=float, default=None)
    parser.add_argument(
        "--sequence",
        action="store_true",
        help="Cycle low -> medium -> high -> low for trigger testing.",
    )
    parser.add_argument("--interval", type=float, default=2.0)
    parser.add_argument("--cycles", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    if not args.sequence:
        send_state(sock, args.host, args.port, STATE_MAP[args.state], args.score)
        return

    sequence = [("low", 0.15), ("medium", 0.55), ("high", 0.9), ("low", 0.1)]
    for _ in range(args.cycles):
        for state, score in sequence:
            send_state(sock, args.host, args.port, state, score)
            time.sleep(args.interval)


if __name__ == "__main__":
    main()
