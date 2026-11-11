"""Measure UDP round-trip and Blender state-application latency."""

from __future__ import annotations

import argparse
import csv
import json
import socket
import statistics
import time
import uuid
from pathlib import Path


STATES = ("low", "medium", "high")


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return float("nan")
    index = (len(ordered) - 1) * q
    lower = int(index)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = index - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def summarise(values: list[float]) -> dict[str, float | int]:
    if not values:
        return {"count": 0}
    return {
        "count": len(values),
        "mean_ms": statistics.fmean(values),
        "median_ms": statistics.median(values),
        "std_ms": statistics.stdev(values) if len(values) > 1 else 0.0,
        "min_ms": min(values),
        "p95_ms": percentile(values, 0.95),
        "max_ms": max(values),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5055)
    parser.add_argument("--messages", type=int, default=30)
    parser.add_argument("--interval", type=float, default=0.03)
    parser.add_argument("--timeout", type=float, default=2.0)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/closed_loop/latency_run"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("127.0.0.1", 0))
    sock.settimeout(args.timeout)

    for index in range(args.messages):
        state = STATES[index % len(STATES)]
        message_id = f"{index:04d}-{uuid.uuid4().hex[:8]}"
        sent_perf_ns = time.perf_counter_ns()
        payload = {
            "message_id": message_id,
            "state": state,
            "score": {"low": 0.15, "medium": 0.55, "high": 0.9}[state],
            "timestamp": time.time(),
            "sent_perf_ns": sent_perf_ns,
            "source": "udp_latency_probe",
        }
        sock.sendto(
            json.dumps(payload, separators=(",", ":")).encode("utf-8"),
            (args.host, args.port),
        )

        row = {
            "index": index,
            "message_id": message_id,
            "state": state,
            "success": False,
            "round_trip_ms": None,
            "send_to_receive_ms": None,
            "apply_ms": None,
            "ack_return_ms": None,
            "difficulty": None,
            "difficulty_level": None,
        }
        try:
            raw, _ = sock.recvfrom(4096)
            client_received_perf_ns = time.perf_counter_ns()
            ack = json.loads(raw.decode("utf-8"))
            if ack.get("message_id") != message_id:
                raise ValueError(
                    f"ACK message mismatch: expected {message_id}, got {ack.get('message_id')}"
                )
            receiver_received_ns = int(ack["received_perf_ns"])
            applied_ns = int(ack["applied_perf_ns"])
            ack_ns = int(ack["ack_perf_ns"])
            row.update(
                {
                    "success": True,
                    "round_trip_ms": (client_received_perf_ns - sent_perf_ns) / 1_000_000.0,
                    "send_to_receive_ms": (receiver_received_ns - sent_perf_ns) / 1_000_000.0,
                    "apply_ms": (applied_ns - receiver_received_ns) / 1_000_000.0,
                    "ack_return_ms": (client_received_perf_ns - ack_ns) / 1_000_000.0,
                    "difficulty": ack.get("difficulty"),
                    "difficulty_level": ack.get("difficulty_level"),
                }
            )
        except (
            socket.timeout,
            ConnectionResetError,
            OSError,
            ValueError,
            KeyError,
            json.JSONDecodeError,
        ) as exc:
            row["error"] = str(exc)
        rows.append(row)
        time.sleep(args.interval)

    sock.close()
    csv_path = args.output_dir / "latency_messages.csv"
    fieldnames = sorted({key for row in rows for key in row})
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    successful = [row for row in rows if row["success"]]
    summary = {
        "host": args.host,
        "port": args.port,
        "messages_sent": len(rows),
        "messages_acknowledged": len(successful),
        "packet_loss_percent": 100.0 * (len(rows) - len(successful)) / max(1, len(rows)),
        "round_trip": summarise([float(row["round_trip_ms"]) for row in successful]),
        "send_to_receive": summarise(
            [float(row["send_to_receive_ms"]) for row in successful]
        ),
        "apply": summarise([float(row["apply_ms"]) for row in successful]),
        "ack_return": summarise([float(row["ack_return_ms"]) for row in successful]),
        "difficulty_sequence": [
            {
                "state": row["state"],
                "difficulty": row["difficulty"],
                "difficulty_level": row["difficulty_level"],
            }
            for row in successful[:6]
        ],
        "note": (
            "Localhost integration latency measured with perf_counter_ns across "
            "the Python sender and Blender receiver processes."
        ),
    }
    (args.output_dir / "latency_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
