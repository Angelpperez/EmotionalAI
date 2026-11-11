"""Probe OpenCV camera indices without modifying camera settings."""

from __future__ import annotations

import argparse
import json

import cv2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-index", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sources = []
    for index in range(args.max_index + 1):
        cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        opened = cap.isOpened()
        row = {"index": index, "opened": opened}
        if opened:
            ok, frame = cap.read()
            row.update(
                {
                    "read_ok": bool(ok),
                    "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                    "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                    "fps": float(cap.get(cv2.CAP_PROP_FPS)),
                    "frame_shape": list(frame.shape) if ok else None,
                }
            )
        cap.release()
        sources.append(row)
    print(json.dumps(sources, indent=2))


if __name__ == "__main__":
    main()
