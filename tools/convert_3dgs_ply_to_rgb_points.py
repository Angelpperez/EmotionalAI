"""Convert a 3D Gaussian Splatting PLY to a standard RGB point PLY.

The gsplat/exported PLY stores color as spherical-harmonics coefficients
(`f_dc_0..2`, `f_rest_*`) plus opacity, scale and rotation. Blender can import
the file, but it does not natively rasterize gaussian splats, so the result is
usually a black vertex cloud. This utility extracts the gaussian centers and
converts the DC color term to RGB for reproducible visual inspection.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from plyfile import PlyData, PlyElement


SH_C0 = 0.28209479177387814


def sigmoid(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(values, -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def require_fields(names: set[str], required: tuple[str, ...]) -> None:
    missing = [name for name in required if name not in names]
    if missing:
        raise ValueError(f"Missing required PLY fields: {', '.join(missing)}")


def convert(input_path: Path, output_path: Path, min_alpha: float, max_points: int | None) -> dict[str, float]:
    ply = PlyData.read(str(input_path))
    vertices = ply["vertex"].data
    names = set(vertices.dtype.names or [])

    require_fields(names, ("x", "y", "z", "f_dc_0", "f_dc_1", "f_dc_2"))

    xyz = np.column_stack((vertices["x"], vertices["y"], vertices["z"])).astype(np.float32)
    dc = np.column_stack((vertices["f_dc_0"], vertices["f_dc_1"], vertices["f_dc_2"])).astype(np.float32)
    rgb_float = np.clip((dc * SH_C0) + 0.5, 0.0, 1.0)
    rgb = np.round(rgb_float * 255.0).astype(np.uint8)

    if "opacity" in names:
        alpha_float = sigmoid(vertices["opacity"].astype(np.float32))
    else:
        alpha_float = np.ones(len(xyz), dtype=np.float32)

    mask = alpha_float >= float(min_alpha)
    if max_points is not None and mask.sum() > max_points:
        kept_indices = np.flatnonzero(mask)
        order = np.argsort(alpha_float[kept_indices])[::-1][:max_points]
        selected = kept_indices[order]
        mask = np.zeros(len(xyz), dtype=bool)
        mask[selected] = True

    xyz = xyz[mask]
    rgb = rgb[mask]
    alpha = np.round(alpha_float[mask] * 255.0).astype(np.uint8)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output = np.empty(
        len(xyz),
        dtype=[
            ("x", "f4"),
            ("y", "f4"),
            ("z", "f4"),
            ("red", "u1"),
            ("green", "u1"),
            ("blue", "u1"),
            ("alpha", "u1"),
        ],
    )
    output["x"] = xyz[:, 0]
    output["y"] = xyz[:, 1]
    output["z"] = xyz[:, 2]
    output["red"] = rgb[:, 0]
    output["green"] = rgb[:, 1]
    output["blue"] = rgb[:, 2]
    output["alpha"] = alpha

    PlyData([PlyElement.describe(output, "vertex")], text=False).write(str(output_path))

    return {
        "input_points": float(len(vertices)),
        "output_points": float(len(output)),
        "min_alpha": float(min_alpha),
        "alpha_mean": float(alpha_float.mean()),
        "rgb_mean_r": float(rgb_float[:, 0].mean()),
        "rgb_mean_g": float(rgb_float[:, 1].mean()),
        "rgb_mean_b": float(rgb_float[:, 2].mean()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Input 3DGS PLY with f_dc_* fields")
    parser.add_argument("output", type=Path, help="Output standard RGB point PLY")
    parser.add_argument("--min-alpha", type=float, default=0.01, help="Drop very transparent gaussians")
    parser.add_argument("--max-points", type=int, default=None, help="Keep only the highest-alpha points")
    args = parser.parse_args()

    metrics = convert(args.input, args.output, args.min_alpha, args.max_points)
    for key, value in metrics.items():
        print(f"{key}: {value}")
    print(f"output: {args.output}")


if __name__ == "__main__":
    main()
