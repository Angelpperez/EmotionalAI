"""Remove dominant planes from a COLMAP dense point cloud using RANSAC.

This is useful when COLMAP reconstructs the floor/backdrop together with the
target object. The script preserves all original vertex properties
(`x y z nx ny nz red green blue`) and only filters vertices.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from plyfile import PlyData, PlyElement


def fit_plane(points: np.ndarray) -> tuple[np.ndarray, float] | None:
    p0, p1, p2 = points
    normal = np.cross(p1 - p0, p2 - p0)
    norm = np.linalg.norm(normal)
    if norm < 1e-8:
        return None
    normal = normal / norm
    offset = -float(np.dot(normal, p0))
    return normal.astype(np.float32), offset


def ransac_plane(
    xyz: np.ndarray,
    rng: np.random.Generator,
    sample_size: int,
    iterations: int,
    threshold: float,
) -> tuple[np.ndarray, float, int]:
    if len(xyz) < 3:
        raise ValueError("Need at least 3 points for plane fitting")

    sample_indices = rng.choice(len(xyz), size=min(sample_size, len(xyz)), replace=False)
    sample = xyz[sample_indices]

    best_normal: np.ndarray | None = None
    best_offset = 0.0
    best_count = -1

    for _ in range(iterations):
        tri = sample[rng.choice(len(sample), size=3, replace=False)]
        plane = fit_plane(tri)
        if plane is None:
            continue
        normal, offset = plane
        distances = np.abs(sample @ normal + offset)
        count = int(np.count_nonzero(distances < threshold))
        if count > best_count:
            best_normal = normal
            best_offset = offset
            best_count = count

    if best_normal is None:
        raise RuntimeError("RANSAC did not find a valid plane")
    return best_normal, best_offset, best_count


def remove_planes(
    vertices: np.ndarray,
    planes: int,
    threshold: float,
    sample_size: int,
    iterations: int,
    seed: int,
) -> tuple[np.ndarray, list[dict[str, float]]]:
    rng = np.random.default_rng(seed)
    mask = np.ones(len(vertices), dtype=bool)
    metrics: list[dict[str, float]] = []

    for plane_idx in range(planes):
        xyz = np.column_stack((vertices["x"][mask], vertices["y"][mask], vertices["z"][mask])).astype(np.float32)
        if len(xyz) < 3:
            break
        normal, offset, sample_inliers = ransac_plane(xyz, rng, sample_size, iterations, threshold)
        distances = np.abs(xyz @ normal + offset)
        inliers = distances < threshold

        active_indices = np.flatnonzero(mask)
        mask[active_indices[inliers]] = False

        metrics.append(
            {
                "plane": float(plane_idx + 1),
                "normal_x": float(normal[0]),
                "normal_y": float(normal[1]),
                "normal_z": float(normal[2]),
                "offset": float(offset),
                "sample_inliers": float(sample_inliers),
                "removed_points": float(np.count_nonzero(inliers)),
                "remaining_points": float(np.count_nonzero(mask)),
            }
        )

    return mask, metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--planes", type=int, default=1)
    parser.add_argument("--threshold", type=float, default=0.08)
    parser.add_argument("--sample-size", type=int, default=100000)
    parser.add_argument("--iterations", type=int, default=1200)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    ply = PlyData.read(str(args.input))
    vertices = ply["vertex"].data
    for field in ("x", "y", "z"):
        if field not in vertices.dtype.names:
            raise ValueError(f"Missing field {field}")

    mask, metrics = remove_planes(
        vertices,
        planes=args.planes,
        threshold=args.threshold,
        sample_size=args.sample_size,
        iterations=args.iterations,
        seed=args.seed,
    )

    filtered = vertices[mask]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    PlyData([PlyElement.describe(filtered, "vertex")], text=False).write(str(args.output))

    print(f"input_points: {len(vertices)}")
    print(f"output_points: {len(filtered)}")
    print(f"removed_points: {len(vertices) - len(filtered)}")
    print(f"threshold: {args.threshold}")
    for metric in metrics:
        print(metric)
    print(f"output: {args.output}")


if __name__ == "__main__":
    main()
