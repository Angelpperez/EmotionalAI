"""Filter a point cloud by connected voxel components.

This removes disconnected residual structures after plane removal while
preserving all original PLY vertex fields.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from plyfile import PlyData, PlyElement


NEIGHBORS_26 = [
    (dx, dy, dz)
    for dx in (-1, 0, 1)
    for dy in (-1, 0, 1)
    for dz in (-1, 0, 1)
    if not (dx == 0 and dy == 0 and dz == 0)
]


def connected_voxel_components(xyz: np.ndarray, voxel_size: float) -> tuple[np.ndarray, list[dict[str, float]]]:
    voxels = np.floor(xyz / voxel_size).astype(np.int32)
    unique_voxels, inverse = np.unique(voxels, axis=0, return_inverse=True)
    voxel_counts = np.bincount(inverse)
    voxel_to_index = {tuple(voxel): idx for idx, voxel in enumerate(unique_voxels)}
    seen = np.zeros(len(unique_voxels), dtype=bool)
    voxel_component = np.full(len(unique_voxels), -1, dtype=np.int32)
    summaries: list[dict[str, float]] = []

    component_id = 0
    for start in range(len(unique_voxels)):
        if seen[start]:
            continue

        stack = [start]
        seen[start] = True
        members = []
        while stack:
            current = stack.pop()
            members.append(current)
            x, y, z = unique_voxels[current]
            for dx, dy, dz in NEIGHBORS_26:
                neighbor = voxel_to_index.get((x + dx, y + dy, z + dz))
                if neighbor is not None and not seen[neighbor]:
                    seen[neighbor] = True
                    stack.append(neighbor)

        members_array = np.array(members, dtype=np.int32)
        voxel_component[members_array] = component_id
        point_count = int(voxel_counts[members_array].sum())
        mins = unique_voxels[members_array].min(axis=0) * voxel_size
        maxs = (unique_voxels[members_array].max(axis=0) + 1) * voxel_size
        size = maxs - mins
        summaries.append(
            {
                "component": float(component_id),
                "points": float(point_count),
                "voxels": float(len(members_array)),
                "min_x": float(mins[0]),
                "min_y": float(mins[1]),
                "min_z": float(mins[2]),
                "size_x": float(size[0]),
                "size_y": float(size[1]),
                "size_z": float(size[2]),
            }
        )
        component_id += 1

    point_components = voxel_component[inverse]
    return point_components, summaries


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--voxel-size", type=float, default=0.25)
    parser.add_argument("--keep-components", type=int, default=1)
    args = parser.parse_args()

    ply = PlyData.read(str(args.input))
    vertices = ply["vertex"].data
    xyz = np.column_stack((vertices["x"], vertices["y"], vertices["z"])).astype(np.float32)
    point_components, summaries = connected_voxel_components(xyz, args.voxel_size)
    summaries_sorted = sorted(summaries, key=lambda item: item["points"], reverse=True)
    keep_ids = {int(item["component"]) for item in summaries_sorted[: args.keep_components]}
    mask = np.array([component in keep_ids for component in point_components], dtype=bool)
    filtered = vertices[mask]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    PlyData([PlyElement.describe(filtered, "vertex")], text=False).write(str(args.output))

    print(f"input_points: {len(vertices)}")
    print(f"output_points: {len(filtered)}")
    print(f"removed_points: {len(vertices) - len(filtered)}")
    print(f"voxel_size: {args.voxel_size}")
    for summary in summaries_sorted[:12]:
        print(summary)
    print(f"output: {args.output}")


if __name__ == "__main__":
    main()
