"""Isolate rock components from the cleaned Basler blaze ToF point cloud.

This is the second, additive cleaning stage. It follows the local
``machine_learning`` repository's RANSAC-plane/component-segmentation method,
adapted to the Basler coordinate units (millimetres) and organized range grid:

1. Fit the dominant background plane with RANSAC.
2. Keep points located at least ``foreground_distance_mm`` in front of it.
3. Label 8-connected components in image coordinates.
4. Reject small components and components touching the image border.

The input cloud is never overwritten. The output includes one combined PLY,
one PLY per rock, a diagnostic preview and quantitative metrics.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from plyfile import PlyData, PlyElement
from scipy import ndimage


WIDTH = 640
HEIGHT = 480


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--ransac-distance-mm", type=float, default=8.0)
    parser.add_argument("--ransac-iterations", type=int, default=1500)
    parser.add_argument("--ransac-sample-points", type=int, default=60_000)
    parser.add_argument("--foreground-distance-mm", type=float, default=15.0)
    parser.add_argument("--min-component-points", type=int, default=500)
    parser.add_argument("--border-margin-pixels", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def fit_plane_ransac(
    xyz: np.ndarray,
    distance_threshold: float,
    iterations: int,
    sample_points: int,
    seed: int,
) -> tuple[np.ndarray, float, np.ndarray]:
    rng = np.random.default_rng(seed)
    sample_indices = rng.choice(
        len(xyz), min(sample_points, len(xyz)), replace=False
    )
    sample = xyz[sample_indices]
    best_score = -1
    best_normal: np.ndarray | None = None
    best_offset = 0.0

    for _ in range(iterations):
        points = sample[rng.choice(len(sample), 3, replace=False)]
        normal = np.cross(points[1] - points[0], points[2] - points[0])
        norm = np.linalg.norm(normal)
        if norm < 1e-9:
            continue
        normal /= norm
        offset = -float(normal @ points[0])
        score = int(
            np.count_nonzero(
                np.abs(sample @ normal + offset) <= distance_threshold
            )
        )
        if score > best_score:
            best_score = score
            best_normal = normal
            best_offset = offset

    if best_normal is None:
        raise RuntimeError("RANSAC could not estimate a plane")

    preliminary_distances = np.abs(xyz @ best_normal + best_offset)
    preliminary_inliers = preliminary_distances <= distance_threshold
    inlier_xyz = xyz[preliminary_inliers]
    if len(inlier_xyz) > 100_000:
        inlier_xyz = inlier_xyz[
            rng.choice(len(inlier_xyz), 100_000, replace=False)
        ]

    homogeneous = np.column_stack(
        [inlier_xyz, np.ones(len(inlier_xyz), dtype=np.float64)]
    )
    _, _, vh = np.linalg.svd(homogeneous, full_matrices=False)
    plane = vh[-1]
    normal = plane[:3]
    norm = np.linalg.norm(normal)
    normal /= norm
    offset = float(plane[3] / norm)

    # Basler points face the negative Z direction. This orientation makes
    # foreground objects have a negative signed distance from the background.
    if normal[2] > 0:
        normal = -normal
        offset = -offset

    signed_distance = xyz @ normal + offset
    inliers = np.abs(signed_distance) <= distance_threshold
    return normal, offset, inliers


def append_rock_id(
    records: np.ndarray,
    rock_ids: np.ndarray,
) -> np.ndarray:
    output_dtype = np.dtype(records.dtype.descr + [("rock_id", "u1")])
    output = np.empty(len(records), dtype=output_dtype)
    for name in records.dtype.names or ():
        output[name] = records[name]
    output["rock_id"] = rock_ids
    return output


def save_ply(path: Path, records: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    PlyData([PlyElement.describe(records, "vertex")], text=False).write(str(path))


def component_statistics(
    records: np.ndarray,
    rock_id: int,
    label_id: int,
) -> dict[str, object]:
    xyz = np.column_stack([records["x"], records["y"], records["z"]]).astype(
        np.float64
    )
    minimum = xyz.min(axis=0)
    maximum = xyz.max(axis=0)
    extent = maximum - minimum
    return {
        "rock_id": rock_id,
        "source_component_label": label_id,
        "points": int(len(records)),
        "pixel_bbox_uv": [
            int(records["pixel_u"].min()),
            int(records["pixel_v"].min()),
            int(records["pixel_u"].max()),
            int(records["pixel_v"].max()),
        ],
        "centroid_xyz_mm": xyz.mean(axis=0).tolist(),
        "axis_aligned_extent_xyz_mm": extent.tolist(),
        "confidence_mean": float(records["confidence"].mean()),
        "stable_observations_mean": float(
            records["stable_observations"].mean()
        ),
    }


def save_preview(
    path: Path,
    records: np.ndarray,
    signed_distance: np.ndarray,
    selected_labels: list[int],
    labels: np.ndarray,
    combined: np.ndarray,
) -> None:
    intensity = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)
    intensity[records["pixel_v"], records["pixel_u"]] = records["red"]

    residual_image = np.full((HEIGHT, WIDTH), np.nan, dtype=np.float32)
    residual_image[records["pixel_v"], records["pixel_u"]] = signed_distance

    palette = np.array(
        [
            [230, 85, 65],
            [45, 150, 220],
            [70, 180, 90],
            [220, 170, 45],
        ],
        dtype=np.uint8,
    )
    component_image = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
    for index, label_id in enumerate(selected_labels):
        component_image[labels == label_id] = palette[index % len(palette)]

    figure, axes = plt.subplots(2, 2, figsize=(12, 9))
    axes[0, 0].imshow(intensity, cmap="gray", vmin=60, vmax=130)
    axes[0, 0].set_title("Intensidad ToF limpia")
    axes[0, 0].axis("off")

    residual_plot = axes[0, 1].imshow(
        residual_image,
        cmap="coolwarm",
        vmin=-150,
        vmax=30,
    )
    axes[0, 1].set_title("Distancia firmada al plano RANSAC [mm]")
    axes[0, 1].axis("off")
    figure.colorbar(residual_plot, ax=axes[0, 1], shrink=0.75)

    axes[1, 0].imshow(component_image)
    axes[1, 0].set_title(
        f"Componentes de roca conservadas: {len(selected_labels)}"
    )
    axes[1, 0].axis("off")

    for rock_id in sorted(np.unique(combined["rock_id"])):
        rock = combined[combined["rock_id"] == rock_id]
        color = palette[(int(rock_id) - 1) % len(palette)] / 255.0
        axes[1, 1].scatter(
            rock["x"],
            rock["y"],
            s=1.0,
            linewidths=0,
            color=color,
            label=f"Roca {int(rock_id)} ({len(rock):,})",
        )
    axes[1, 1].set_title(f"Nube segmentada: {len(combined):,} puntos")
    axes[1, 1].set_xlabel("X [mm]")
    axes[1, 1].set_ylabel("Y [mm]")
    axes[1, 1].set_aspect("equal")
    axes[1, 1].invert_yaxis()
    axes[1, 1].legend()

    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=220)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    started = time.perf_counter()
    ply = PlyData.read(str(args.input))
    records = ply["vertex"].data
    required = {
        "x",
        "y",
        "z",
        "red",
        "pixel_u",
        "pixel_v",
        "confidence",
        "stable_observations",
    }
    missing = required.difference(records.dtype.names or ())
    if missing:
        raise ValueError(f"Input PLY is missing properties: {sorted(missing)}")

    xyz = np.column_stack([records["x"], records["y"], records["z"]]).astype(
        np.float64
    )
    normal, offset, plane_inliers = fit_plane_ransac(
        xyz,
        args.ransac_distance_mm,
        args.ransac_iterations,
        args.ransac_sample_points,
        args.seed,
    )
    signed_distance = xyz @ normal + offset
    foreground = signed_distance < -args.foreground_distance_mm

    foreground_mask = np.zeros((HEIGHT, WIDTH), dtype=bool)
    foreground_mask[
        records["pixel_v"][foreground], records["pixel_u"][foreground]
    ] = True
    labels, component_count = ndimage.label(
        foreground_mask, structure=np.ones((3, 3), dtype=np.uint8)
    )

    candidates: list[dict[str, object]] = []
    selected: list[tuple[int, float, float]] = []
    margin = args.border_margin_pixels
    for label_id in range(1, component_count + 1):
        v, u = np.where(labels == label_id)
        point_count = int(len(u))
        if point_count == 0:
            continue
        touches_border = bool(
            u.min() <= margin
            or v.min() <= margin
            or u.max() >= WIDTH - 1 - margin
            or v.max() >= HEIGHT - 1 - margin
        )
        accepted = (
            point_count >= args.min_component_points and not touches_border
        )
        row = {
            "label": label_id,
            "points": point_count,
            "pixel_bbox_uv": [
                int(u.min()),
                int(v.min()),
                int(u.max()),
                int(v.max()),
            ],
            "pixel_centroid_uv": [float(u.mean()), float(v.mean())],
            "touches_border": touches_border,
            "accepted": accepted,
        }
        candidates.append(row)
        if accepted:
            selected.append((label_id, float(v.mean()), float(u.mean())))

    selected.sort(key=lambda item: (item[1], item[2]))
    if not selected:
        raise RuntimeError("No rock components passed the selection criteria")

    selected_labels = [item[0] for item in selected]
    record_labels = labels[records["pixel_v"], records["pixel_u"]]
    selected_record_mask = np.isin(record_labels, selected_labels)
    selected_records = records[selected_record_mask]
    selected_record_labels = record_labels[selected_record_mask]

    rock_ids = np.zeros(len(selected_records), dtype=np.uint8)
    label_to_rock_id = {
        label_id: rock_id
        for rock_id, label_id in enumerate(selected_labels, start=1)
    }
    for label_id, rock_id in label_to_rock_id.items():
        rock_ids[selected_record_labels == label_id] = rock_id
    combined = append_rock_id(selected_records, rock_ids)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    combined_path = args.output_dir / "blaze101_tof_rocks_clean_combined.ply"
    save_ply(combined_path, combined)

    rock_metrics = []
    for rock_id in range(1, len(selected_labels) + 1):
        rock = combined[combined["rock_id"] == rock_id]
        rock_path = args.output_dir / f"blaze101_tof_rock_{rock_id:02d}.ply"
        save_ply(rock_path, rock)
        rock_metrics.append(
            {
                **component_statistics(
                    rock,
                    rock_id,
                    selected_labels[rock_id - 1],
                ),
                "output_ply": str(rock_path),
            }
        )

    preview_path = args.output_dir / "rock_segmentation_preview.png"
    save_preview(
        preview_path,
        records,
        signed_distance,
        selected_labels,
        labels,
        combined,
    )

    metrics = {
        "input_ply": str(args.input),
        "input_points": int(len(records)),
        "method": "SOR-cleaned ToF -> RANSAC dominant plane -> organized connected components",
        "plane_model": {
            "normal": normal.tolist(),
            "offset": offset,
            "equation": "normal_x*x + normal_y*y + normal_z*z + offset = 0",
        },
        "ransac_distance_mm": args.ransac_distance_mm,
        "ransac_iterations": args.ransac_iterations,
        "plane_inlier_points": int(plane_inliers.sum()),
        "plane_inlier_percent": float(100.0 * plane_inliers.mean()),
        "foreground_distance_mm": args.foreground_distance_mm,
        "foreground_candidate_points": int(foreground.sum()),
        "component_count": int(component_count),
        "min_component_points": args.min_component_points,
        "border_margin_pixels": args.border_margin_pixels,
        "accepted_components": len(selected_labels),
        "selected_rock_points": int(len(combined)),
        "rejected_non_rock_points": int(len(records) - len(combined)),
        "candidate_components": sorted(
            candidates, key=lambda row: int(row["points"]), reverse=True
        ),
        "rocks": rock_metrics,
        "combined_output_ply": str(combined_path),
        "preview": str(preview_path),
        "elapsed_seconds": time.perf_counter() - started,
        "limitations": [
            "Axis-aligned extents are measured in the Basler camera frame.",
            "Only surfaces visible from this fixed ToF viewpoint are present.",
            "This segmentation is not a 3DGS model and has no RGB texture.",
        ],
    }
    (args.output_dir / "rock_segmentation_metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
