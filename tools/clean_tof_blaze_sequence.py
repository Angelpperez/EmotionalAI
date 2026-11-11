"""Build a cleaned canonical point cloud from a fixed Basler blaze sequence.

The local ``machine_learning`` repository removes point-cloud outliers with
Open3D statistical outlier removal using 20 neighbours and ``std_ratio=2.0``.
This script preserves that method and adds ToF-specific filtering before SOR:

1. Invalid pixels are removed through the PLY ``range_grid``/range image.
2. Low-confidence measurements are rejected.
3. A temporal median selects the stable surface instead of transient
   foreground occluders.
4. Depth-discontinuity pixels are removed to reduce mixed/flying pixels.
5. Statistical outlier removal is applied to the resulting canonical cloud.

Inputs are never modified. The output is a new binary PLY plus masks, metrics
and a diagnostic preview.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from plyfile import PlyData, PlyElement
from scipy import ndimage
from scipy.spatial import cKDTree


WIDTH = 640
HEIGHT = 480


def load_u16(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        array = np.asarray(image)
    if array.shape != (HEIGHT, WIDTH):
        raise ValueError(f"Unexpected image shape for {path}: {array.shape}")
    return array.astype(np.uint16, copy=False)


def save_mask(path: Path, mask: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((mask.astype(np.uint8) * 255), mode="L").save(path)


def depth_edge_mask(
    depth: np.ndarray,
    valid: np.ndarray,
    threshold: int,
    dilation_pixels: int,
) -> np.ndarray:
    edge = np.zeros(depth.shape, dtype=bool)

    horizontal_valid = valid[:, 1:] & valid[:, :-1]
    horizontal_jump = np.abs(
        depth[:, 1:].astype(np.int32) - depth[:, :-1].astype(np.int32)
    )
    horizontal_edge = horizontal_valid & (horizontal_jump > threshold)
    edge[:, 1:] |= horizontal_edge
    edge[:, :-1] |= horizontal_edge

    vertical_valid = valid[1:, :] & valid[:-1, :]
    vertical_jump = np.abs(
        depth[1:, :].astype(np.int32) - depth[:-1, :].astype(np.int32)
    )
    vertical_edge = vertical_valid & (vertical_jump > threshold)
    edge[1:, :] |= vertical_edge
    edge[:-1, :] |= vertical_edge

    if dilation_pixels > 0:
        edge = ndimage.binary_dilation(edge, iterations=dilation_pixels)
    return edge


def statistical_outlier_mask(
    xyz: np.ndarray,
    neighbours: int,
    std_ratio: float,
) -> tuple[np.ndarray, dict[str, float]]:
    if len(xyz) <= neighbours:
        raise ValueError(
            f"Need more than {neighbours} points for statistical filtering"
        )

    tree = cKDTree(xyz)
    distances, _ = tree.query(xyz, k=neighbours + 1, workers=-1)
    mean_neighbour_distance = distances[:, 1:].mean(axis=1)
    global_mean = float(mean_neighbour_distance.mean())
    global_std = float(mean_neighbour_distance.std(ddof=0))
    threshold = global_mean + std_ratio * global_std
    keep = mean_neighbour_distance <= threshold
    return keep, {
        "sor_mean_neighbour_distance": global_mean,
        "sor_std_neighbour_distance": global_std,
        "sor_distance_threshold": float(threshold),
    }


def point_record_dtype() -> np.dtype:
    return np.dtype(
        [
            ("x", "<f4"),
            ("y", "<f4"),
            ("z", "<f4"),
            ("red", "u1"),
            ("green", "u1"),
            ("blue", "u1"),
            ("confidence", "<u2"),
            ("range_raw", "<u2"),
            ("source_frame", "<u2"),
            ("pixel_u", "<u2"),
            ("pixel_v", "<u2"),
            ("stable_observations", "<u2"),
        ]
    )


def collect_selected_points(
    input_root: Path,
    stems: list[str],
    selected_frame: np.ndarray,
    final_pixel_mask: np.ndarray,
    confidence_threshold: int,
    temporal_median: np.ndarray,
    temporal_threshold: int,
    stable_counts: np.ndarray,
) -> tuple[np.ndarray, list[dict[str, object]]]:
    output_chunks: list[np.ndarray] = []
    frame_rows: list[dict[str, object]] = []
    flat_final = final_pixel_mask.ravel()
    flat_selected_frame = selected_frame.ravel()

    for frame_index, stem in enumerate(stems):
        assigned_pixels = np.flatnonzero(
            flat_final & (flat_selected_frame == frame_index)
        )
        if assigned_pixels.size == 0:
            frame_rows.append(
                {
                    "frame_index": frame_index,
                    "stem": stem,
                    "assigned_pixels": 0,
                    "mapped_points": 0,
                    "mapping_misses": 0,
                }
            )
            continue

        ply_path = input_root / f"{stem}-points_intensity.ply"
        ply = PlyData.read(str(ply_path))
        vertices = ply["vertex"].data
        range_grid = ply["range_grid"].data["vertex_indices"]
        pixel_to_vertex = np.fromiter(
            (
                int(indices[0]) if len(indices) == 1 else -1
                for indices in range_grid
            ),
            dtype=np.int32,
            count=WIDTH * HEIGHT,
        )
        vertex_indices = pixel_to_vertex[assigned_pixels]
        mapping_valid = vertex_indices >= 0
        mapped_pixels = assigned_pixels[mapping_valid]
        mapped_vertices = vertex_indices[mapping_valid]

        confidence = load_u16(input_root / f"{stem}-confidence.tiff").ravel()
        ranges = load_u16(input_root / f"{stem}-range_gray.tiff").ravel()
        median_flat = temporal_median.ravel()
        temporal_ok = (
            np.abs(
                ranges[mapped_pixels].astype(np.int32)
                - median_flat[mapped_pixels].astype(np.int32)
            )
            <= temporal_threshold
        )
        confidence_ok = confidence[mapped_pixels] >= confidence_threshold
        verified = temporal_ok & confidence_ok
        mapped_pixels = mapped_pixels[verified]
        mapped_vertices = mapped_vertices[verified]

        chunk = np.empty(mapped_pixels.size, dtype=point_record_dtype())
        for name in ("x", "y", "z", "red", "green", "blue"):
            chunk[name] = vertices[name][mapped_vertices]
        chunk["confidence"] = confidence[mapped_pixels]
        chunk["range_raw"] = ranges[mapped_pixels]
        chunk["source_frame"] = frame_index
        chunk["pixel_u"] = mapped_pixels % WIDTH
        chunk["pixel_v"] = mapped_pixels // WIDTH
        chunk["stable_observations"] = stable_counts.ravel()[mapped_pixels]
        output_chunks.append(chunk)

        frame_rows.append(
            {
                "frame_index": frame_index,
                "stem": stem,
                "assigned_pixels": int(assigned_pixels.size),
                "mapped_points": int(mapped_pixels.size),
                "mapping_misses": int(assigned_pixels.size - mapped_pixels.size),
            }
        )

    if not output_chunks:
        raise RuntimeError("No points survived temporal selection")
    return np.concatenate(output_chunks), frame_rows


def save_preview(
    path: Path,
    records_before_sor: np.ndarray,
    records_after_sor: np.ndarray,
    temporal_median: np.ndarray,
    final_pixel_mask: np.ndarray,
    edge_mask: np.ndarray,
    max_points: int = 100_000,
) -> None:
    rng = np.random.default_rng(42)

    def sample(records: np.ndarray) -> np.ndarray:
        if len(records) <= max_points:
            return records
        indices = rng.choice(len(records), size=max_points, replace=False)
        return records[indices]

    before = sample(records_before_sor)
    after = sample(records_after_sor)
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    depth_display = temporal_median.astype(float)
    depth_display[depth_display == 0] = np.nan
    axes[0, 0].imshow(depth_display, cmap="turbo")
    axes[0, 0].set_title("Mediana temporal de rango")
    axes[0, 0].axis("off")

    diagnostic = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
    diagnostic[final_pixel_mask] = (40, 180, 60)
    diagnostic[edge_mask] = (220, 60, 40)
    axes[0, 1].imshow(diagnostic)
    axes[0, 1].set_title("Verde: conservado | rojo: borde/oclusion")
    axes[0, 1].axis("off")

    axes[1, 0].scatter(
        before["x"],
        before["y"],
        c=before["red"],
        cmap="gray",
        s=0.15,
        linewidths=0,
    )
    axes[1, 0].set_title(f"Antes de SOR: {len(records_before_sor):,} puntos")
    axes[1, 0].set_aspect("equal")
    axes[1, 0].invert_yaxis()

    axes[1, 1].scatter(
        after["x"],
        after["y"],
        c=after["red"],
        cmap="gray",
        s=0.15,
        linewidths=0,
    )
    axes[1, 1].set_title(f"Despues de SOR: {len(records_after_sor):,} puntos")
    axes[1, 1].set_aspect("equal")
    axes[1, 1].invert_yaxis()

    for axis in axes[1]:
        axis.set_xlabel("X [mm]")
        axis.set_ylabel("Y [mm]")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=Path("captures/ToF"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/tof_cleaning/blaze101_canonical"),
    )
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--confidence-threshold", type=int, default=1800)
    parser.add_argument("--temporal-threshold", type=int, default=200)
    parser.add_argument("--min-stable-fraction", type=float, default=0.60)
    parser.add_argument("--edge-threshold", type=int, default=200)
    parser.add_argument("--edge-dilation-pixels", type=int, default=1)
    parser.add_argument("--sor-neighbours", type=int, default=20)
    parser.add_argument("--sor-std-ratio", type=float, default=2.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    started = time.perf_counter()
    point_files = sorted(args.input_root.glob("*-points_intensity.ply"))
    if args.max_frames is not None:
        point_files = point_files[: args.max_frames]
    if not point_files:
        raise FileNotFoundError(f"No points_intensity PLY files in {args.input_root}")

    stems = [
        path.name.removesuffix("-points_intensity.ply")
        for path in point_files
    ]
    for stem in stems:
        for suffix in ("confidence.tiff", "range_gray.tiff", "points_intensity.ply"):
            path = args.input_root / f"{stem}-{suffix}"
            if not path.exists():
                raise FileNotFoundError(path)

    range_stack = np.empty((len(stems), HEIGHT, WIDTH), dtype=np.uint16)
    for index, stem in enumerate(stems):
        range_stack[index] = load_u16(
            args.input_root / f"{stem}-range_gray.tiff"
        )

    valid_stack = range_stack > 0
    masked_ranges = np.where(valid_stack, range_stack, np.nan).astype(np.float32)
    temporal_median_float = np.nanmedian(masked_ranges, axis=0)
    temporal_valid = np.isfinite(temporal_median_float)
    temporal_median = np.where(
        temporal_valid, np.rint(temporal_median_float), 0
    ).astype(np.uint16)
    del masked_ranges

    selected_frame = np.full((HEIGHT, WIDTH), -1, dtype=np.int16)
    best_difference = np.full((HEIGHT, WIDTH), np.iinfo(np.uint16).max, dtype=np.uint16)
    stable_counts = np.zeros((HEIGHT, WIDTH), dtype=np.uint16)

    frame_quality_rows = []
    for frame_index, stem in enumerate(stems):
        confidence = load_u16(args.input_root / f"{stem}-confidence.tiff")
        ranges = range_stack[frame_index]
        difference = np.abs(
            ranges.astype(np.int32) - temporal_median.astype(np.int32)
        )
        valid = ranges > 0
        confidence_ok = confidence >= args.confidence_threshold
        temporal_ok = temporal_valid & (difference <= args.temporal_threshold)
        stable = valid & confidence_ok & temporal_ok
        stable_counts += stable.astype(np.uint16)

        update = stable & (difference < best_difference)
        selected_frame[update] = frame_index
        best_difference[update] = difference[update].astype(np.uint16)
        frame_quality_rows.append(
            {
                "frame_index": frame_index,
                "stem": stem,
                "valid_range_pixels": int(valid.sum()),
                "confidence_pixels": int((valid & confidence_ok).sum()),
                "temporally_stable_pixels": int(stable.sum()),
            }
        )

    min_stable_count = max(
        1, int(math.ceil(args.min_stable_fraction * len(stems)))
    )
    stable_pixel_mask = (
        temporal_valid
        & (selected_frame >= 0)
        & (stable_counts >= min_stable_count)
    )
    edges = depth_edge_mask(
        temporal_median,
        temporal_valid,
        args.edge_threshold,
        args.edge_dilation_pixels,
    )
    final_pixel_mask = stable_pixel_mask & ~edges

    records_before_sor, mapping_rows = collect_selected_points(
        args.input_root,
        stems,
        selected_frame,
        final_pixel_mask,
        args.confidence_threshold,
        temporal_median,
        args.temporal_threshold,
        stable_counts,
    )

    xyz = np.column_stack(
        [
            records_before_sor["x"],
            records_before_sor["y"],
            records_before_sor["z"],
        ]
    ).astype(np.float64)
    finite = np.isfinite(xyz).all(axis=1)
    records_before_sor = records_before_sor[finite]
    xyz = xyz[finite]

    sor_keep, sor_metrics = statistical_outlier_mask(
        xyz,
        args.sor_neighbours,
        args.sor_std_ratio,
    )
    records_after_sor = records_before_sor[sor_keep]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_ply = (
        args.output_dir
        / "blaze101_tof_canonical_clean_conf_temporal_edges_sor20_std2.ply"
    )
    PlyData(
        [PlyElement.describe(records_after_sor, "vertex")],
        text=False,
    ).write(str(output_ply))

    save_mask(args.output_dir / "mask_temporally_stable.png", stable_pixel_mask)
    save_mask(args.output_dir / "mask_depth_edges_rejected.png", edges)
    save_mask(args.output_dir / "mask_final_selected.png", final_pixel_mask)
    Image.fromarray(temporal_median, mode="I;16").save(
        args.output_dir / "range_temporal_median.tiff"
    )

    frame_stats = pd.DataFrame(frame_quality_rows).merge(
        pd.DataFrame(mapping_rows),
        on=["frame_index", "stem"],
        how="left",
    )
    frame_stats.to_csv(args.output_dir / "frame_cleaning_stats.csv", index=False)

    elapsed = time.perf_counter() - started
    metrics = {
        "input_root": str(args.input_root),
        "frames": len(stems),
        "image_width": WIDTH,
        "image_height": HEIGHT,
        "possible_pixels": WIDTH * HEIGHT,
        "temporally_valid_pixels": int(temporal_valid.sum()),
        "minimum_stable_observations": min_stable_count,
        "temporally_stable_pixels": int(stable_pixel_mask.sum()),
        "depth_edge_pixels_rejected": int((stable_pixel_mask & edges).sum()),
        "selected_pixels_before_mapping": int(final_pixel_mask.sum()),
        "points_before_sor": int(len(records_before_sor)),
        "points_after_sor": int(len(records_after_sor)),
        "sor_removed_points": int((~sor_keep).sum()),
        "sor_removed_percent": float(100.0 * (~sor_keep).sum() / len(sor_keep)),
        "confidence_threshold": args.confidence_threshold,
        "temporal_threshold_raw": args.temporal_threshold,
        "temporal_threshold_approx_mm": float(args.temporal_threshold * 0.15243753),
        "min_stable_fraction": args.min_stable_fraction,
        "edge_threshold_raw": args.edge_threshold,
        "edge_threshold_approx_mm": float(args.edge_threshold * 0.15243753),
        "edge_dilation_pixels": args.edge_dilation_pixels,
        "sor_neighbours": args.sor_neighbours,
        "sor_std_ratio": args.sor_std_ratio,
        **sor_metrics,
        "output_ply": str(output_ply),
        "elapsed_seconds": elapsed,
        "notes": [
            "Intensity grayscale is retained as vertex RGB for inspection.",
            "The cloud remains in the original Basler camera coordinate system.",
            "No plane/ground segmentation is applied in this cleaning stage.",
            "Input captures are not modified.",
        ],
    }
    (args.output_dir / "cleaning_metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    save_preview(
        args.output_dir / "cleaning_preview.png",
        records_before_sor,
        records_after_sor,
        temporal_median,
        final_pixel_mask,
        edges,
    )
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
