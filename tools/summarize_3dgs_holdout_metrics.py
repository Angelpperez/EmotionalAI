"""Summarize per-view PSNR/SSIM from gsplat side-by-side validation renders."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
from torchmetrics.image import (
    PeakSignalNoiseRatio,
    StructuralSimilarityIndexMeasure,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--renders-dir", required=True, type=Path)
    parser.add_argument("--images-dir", required=True, type=Path)
    parser.add_argument("--test-every", type=int, default=8)
    parser.add_argument("--step", type=int, default=6999)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args()


def image_tensor(array: np.ndarray) -> torch.Tensor:
    return (
        torch.from_numpy(array.copy())
        .permute(2, 0, 1)
        .unsqueeze(0)
        .float()
        / 255.0
    )


def summary(values: list[float]) -> dict[str, float]:
    return {
        "mean": statistics.fmean(values),
        "std": statistics.stdev(values) if len(values) > 1 else 0.0,
        "min": min(values),
        "max": max(values),
        "median": statistics.median(values),
    }


def make_contact_sheet(
    rows: list[dict[str, object]],
    output_path: Path,
    sample_count: int = 6,
) -> None:
    selected_indices = np.linspace(
        0, len(rows) - 1, min(sample_count, len(rows)), dtype=int
    )
    tiles = []
    for row_index in selected_indices:
        row = rows[int(row_index)]
        image = Image.open(str(row["render_pair"])).convert("RGB")
        max_width = 900
        if image.width > max_width:
            height = round(image.height * max_width / image.width)
            image = image.resize((max_width, height), Image.Resampling.LANCZOS)
        canvas = Image.new("RGB", (image.width, image.height + 54), "white")
        canvas.paste(image, (0, 54))
        draw = ImageDraw.Draw(canvas)
        draw.text(
            (12, 8),
            (
                f"{row['image_name']} | PSNR {float(row['psnr_db']):.2f} dB "
                f"| SSIM {float(row['ssim']):.4f} | real / 3DGS"
            ),
            fill="black",
        )
        tiles.append(canvas)

    columns = 2
    rows_count = math.ceil(len(tiles) / columns)
    tile_width = max(tile.width for tile in tiles)
    tile_height = max(tile.height for tile in tiles)
    sheet = Image.new(
        "RGB",
        (columns * tile_width, rows_count * tile_height),
        (235, 235, 235),
    )
    for index, tile in enumerate(tiles):
        x = (index % columns) * tile_width
        y = (index // columns) * tile_height
        sheet.paste(tile, (x, y))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path, quality=95)


def main() -> None:
    args = parse_args()
    render_paths = sorted(
        args.renders_dir.glob(f"val_step{args.step}_*.png")
    )
    image_paths = sorted(args.images_dir.glob("*"))
    image_paths = [
        path
        for index, path in enumerate(image_paths)
        if path.is_file() and index % args.test_every == 0
    ]
    if not render_paths:
        raise FileNotFoundError(
            f"No validation renders found in {args.renders_dir}"
        )
    if len(render_paths) != len(image_paths):
        raise ValueError(
            f"Render/image count mismatch: {len(render_paths)} vs "
            f"{len(image_paths)}"
        )

    psnr_metric = PeakSignalNoiseRatio(data_range=1.0)
    ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0)
    rows: list[dict[str, object]] = []
    for validation_index, (render_path, image_path) in enumerate(
        zip(render_paths, image_paths)
    ):
        pair = np.asarray(Image.open(render_path).convert("RGB"))
        half_width = pair.shape[1] // 2
        reference = pair[:, :half_width]
        prediction = pair[:, half_width : half_width * 2]
        reference_tensor = image_tensor(reference)
        prediction_tensor = image_tensor(prediction)
        psnr = float(psnr_metric(prediction_tensor, reference_tensor))
        ssim = float(ssim_metric(prediction_tensor, reference_tensor))
        rows.append(
            {
                "validation_index": validation_index,
                "dataset_index": validation_index * args.test_every,
                "image_name": image_path.name,
                "render_pair": str(render_path),
                "psnr_db": psnr,
                "ssim": ssim,
            }
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "holdout_per_view_metrics.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    psnr_values = [float(row["psnr_db"]) for row in rows]
    ssim_values = [float(row["ssim"]) for row in rows]
    result = {
        "views": len(rows),
        "test_every": args.test_every,
        "held_out_dataset_indices": [
            int(row["dataset_index"]) for row in rows
        ],
        "psnr_db_png_recomputed": summary(psnr_values),
        "ssim_png_recomputed": summary(ssim_values),
        "note": (
            "Per-view values are recomputed from the saved 8-bit side-by-side "
            "validation PNGs. The trainer JSON contains the authoritative "
            "mean computed from float tensors before PNG quantization."
        ),
    }
    (args.output_dir / "holdout_per_view_summary.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    make_contact_sheet(
        rows,
        args.output_dir / "holdout_real_vs_3dgs_contact_sheet.jpg",
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
