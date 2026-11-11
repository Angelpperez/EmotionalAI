"""Generate provisional rock masks for COLMAP.

The masks are intentionally treated as provisional. They are useful to test
whether background leakage is the dominant failure mode, but they must be
visually audited before they are used as doctoral evidence.

COLMAP expects per-image masks as:

    mask_root / relative_image_path + ".png"

For example, image `images/level_01_0000.jpg` maps to
`masks/level_01_0000.jpg.png`.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


def find_images(root: Path) -> list[Path]:
    extensions = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
    return sorted(path for path in root.rglob("*") if path.suffix.lower() in extensions)


def largest_components(mask: np.ndarray, min_area: int, max_components: int) -> np.ndarray:
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if count <= 1:
        return mask

    components: list[tuple[int, int]] = []
    for label in range(1, count):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area >= min_area:
            components.append((area, label))

    components.sort(reverse=True)
    keep = np.zeros_like(mask)
    for _, label in components[:max_components]:
        keep[labels == label] = 255
    return keep


def generate_mask(image: np.ndarray, max_width: int) -> np.ndarray:
    original_height, original_width = image.shape[:2]
    scale = 1.0
    if original_width > max_width:
        scale = max_width / float(original_width)
        image = cv2.resize(
            image,
            (max_width, int(round(original_height * scale))),
            interpolation=cv2.INTER_AREA,
        )

    height, width = image.shape[:2]
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    h_grid, w_grid = np.indices((height, width))
    central = (w_grid > width * 0.03) & (w_grid < width * 0.97)
    vertical = (h_grid > height * 0.08) & (h_grid < height * 0.92)

    value = hsv[:, :, 2]
    saturation = hsv[:, :, 1]
    lightness = lab[:, :, 0]

    dark_rock_seed = (value < 165) & (lightness < 175) & central & vertical
    textured_seed = (cv2.Laplacian(gray, cv2.CV_32F) ** 2 > 20.0) & (value < 190) & central & vertical
    probable_fg = dark_rock_seed | textured_seed

    sure_bg = np.zeros((height, width), dtype=bool)
    sure_bg[: int(height * 0.07), :] = True
    sure_bg[int(height * 0.94) :, :] = True
    sure_bg[:, : int(width * 0.02)] = True
    sure_bg[:, int(width * 0.98) :] = True
    sure_bg |= (value > 205) & (saturation < 55)

    grabcut = np.full((height, width), cv2.GC_PR_BGD, dtype=np.uint8)
    grabcut[probable_fg] = cv2.GC_PR_FGD
    grabcut[sure_bg] = cv2.GC_BGD

    eroded_fg = cv2.erode(probable_fg.astype(np.uint8) * 255, np.ones((7, 7), np.uint8), iterations=1)
    grabcut[eroded_fg > 0] = cv2.GC_FGD

    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)
    cv2.grabCut(image, grabcut, None, bgd_model, fgd_model, 4, cv2.GC_INIT_WITH_MASK)

    mask = np.where((grabcut == cv2.GC_FGD) | (grabcut == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8), iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((13, 13), np.uint8), iterations=2)
    mask = largest_components(mask, min_area=max(1500, int(width * height * 0.002)), max_components=4)
    if scale != 1.0:
        mask = cv2.resize(mask, (original_width, original_height), interpolation=cv2.INTER_NEAREST)
    return mask


def overlay_mask(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    overlay = image.copy()
    color = np.zeros_like(image)
    color[:, :, 1] = 255
    alpha = (mask > 0).astype(np.float32)[:, :, None] * 0.35
    overlay = (overlay * (1.0 - alpha) + color * alpha).astype(np.uint8)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, (0, 255, 255), 2)
    return overlay


def write_contact_sheet(samples: list[np.ndarray], labels: list[str], output: Path, cols: int = 4) -> None:
    if not samples:
        return
    thumb_w = 320
    thumb_h = int(samples[0].shape[0] * (thumb_w / samples[0].shape[1]))
    thumbs = []
    for image, label in zip(samples, labels):
        thumb = cv2.resize(image, (thumb_w, thumb_h), interpolation=cv2.INTER_AREA)
        cv2.putText(thumb, label, (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
        thumbs.append(thumb)

    rows = int(np.ceil(len(thumbs) / cols))
    sheet = np.zeros((rows * thumb_h, cols * thumb_w, 3), dtype=np.uint8)
    for idx, thumb in enumerate(thumbs):
        row = idx // cols
        col = idx % cols
        sheet[row * thumb_h : (row + 1) * thumb_h, col * thumb_w : (col + 1) * thumb_w] = thumb

    output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output), sheet)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-root", required=True, type=Path)
    parser.add_argument("--mask-root", required=True, type=Path)
    parser.add_argument("--contact-sheet", required=True, type=Path)
    parser.add_argument("--sample-count", type=int, default=16)
    parser.add_argument("--max-width", type=int, default=640)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    images = find_images(args.image_root)
    if not images:
        raise SystemExit(f"No images found in {args.image_root}")

    samples: list[np.ndarray] = []
    labels: list[str] = []
    sample_indices = set(np.linspace(0, len(images) - 1, min(args.sample_count, len(images)), dtype=int).tolist())

    coverage = []
    for idx, image_path in enumerate(images):
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Could not read {image_path}")

        rel = image_path.relative_to(args.image_root)
        mask_path = args.mask_root / Path(str(rel) + ".png")
        if mask_path.exists() and not args.overwrite:
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if mask is None:
                raise ValueError(f"Could not read existing mask {mask_path}")
        else:
            mask = generate_mask(image, args.max_width)
            mask_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(mask_path), mask)
        coverage.append(float((mask > 0).mean()))

        if idx in sample_indices:
            samples.append(overlay_mask(image, mask))
            labels.append(rel.name)

    write_contact_sheet(samples, labels, args.contact_sheet)
    print(f"images: {len(images)}")
    print(f"mask_root: {args.mask_root}")
    print(f"coverage_mean: {np.mean(coverage):.4f}")
    print(f"coverage_min: {np.min(coverage):.4f}")
    print(f"coverage_max: {np.max(coverage):.4f}")
    print(f"contact_sheet: {args.contact_sheet}")


if __name__ == "__main__":
    main()
