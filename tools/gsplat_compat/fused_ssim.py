"""Reemplazo PyTorch para la extension CUDA opcional de SSIM."""

from __future__ import annotations

import torch
import torch.nn.functional as F


_WINDOW_CACHE: dict[tuple[int, int, torch.device, torch.dtype], torch.Tensor] = {}


def _gaussian_window(
    window_size: int, channel: int, device: torch.device, dtype: torch.dtype
) -> torch.Tensor:
    key = (window_size, channel, device, dtype)
    if key not in _WINDOW_CACHE:
        coords = torch.arange(window_size, device=device, dtype=torch.float32)
        coords = coords - window_size // 2
        kernel_1d = torch.exp(-(coords**2) / (2 * 1.5**2))
        kernel_1d = kernel_1d / kernel_1d.sum()
        kernel_2d = kernel_1d[:, None] @ kernel_1d[None, :]
        window = kernel_2d[None, None].expand(channel, 1, -1, -1).contiguous()
        _WINDOW_CACHE[key] = window.to(dtype=dtype)
    return _WINDOW_CACHE[key]


def fused_ssim(
    img1: torch.Tensor,
    img2: torch.Tensor,
    padding: str = "valid",
    window_size: int = 11,
    **_: object,
) -> torch.Tensor:
    """Calcula el SSIM medio para tensores BCHW normalizados."""
    if img1.shape != img2.shape:
        raise ValueError(
            f"Las entradas SSIM deben tener la misma forma: {img1.shape}, {img2.shape}"
        )
    if img1.ndim != 4:
        raise ValueError(f"Las entradas SSIM deben ser BCHW: ndim={img1.ndim}")

    channel = img1.shape[1]
    pad = 0 if padding == "valid" else window_size // 2
    window = _gaussian_window(window_size, channel, img1.device, img1.dtype)

    mu1 = F.conv2d(img1, window, padding=pad, groups=channel)
    mu2 = F.conv2d(img2, window, padding=pad, groups=channel)
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=pad, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=pad, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=pad, groups=channel) - mu1_mu2

    c1 = 0.01**2
    c2 = 0.03**2
    ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / (
        (mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2)
    )
    return ssim_map.mean()
