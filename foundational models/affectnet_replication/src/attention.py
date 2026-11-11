"""Residual attention rollout from actual eager attention weights; no causal claims."""
from __future__ import annotations

from src.utils import write_csv, write_json


def attention_rollout(attentions):
    import numpy as np
    if not attentions:
        raise ValueError("Model returned no attention tensors.")
    tokens = attentions[0].shape[-1]
    rollout = np.eye(tokens, dtype=np.float64)
    for layer in attentions:
        if layer.ndim != 4 or layer.shape[0] != 1 or layer.shape[-2:] != (tokens, tokens):
            raise ValueError("Unexpected attention tensor shape.")
        if not np.isfinite(layer).all() or (layer < -1e-7).any():
            raise ValueError("Attention weights are nonfinite or negative.")
        if not np.allclose(layer.sum(axis=-1), 1, atol=1e-5):
            raise ValueError("Attention rows do not sum to one in evaluation mode.")
        average = layer[0].mean(axis=0).astype(np.float64) + np.eye(tokens)
        average /= average.sum(axis=-1, keepdims=True)
        rollout = average @ rollout
    patches = tokens - 1
    side = int(round(patches ** 0.5))
    if side * side != patches:
        raise ValueError("Non-square patch lattice requires an explicit spatial mapping.")
    return rollout[0, 1:].reshape(side, side).astype("float32")


def extract_attention(runtime, selected, output):
    import numpy as np
    import torch
    from PIL import Image
    model, processor, device = runtime["model"], runtime["processor"], runtime["device"]
    canonical_ids = runtime["canonical_ids"]
    saved = np.load(output / "predictions" / "logits_vit.npy", allow_pickle=False)
    maps, details = {}, []
    destination = output / "attention"
    destination.mkdir(exist_ok=True)
    model.eval()
    for row in selected:
        with Image.open(row["filepath"]) as im:
            inputs = processor(images=im.convert("RGB"), return_tensors="pt").to(device)
        with torch.inference_mode():
            result = model(**inputs, output_attentions=True)
        logits = result.logits[:, canonical_ids].cpu().numpy()[0]
        reference = saved[int(row["prediction_index"])]
        if not np.allclose(logits, reference, atol=1e-4, rtol=1e-4) or int(logits.argmax()) != int(reference.argmax()):
            raise ValueError("Attention pass changed the reference prediction/logits beyond tolerance.")
        layers = [a.cpu().numpy() for a in result.attentions] if result.attentions is not None else []
        grid = attention_rollout(layers)
        np.save(destination / f"{row['sample_id']}.npy", grid, allow_pickle=False)
        maps[row["sample_id"]] = grid
        details.append({"sample_id": row["sample_id"], "n_layers": len(layers),
                        "n_heads": layers[0].shape[1], "n_tokens": layers[0].shape[-1],
                        "max_absolute_logit_difference": float(np.max(np.abs(logits-reference))),
                        "map_min": float(grid.min()), "map_max": float(grid.max())})
    write_csv(destination / "attention_validation.csv", details,
              ["sample_id", "n_layers", "n_heads", "n_tokens", "max_absolute_logit_difference", "map_min", "map_max"])
    write_json(destination / "metadata.json", {"status": "completed", "n_examples": len(maps),
        "method": "head mean; residual identity; row normalization; multiply layers; CLS-to-patch row",
        "backend": "eager", "weights_modified": False, "normalization": "min-max per image for display only",
        "overlay_alpha": 0.45, "overlay_interpolation": "bilinear", "warning": "Attention visualization is interpretative, not causal."})
    return maps
