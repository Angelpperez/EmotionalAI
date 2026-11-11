from __future__ import annotations

from pathlib import Path

from config import CLASSES, MODEL_ID
from src.metrics import save_evaluation
from src.utils import normalize_label, write_csv, write_json

FILTERED_PREDICTION_FIELDS = ["sample_id", "prediction_index", "filepath", "split", "source_label",
                              "canonical_true_label", "true_label", "predicted_label", "confidence", "correct",
                              "file_sha256", "rgb_sha256", *[f"p_{c}" for c in CLASSES]]


def run_filtered_vit(records, output, profile, device="auto", batch_size=None):
    """One frozen evaluation; persisted arrays use canonical, not checkpoint, class order."""
    import csv
    import time
    import numpy as np
    import torch
    from PIL import Image
    from huggingface_hub import snapshot_download
    from transformers import AutoImageProcessor, AutoModelForImageClassification
    from src.metrics import validate_prediction_arrays
    from src.utils import sha256_file, utc_now

    if not records or any(r["split"] != "validation" for r in records):
        raise ValueError("Only the frozen validation manifest may enter this inference.")
    device = ("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else device
    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA requested but unavailable.")
    revision = profile["model_revision"]
    cache = str(Path(profile["cache_root"]) / "huggingface")
    snapshot = Path(snapshot_download(
        repo_id=profile["model_id"], revision=revision, cache_dir=cache,
        local_dir=Path(cache) / "checkpoints" / revision,
        allow_patterns=["config.json", "preprocessor_config.json", "model.safetensors"]))
    if snapshot.name != revision:
        raise ValueError(f"Unexpected resolved model snapshot: {snapshot}")
    processor = AutoImageProcessor.from_pretrained(snapshot, use_fast=False, local_files_only=True, trust_remote_code=False)
    model = AutoModelForImageClassification.from_pretrained(snapshot, use_safetensors=True,
        local_files_only=True, trust_remote_code=False, attn_implementation=profile["attention_implementation"])
    mapping = verified_model_mapping(model.config.id2label)
    canonical_ids = [next(i for i, c in mapping.items() if c == name) for name in CLASSES]
    if not isinstance(model.classifier, torch.nn.Linear):
        raise ValueError("CLS extraction requires a verified linear classifier.")
    model.to(device).eval().requires_grad_(False)
    info = {"model_id": profile["model_id"], "resolved_revision": revision, "snapshot_path": str(snapshot),
            "id2label": model.config.id2label, "canonical_mapping": mapping,
            "canonical_output_order": list(CLASSES), "model_ids_in_canonical_order": canonical_ids,
            "processor": processor.to_dict(), "device": device, "dtype": "float32",
            "attention_implementation": profile["attention_implementation"], "fine_tuning": False,
            "files": {p.name: {"sha256": sha256_file(p), "n_bytes": p.stat().st_size} for p in sorted(snapshot.iterdir()) if p.is_file()},
            "cls_extraction": "input to model.classifier, after final encoder normalization"}
    write_json(output / "metrics" / "vit_model_metadata.json", info)
    print(f"model.config.id2label = {model.config.id2label}; canonical mapping = {mapping}", flush=True)
    captured = {}
    handle = model.classifier.register_forward_pre_hook(lambda _module, inputs: captured.update(cls=inputs[0].detach()))

    def forward(items):
        images = []
        for row in items:
            try:
                with Image.open(row["filepath"]) as im:
                    images.append(im.convert("RGB"))
            except (OSError, ValueError) as exc:
                write_csv(output / "predictions" / "prediction_failures.csv",
                          [{"sample_id": row["sample_id"], "filepath": row["filepath"], "reason": str(exc)}],
                          ["sample_id", "filepath", "reason"])
                raise RuntimeError(f"Frozen evaluation image became unreadable: {row['filepath']}") from exc
        inputs = processor(images=images, return_tensors="pt").to(device)
        with torch.inference_mode():
            logits_model_order = model(**inputs).logits
            cls = captured.pop("cls")
            reconstructed = model.classifier(cls)
            captured.clear()
            if not torch.allclose(reconstructed, logits_model_order, atol=1e-5, rtol=1e-4):
                raise ValueError("Captured CLS does not reproduce the classifier logits.")
            logits = logits_model_order[:, canonical_ids]
            probs = logits.softmax(dim=1)
        if not torch.isfinite(logits).all() or not torch.isfinite(cls).all():
            raise ValueError("Nonfinite logits or CLS.")
        return logits.cpu().numpy().copy(), probs.cpu().numpy().copy(), cls.cpu().numpy().copy()

    chosen_batch = batch_size or profile["batch_size"]
    pilot_indexes = np.random.default_rng(profile["seed"]).permutation(len(records))[:profile["pilot_size"]]
    pilot = [records[int(i)] for i in pilot_indexes]
    started = time.perf_counter()
    adjustments = []
    try:
        while True:
            try:
                for i in range(0, len(pilot), chosen_batch):
                    forward(pilot[i:i + chosen_batch])
                break
            except torch.cuda.OutOfMemoryError:
                if chosen_batch == 1:
                    raise
                adjustments.append({"phase": "pilot", "old_batch": chosen_batch, "new_batch": max(1, chosen_batch//2)})
                chosen_batch = max(1, chosen_batch//2)
                captured.clear()
                torch.cuda.empty_cache()
        pilot_seconds = time.perf_counter() - started
        write_json(output / "logs" / "pilot.json", {"status": "completed", "n_images": len(pilot),
            "sample_ids": [r["sample_id"] for r in pilot], "batch_size": chosen_batch, "seconds": pilot_seconds,
            "images_per_second": len(pilot)/pilot_seconds, "adjustments": adjustments,
            "note": "Infrastructure pilot only; predictions excluded from final reporting"})
        print(f"Pilot: {len(pilot)} images in {pilot_seconds:.2f}s; batch={chosen_batch}", flush=True)
        n = len(records)
        paths = {"logits": output / "predictions" / "logits_vit.npy",
                 "probabilities": output / "predictions" / "probabilities_vit.npy",
                 "embeddings": output / "embeddings" / "vit_cls_embeddings.npy"}
        for p in paths.values():
            p.parent.mkdir(parents=True, exist_ok=True)
        arrays, rows, offset = {}, [], 0
        csv_path = output / "predictions" / "predictions_vit.csv"
        started = time.perf_counter()
        with csv_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=FILTERED_PREDICTION_FIELDS)
            writer.writeheader()
            while offset < n:
                batch = records[offset:offset + chosen_batch]
                try:
                    logits, probabilities, cls = forward(batch)
                except torch.cuda.OutOfMemoryError:
                    if chosen_batch == 1:
                        raise
                    adjustments.append({"phase": "evaluation", "offset": offset, "old_batch": chosen_batch,
                                        "new_batch": max(1, chosen_batch//2)})
                    chosen_batch = max(1, chosen_batch//2)
                    captured.clear()
                    torch.cuda.empty_cache()
                    continue
                if not arrays:
                    for name, width in (("logits", 7), ("probabilities", 7), ("embeddings", cls.shape[1])):
                        arrays[name] = np.lib.format.open_memmap(str(paths[name]) + ".partial.npy", mode="w+", dtype="float32", shape=(n, width))
                for name, data in (("logits", logits), ("probabilities", probabilities), ("embeddings", cls)):
                    arrays[name][offset:offset+len(batch)] = data
                    arrays[name].flush()
                for j, record in enumerate(batch):
                    predicted = CLASSES[int(probabilities[j].argmax())]
                    row = {key: record[key] for key in ("sample_id", "filepath", "split", "source_label", "canonical_true_label", "true_label", "file_sha256", "rgb_sha256")}
                    row.update(prediction_index=offset+j, predicted_label=predicted,
                               confidence=float(probabilities[j].max()), correct=predicted == record["true_label"])
                    row.update({f"p_{c}": float(probabilities[j, k]) for k, c in enumerate(CLASSES)})
                    writer.writerow(row)
                    rows.append(row)
                offset += len(batch)
                f.flush()
                write_json(output / "logs" / "progress.json", {"status": "running", "n_expected": n,
                    "n_completed": offset, "n_failed": 0, "batch_size": chosen_batch,
                    "seconds": time.perf_counter()-started, "updated_at_utc": utc_now()})
                if offset % (chosen_batch * 10) == 0 or offset == n:
                    print(f"ViT evaluation: {offset}/{n}", flush=True)
        validate_prediction_arrays(rows, arrays["logits"], arrays["probabilities"], arrays["embeddings"])
        # Close Windows memory mappings before renaming completed arrays.
        for array in arrays.values():
            array.flush()
            array._mmap.close()
        arrays.clear()
        for p in paths.values():
            Path(str(p) + ".partial.npy").replace(p)
        weight = model.classifier.weight.detach().cpu().numpy()[canonical_ids].copy()
        bias = model.classifier.bias.detach().cpu().numpy()[canonical_ids].copy()
        np.savez(output / "embeddings" / "classifier_parameters.npz", weight=weight, bias=bias)
        saved_cls = np.load(paths["embeddings"], allow_pickle=False)
        saved_logits = np.load(paths["logits"], allow_pickle=False)
        reconstructed = saved_cls @ weight.T + bias
        cls_ok = np.allclose(reconstructed, saved_logits, atol=1e-5, rtol=1e-4)
        write_json(output / "embeddings" / "cls_verification.json", {"status": "completed" if cls_ok else "failed",
            "maximum_absolute_logit_error": float(np.max(np.abs(reconstructed-saved_logits))), "atol": 1e-5, "rtol": 1e-4,
            "n_images": n, "embedding_dimension": saved_cls.shape[1]})
        if not cls_ok:
            raise ValueError("Persisted CLS embeddings do not reproduce the stored logits.")
        write_csv(output / "embeddings" / "vit_cls_labels.csv", rows,
                  ["sample_id", "prediction_index", "filepath", "true_label", "predicted_label"])
        write_csv(output / "predictions" / "prediction_failures.csv", [], ["sample_id", "filepath", "reason"])
        elapsed = time.perf_counter()-started
        progress = {"status": "completed", "n_expected": n, "n_completed": n, "n_failed": 0,
                    "seconds": elapsed, "batch_size": chosen_batch, "batch_adjustments": adjustments}
        write_json(output / "logs" / "progress.json", progress)
        return {"status": "completed", "n_processed": n, "n_failed": 0, "model": info, "timing": progress}, {
            "model": model, "processor": processor, "canonical_ids": canonical_ids, "device": device, "rows": rows}
    finally:
        handle.remove()


def verified_model_mapping(id2label):
    mapping = {int(k): normalize_label(v) for k, v in id2label.items()}
    if set(mapping) != set(range(7)) or set(mapping.values()) != set(CLASSES):
        raise ValueError(f"Model id2label must bijectively map seven IDs to the protocol: {id2label}")
    return mapping


def run_vit(records, output, batch_size=32, device="auto", revision="main"):
    validation = [r for r in records if r["split"] == "validation"]
    if not validation:
        return {"status": "blocked", "reason": "No verified validation images.", "n_processed": 0,
                "n_failed": 0, "metrics": None}
    import torch
    from PIL import Image
    from transformers import AutoImageProcessor, AutoModelForImageClassification
    device = ("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else device
    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is unavailable.")
    # Prefer safetensors; never enable remote model code.
    processor = AutoImageProcessor.from_pretrained(MODEL_ID, revision=revision, use_fast=False, trust_remote_code=False)
    model = AutoModelForImageClassification.from_pretrained(MODEL_ID, revision=revision,
                                                            use_safetensors=True, trust_remote_code=False)
    mapping = verified_model_mapping(model.config.id2label)
    info = {"model_id": MODEL_ID, "requested_revision": revision,
            "resolved_revision": getattr(model.config, "_commit_hash", None),
            "raw_id2label": model.config.id2label, "canonical_id2label": mapping,
            "processor": processor.to_dict(), "device": device, "fine_tuning": False,
            "preprocessing": "PIL RGB, model image processor; no additional face crop or alignment"}
    write_json(output / "metrics" / "vit_model_metadata.json", info)
    print(f"Verified model.config.id2label: {model.config.id2label} -> {mapping}", flush=True)
    model.to(device).eval()
    predictions, failures = [], []
    for offset in range(0, len(validation), batch_size):
        images, good = [], []
        for row in validation[offset:offset + batch_size]:
            try:
                with Image.open(row["filepath"]) as image:
                    images.append(image.convert("RGB"))
                good.append(row)
            except (OSError, ValueError) as exc:
                failures.append({"filepath": row["filepath"], "reason": str(exc)})
        if images:
            inputs = processor(images=images, return_tensors="pt").to(device)
            with torch.inference_mode():
                logits = model(**inputs).logits
                if logits.shape[-1] != 7 or not torch.isfinite(logits).all():
                    raise ValueError("Expected seven finite logits per image.")
                confidence, indexes = logits.softmax(dim=-1).max(dim=-1)
            for row, index, probability in zip(good, indexes.cpu().tolist(), confidence.cpu().tolist()):
                predictions.append({"filepath": row["filepath"], "true_label": row["true_label"],
                                    "predicted_label": mapping[index], "confidence": probability})
        print(f"ViT: {min(offset + batch_size, len(validation))}/{len(validation)} images", flush=True)
    write_csv(output / "predictions" / "vit_failures.csv", failures, ["filepath", "reason"])
    result = {"status": "completed" if predictions else "failed", "n_attempted": len(validation),
              "n_processed": len(predictions), "n_failed": len(failures), "model": info, "metrics": None}
    if predictions:
        result["metrics"] = save_evaluation(predictions, output, "vit", {
            "evaluation_population": "all readable validation images; no OpenFace filtering",
            "n_attempted": len(validation), "n_failed": len(failures)})
    return result
