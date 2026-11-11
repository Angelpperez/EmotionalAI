from __future__ import annotations

from pathlib import Path

from config import CLASSES, PAPER_ACCURACY_PERCENT
from src.utils import write_csv, write_json


def classification_metrics(y_true, y_pred):
    import numpy as np
    from sklearn.metrics import (accuracy_score, balanced_accuracy_score, confusion_matrix,
                                 precision_recall_fscore_support)
    if len(y_true) == 0 or len(y_true) != len(y_pred):
        raise ValueError("Metrics require nonempty, equally sized true and predicted labels.")
    if (set(y_true) | set(y_pred)) - set(CLASSES):
        raise ValueError("Metrics accept exactly the seven canonical classes.")
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=list(CLASSES), zero_division=0)
    matrix = confusion_matrix(y_true, y_pred, labels=list(CLASSES))
    denominators = matrix.sum(axis=1, keepdims=True)
    normalized = np.divide(matrix, denominators, out=np.zeros(matrix.shape, dtype=float), where=denominators != 0)
    return {"n_images": len(y_true), "accuracy": float(accuracy_score(y_true, y_pred)),
            "macro_f1": float(f1.mean()), "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
            "precision_macro": float(precision.mean()), "recall_macro": float(recall.mean()),
            "uar": float(recall.mean()), "class_order": list(CLASSES),
            "per_class": {c: {"precision": float(p), "recall": float(r), "f1": float(f), "support": int(s)}
                          for c, p, r, f, s in zip(CLASSES, precision, recall, f1, support)},
            "confusion_matrix": matrix.tolist(), "confusion_matrix_normalized_true": normalized.tolist(),
            "missing_true_classes": [c for c, s in zip(CLASSES, support) if not s],
            "units": "fraction [0,1]; confusion_matrix contains integer counts"}


def validate_prediction_arrays(rows, logits, probabilities, embeddings=None):
    import numpy as np
    n = len(rows)
    if logits.shape != (n, 7) or probabilities.shape != (n, 7):
        raise ValueError("Logits/probability rows are not aligned with predictions.")
    if not np.isfinite(logits).all() or not np.isfinite(probabilities).all():
        raise ValueError("Nonfinite prediction arrays.")
    if (probabilities < 0).any() or (probabilities > 1).any() or not np.allclose(probabilities.sum(axis=1), 1, atol=1e-6):
        raise ValueError("Invalid probability simplex.")
    indexes = probabilities.argmax(axis=1)
    if not np.array_equal(logits.argmax(axis=1), indexes):
        raise ValueError("Softmax changed the argmax.")
    stable = np.exp(logits - logits.max(axis=1, keepdims=True))
    stable /= stable.sum(axis=1, keepdims=True)
    if not np.allclose(stable, probabilities, atol=1e-6, rtol=1e-5):
        raise ValueError("Stored probabilities are not the softmax of the stored logits.")
    if len({r["sample_id"] for r in rows}) != n:
        raise ValueError("Predictions contain duplicate sample IDs.")
    for i, row in enumerate(rows):
        if int(row["prediction_index"]) != i or row["predicted_label"] != CLASSES[indexes[i]]:
            raise ValueError("Prediction index or label is not aligned.")
        if row["true_label"] != row["canonical_true_label"]:
            raise ValueError("True label aliases disagree.")
        if not np.isclose(float(row["confidence"]), probabilities[i, indexes[i]], atol=1e-7):
            raise ValueError("Confidence does not match maximum softmax.")
        if bool(row["correct"]) != (row["predicted_label"] == row["true_label"]):
            raise ValueError("Correctness flag disagrees with labels.")
    if embeddings is not None and (len(embeddings) != n or not np.isfinite(embeddings).all()):
        raise ValueError("CLS embeddings are not finite and aligned.")


def calibration_statistics(rows, n_bins=10):
    import numpy as np
    if not rows:
        raise ValueError("Calibration requires real predictions.")
    confidence = np.asarray([float(r["confidence"]) for r in rows])
    correct = np.asarray([r["true_label"] == r["predicted_label"] for r in rows])
    if not np.isfinite(confidence).all() or (confidence < 0).any() or (confidence > 1).any():
        raise ValueError("Invalid confidence values.")
    ids = np.minimum(np.floor(confidence * n_bins).astype(int), n_bins - 1)
    bins, ece = [], 0.0
    for i in range(n_bins):
        mask = ids == i
        n = int(mask.sum())
        acc = float(correct[mask].mean()) if n else None
        conf = float(confidence[mask].mean()) if n else None
        if n:
            ece += n / len(rows) * abs(acc - conf)
        bins.append({"bin": i, "lower": i/n_bins, "upper": (i+1)/n_bins,
                     "right_edge_inclusive": i == n_bins - 1, "n": n, "accuracy": acc, "mean_confidence": conf})
    return {"ece": float(ece), "n_bins": n_bins, "n_images": len(rows), "bins": bins,
            "n_correct": int(correct.sum()), "n_incorrect": int((~correct).sum()),
            "mean_confidence_correct": float(confidence[correct].mean()) if correct.any() else None,
            "mean_confidence_incorrect": float(confidence[~correct].mean()) if (~correct).any() else None,
            "definition": "top-label ECE; equal-width bins; left closed, right open except final bin", "units": "fraction [0,1]"}


def save_filtered_metrics(rows, output, inspection):
    from sklearn.metrics import classification_report
    result = classification_metrics([r["true_label"] for r in rows], [r["predicted_label"] for r in rows])
    if result["missing_true_classes"]:
        raise ValueError("Principal evaluation requires positive support for all seven classes.")
    if abs(result["balanced_accuracy"] - result["uar"]) > 1e-12:
        raise ValueError("Balanced Accuracy and UAR disagree with all seven classes present.")
    result.update(experiment="Technical validation on AffectNetFiltered", dataset="AffectNetFiltered",
                  evaluation_population="provided validation split; protocol 1.0 exclusions; unique readable content",
                  evaluation_manifest_sha256=inspection["evaluation_manifest_sha256"],
                  paper_reference_accuracy=0.439, delta_accuracy_pp=100 * (result["accuracy"] - 0.439),
                  comparison_caveat="Different dataset preprocessing; not a direct replication.")
    write_json(output / "metrics" / "metrics_vit.json", result)
    report = classification_report([r["true_label"] for r in rows], [r["predicted_label"] for r in rows],
                                   labels=list(CLASSES), output_dict=True, zero_division=0)
    report_rows = [{"class": k, "precision": v["precision"], "recall": v["recall"],
                    "f1": v["f1-score"], "support": v["support"]} for k, v in report.items() if isinstance(v, dict)]
    write_csv(output / "metrics" / "classification_report_vit.csv", report_rows, ["class", "precision", "recall", "f1", "support"])
    for key, name in (("confusion_matrix", "confusion_matrix_vit.csv"),
                      ("confusion_matrix_normalized_true", "confusion_matrix_vit_normalized.csv")):
        matrix_rows = [{"true_label": c, **dict(zip(CLASSES, row))} for c, row in zip(CLASSES, result[key])]
        write_csv(output / "metrics" / name, matrix_rows, ["true_label", *CLASSES])
    return result


def save_evaluation(rows, output, slug, extra=None):
    output = Path(output)
    result = classification_metrics([r["true_label"] for r in rows], [r["predicted_label"] for r in rows])
    result.update(extra or {})
    write_json(output / "metrics" / f"metrics_{slug}.json", result)
    write_csv(output / "predictions" / f"predictions_{slug}.csv", rows,
              ["filepath", "true_label", "predicted_label", "confidence"])
    plot_confusion(result, output / "figures" / f"confusion_matrix_{slug}.png", normalize=False)
    plot_confusion(result, output / "figures" / f"confusion_matrix_{slug}_normalized.png", normalize=True)
    return result


def _plotting():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "figure.facecolor": "white", "axes.facecolor": "white"})
    return plt


def plot_confusion(metrics, path, normalize):
    import numpy as np
    plt = _plotting()
    matrix = np.asarray(metrics["confusion_matrix_normalized_true" if normalize else "confusion_matrix"])
    fig, ax = plt.subplots(figsize=(8.4, 7))
    m = ax.imshow(matrix, cmap="Blues", vmin=0, vmax=1 if normalize else None)
    fig.colorbar(m, ax=ax, fraction=0.046)
    ax.set(xticks=range(7), yticks=range(7), xticklabels=CLASSES, yticklabels=CLASSES,
           xlabel="Predicted label", ylabel="True label",
           title=f"{'Normalized confusion matrix' if normalize else 'Confusion matrix'} (N={metrics['n_images']})")
    plt.setp(ax.get_xticklabels(), rotation=35, ha="right")
    for i in range(7):
        for j in range(7):
            rgb = np.asarray(m.cmap(m.norm(matrix[i, j])))[:3]
            linear = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
            luminance = float(linear @ np.asarray([0.2126, 0.7152, 0.0722]))
            ax.text(j, i, f"{matrix[i,j]:.2f}" if normalize else str(int(matrix[i,j])),
                    ha="center", va="center", color="white" if luminance < 0.179 else "black", fontsize=8)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def plot_pipeline(path):
    plt = _plotting()
    fig, ax = plt.subplots(figsize=(11, 3.4))
    ax.set(xlim=(0, 11), ylim=(0, 3.4))
    ax.axis("off")
    for y, labels in ((2.4, ("RGB", "OpenFace", "AU", "SVC", "Prediction")),
                      (0.9, ("RGB", "Image processor", "ViT-FER", "Prediction"))):
        step = 10 / len(labels)
        boxes = []
        for i, label in enumerate(labels):
            x = 0.65 + step * i
            boxes.append(ax.text(x, y, label, ha="center", va="center", fontsize=10,
                         bbox={"boxstyle": "round,pad=0.5", "facecolor": "#edf1f3", "edgecolor": "#697983"}))
        fig.canvas.draw()
        for left, right in zip(boxes, boxes[1:]):
            ax.annotate("", xy=right.get_position(), xytext=left.get_position(),
                        arrowprops={"arrowstyle": "->", "color": "#697983",
                                    "patchA": left.get_bbox_patch(), "patchB": right.get_bbox_patch(),
                                    "shrinkA": 6, "shrinkB": 6})
    ax.text(5.3, 3.15, "Explicit representation / deep representation", ha="center", fontsize=12)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def comparison_table(results, output):
    """Paper numbers never enter OUR RESULT cells. Absent results stay null."""
    output = Path(output)
    rows = []
    for model, paper in PAPER_ACCURACY_PERCENT.items():
        result = results.get(model, {})
        m = result.get("metrics") or {}
        acc = 100 * m["accuracy"] if "accuracy" in m else None
        rows.append({"Model": model, "Representation": "RGB / ViT" if model == "ViT-FER" else "OpenFace / AU / SVC",
                     "Status": result.get("status", "blocked"), "OUR RESULT Accuracy (%)": acc,
                     "OUR RESULT Macro-F1 (%)": 100 * m["macro_f1"] if "macro_f1" in m else None,
                     "OUR RESULT Balanced Accuracy (%)": 100 * m["balanced_accuracy"] if "balanced_accuracy" in m else None,
                     "OUR RESULT UAR (%)": 100 * m["uar"] if "uar" in m else None,
                     "Reference paper Accuracy (%)": paper,
                     "delta_accuracy (percentage points)": acc - paper if acc is not None else None,
                     "N evaluated": m.get("n_images"), "Population": m.get("evaluation_population"),
                     "Reason": result.get("reason", "")})
    write_csv(output / "comparison.csv", rows, list(rows[0]))
    header = "Model | OUR Accuracy (%) | OUR Macro-F1 (%) | OUR Balanced Accuracy (%) | OUR UAR (%) | PAPER Accuracy (%) | delta (pp) | N | Status"
    lines = [header, " | ".join(["---"] * 9)]
    fmt = lambda x: "N/D" if x is None else f"{x:.3f}"
    for r in rows:
        lines.append(" | ".join([r["Model"], fmt(r["OUR RESULT Accuracy (%)"]), fmt(r["OUR RESULT Macro-F1 (%)"]),
                                 fmt(r["OUR RESULT Balanced Accuracy (%)"]), fmt(r["OUR RESULT UAR (%)"]),
                                 fmt(r["Reference paper Accuracy (%)"]), fmt(r["delta_accuracy (percentage points)"]),
                                 str(r["N evaluated"]) if r["N evaluated"] is not None else "N/D", r["Status"]]))
    content = "\n".join(lines) + "\n\nN/D = no experimental result; paper values are reference only.\n"
    (output / "comparison.md").write_text(content, encoding="utf-8")
    return rows, content


def plot_comparisons(rows, output):
    """All five positions are retained; missing runs are labelled, never plotted as zero."""
    if not any(r["OUR RESULT Accuracy (%)"] is not None for r in rows):
        return []
    plt = _plotting()
    paths = []
    for key, slug, title in (("OUR RESULT Accuracy (%)", "accuracy", "Accuracy"),
                             ("OUR RESULT Macro-F1 (%)", "macro_f1", "Macro-F1")):
        fig, ax = plt.subplots(figsize=(10, 5.5))
        for i, r in enumerate(rows):
            value = r[key]
            if value is None:
                ax.text(i, 3, "Not run", ha="center", fontsize=9)
            else:
                ax.bar(i, value, color="#647e8b", width=0.6)
                ax.text(i, value + 1, f"{value:.2f}", ha="center", fontsize=9)
        ax.set(xticks=range(len(rows)), xticklabels=[r["Model"] for r in rows],
               ylabel=f"{title} (%)", ylim=(0, 105), title="OUR RESULT — " + title)
        ax.spines[["top", "right"]].set_visible(False)
        plt.setp(ax.get_xticklabels(), rotation=18, ha="right")
        fig.text(0.5, 0.01, "See comparison.csv for evaluated populations; OpenFace failures can change N.", ha="center", fontsize=8)
        fig.tight_layout(rect=(0, 0.04, 1, 1))
        path = Path(output) / "figures" / f"comparison_{slug}.png"
        fig.savefig(path, dpi=300)
        plt.close(fig)
        paths.append(str(path))
    return paths
