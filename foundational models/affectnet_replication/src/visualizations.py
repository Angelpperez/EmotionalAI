"""Presentation figures from the frozen manifest and actual predictions only."""
from __future__ import annotations

from config import CLASSES
from src.metrics import _plotting, plot_confusion
from src.utils import write_json

COLORS = ("#78858e", "#8f7c9c", "#ad756c", "#b39b61", "#6e88a8", "#7e9980", "#6e9b9e")
CAVEAT = "Different dataset preprocessing; not a direct replication."


def _save(fig, output, name):
    path = output / "figures" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, facecolor="white")
    _plotting().close(fig)
    return path


def _photo(ax, row, annotate=False):
    from PIL import Image
    ax.axis("off")
    if row is None:
        ax.text(.5, .5, "No example available", ha="center", va="center", transform=ax.transAxes)
        return
    with Image.open(row["filepath"]) as im:
        ax.imshow(im.convert("RGB"))
    if annotate:
        ax.set_title(f"TRUE: {row['true_label']}\nPRED: {row['predicted_label']}  p={row['confidence']:.3f}", fontsize=9, pad=6)


def dataset_examples(panels, n, output):
    plt = _plotting()
    fig, axes = plt.subplots(4, 7, figsize=(14, 9))
    for j, c in enumerate(CLASSES):
        samples = panels["dataset"][c]
        for i in range(4):
            _photo(axes[i, j], samples[i] if i < len(samples) else None)
        axes[0, j].set_title(c.title(), fontsize=12, pad=10)
    fig.suptitle("Images in the evaluated AffectNetFiltered split", fontsize=17, y=.97)
    fig.text(.5, .025, f"N = {n} images evaluated | Four random images per class, seed 42", ha="center", fontsize=11)
    fig.subplots_adjust(left=.025, right=.975, top=.90, bottom=.065, wspace=.10, hspace=.08)
    return _save(fig, output, "dataset_examples.png")


def prediction_examples(panels, output):
    plt = _plotting()
    fig = plt.figure(figsize=(15, 9))
    grid = fig.add_gridspec(3, 6, width_ratios=[1.2, 1, 1, 1, 1, 1], hspace=.60, wspace=.12)
    entries = (("A. Correct\nhigh confidence", panels["correct_high"]),
               ("B. Correct\nlow confidence", panels["correct_low"]),
               ("C. Incorrect\nhigh confidence", panels["incorrect_high"]))
    for i, (label, examples) in enumerate(entries):
        ax = fig.add_subplot(grid[i, 0]); ax.axis("off")
        ax.text(.02, .5, label, va="center", fontsize=12)
        for j in range(5):
            _photo(fig.add_subplot(grid[i, j+1]), examples[j] if j < len(examples) else None, annotate=True)
    fig.suptitle("Prediction examples selected by confidence rank", fontsize=17, y=.98)
    fig.text(.5, .02, "Extremes illustrate behavior; they do not estimate prevalence. p = maximum softmax, uncalibrated.", ha="center", fontsize=10)
    fig.subplots_adjust(left=.02, right=.98, top=.88, bottom=.08)
    return _save(fig, output, "prediction_examples.png")


def top_confusions(panels, output):
    plt = _plotting()
    entries = panels["confusions"]
    if not entries:
        raise ValueError("No off-diagonal errors exist for this figure.")
    fig = plt.figure(figsize=(12, 3*len(entries)+1))
    grid = fig.add_gridspec(len(entries), 4, width_ratios=[1.35, 1, 1, 1], hspace=.22)
    for i, entry in enumerate(entries):
        p = entry["pair"]
        ax = fig.add_subplot(grid[i, 0]); ax.axis("off")
        ax.text(.02, .5, f"TRUE: {p['true_label']}\nPRED: {p['predicted_label']}\n\n{100*p['rate']:.2f}% of true class\n{p['n']} / {p['support']} images", va="center", fontsize=12)
        for j in range(3):
            _photo(fig.add_subplot(grid[i, j+1]), entry["rows"][j] if j < len(entry["rows"]) else None)
    fig.suptitle("Largest directed confusions by normalized rate", fontsize=16, y=.98)
    fig.text(.5, .02, "Examples sampled within each pair with seed 42; arrows represent classifier errors, not psychological states.", ha="center", fontsize=9)
    fig.subplots_adjust(left=.02, right=.98, top=.91, bottom=.07)
    return _save(fig, output, "top_confusions.png")


def class_figures(metrics, output, kind):
    plt = _plotting()
    fig, ax = plt.subplots(figsize=(10, 5.8))
    values = [metrics["per_class"][c]["f1"]*100 if kind == "f1" else metrics["per_class"][c]["support"] for c in CLASSES]
    bars = ax.bar(CLASSES, values, color=COLORS)
    for bar, value in zip(bars, values):
        label = f"{value:.2f}" if kind == "f1" else str(int(value))
        ax.annotate(label, (bar.get_x()+bar.get_width()/2, value), xytext=(0, 5), textcoords="offset points", ha="center", fontsize=10)
    ax.set_ylabel("F1 (%)" if kind == "f1" else "Images in final evaluation")
    ax.set_title(f"{'F1 by class' if kind == 'f1' else 'Class distribution'} | N = {metrics['n_images']}")
    ax.set_ylim(0, 105 if kind == "f1" else max(values)*1.15)
    ax.spines[["top", "right"]].set_visible(False)
    fig.text(.5, .025, "AffectNetFiltered | Provided validation split after protocol 1.0 exclusions | contempt excluded", ha="center", fontsize=9)
    fig.tight_layout(rect=(0, .06, 1, 1))
    return _save(fig, output, "f1_per_class_vit.png" if kind == "f1" else "class_distribution.png")


def paper_comparison(metrics, output):
    plt = _plotting()
    fig, ax = plt.subplots(figsize=(9, 6))
    values = [43.9, 100*metrics["accuracy"]]
    bars = ax.bar(["PAPER REFERENCE\nOriginal AffectNet", "OUR RESULT\nAffectNetFiltered"], values, color=["#b1b8bc", "#647e8b"], width=.5)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x()+bar.get_width()/2, value+1.5, f"{value:.2f}%", ha="center", fontsize=13)
    ax.set(ylabel="Accuracy (%)", ylim=(0, 100), title="ViT-FER: contextual reference and current evaluation")
    ax.spines[["top", "right"]].set_visible(False)
    fig.text(.5, .04, CAVEAT, ha="center", fontsize=11)
    fig.text(.5, .01, f"OUR N = {metrics['n_images']} | PAPER N = 3500 | Delta is descriptive, not evidence of improvement.", ha="center", fontsize=9)
    fig.tight_layout(rect=(0, .08, 1, 1))
    return _save(fig, output, "paper_vs_current_accuracy.png")


def summary_figure(metrics, output):
    plt = _plotting()
    fig, ax = plt.subplots(figsize=(12.8, 7.2))
    ax.axis("off")
    fig.suptitle("Technical validation on AffectNetFiltered", fontsize=22, y=.95)
    blocks = [(0.04, "DATASET", f"AffectNetFiltered\nN = {metrics['n_images']}\n7 classes\nProvided validation split"),
              (0.37, "MODEL", "ViT-FER\nOff-the-shelf\nNo fine-tuning\nFrozen checkpoint"),
              (0.69, "OUR RESULT", f"Accuracy = {100*metrics['accuracy']:.2f}%\nMacro-F1 = {100*metrics['macro_f1']:.2f}%\nBalanced Acc. = {100*metrics['balanced_accuracy']:.2f}%\nUAR = {100*metrics['uar']:.2f}%")]
    for x, title, text in blocks:
        ax.text(x, .76, title, transform=ax.transAxes, fontsize=12, color="#52636b", weight="bold")
        ax.text(x, .65, text, transform=ax.transAxes, fontsize=17, va="top", linespacing=1.65)
    for x in (.31, .64):
        ax.annotate("", xy=(x+.035, .52), xytext=(x-.025, .52), xycoords="axes fraction", arrowprops={"arrowstyle": "->", "color": "#78858e", "lw": 1.8})
    ax.axhline(.25, xmin=.04, xmax=.96, color="#d8dde0", lw=1)
    ax.text(.04, .17, "PAPER REFERENCE", transform=ax.transAxes, fontsize=11, color="#52636b", weight="bold")
    ax.text(.04, .08, "Original AffectNet | ViT-FER Accuracy = 43.9% | N = 3500", transform=ax.transAxes, fontsize=15)
    fig.text(.5, .025, "Technical validation — different dataset preprocessing; not a direct replication.", ha="center", fontsize=11)
    return _save(fig, output, "experiment_summary_16x9.png")


def confidence_figure(rows, calibration, output):
    import numpy as np
    plt = _plotting()
    fig, ax = plt.subplots(figsize=(9, 5.8))
    for correct, color, label, key in ((True, "#647e8b", "Correct", "mean_confidence_correct"),
                                      (False, "#ad756c", "Incorrect", "mean_confidence_incorrect")):
        values = [r["confidence"] for r in rows if r["correct"] == correct]
        if values:
            ax.hist(values, bins=np.linspace(0, 1, 21), density=True, alpha=.5, color=color,
                    label=f"{label}: N={len(values)}, mean={calibration[key]:.3f}")
            ax.axvline(calibration[key], color=color, ls="--", lw=1.4)
    ax.set(xlabel="Maximum softmax confidence (uncalibrated)", ylabel="Density within each group", xlim=(0, 1),
           title="Confidence for correct and incorrect predictions")
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return _save(fig, output, "confidence_correct_vs_wrong.png")


def reliability_figure(calibration, output):
    plt = _plotting()
    fig, (ax, counts) = plt.subplots(2, 1, figsize=(8, 7.5), gridspec_kw={"height_ratios": [3, 1]}, sharex=True)
    bins = [b for b in calibration["bins"] if b["n"]]
    ax.plot([0, 1], [0, 1], ls="--", color="#a1a9af", label="Perfect agreement")
    ax.plot([b["mean_confidence"] for b in bins], [b["accuracy"] for b in bins], "o-", color="#647e8b", label="Observed top-label accuracy")
    ax.set(ylabel="Empirical accuracy", ylim=(0, 1), title=f"Reliability diagram | ECE = {100*calibration['ece']:.2f}%")
    ax.legend(frameon=False, fontsize=9)
    counts.bar([(b["lower"]+b["upper"])/2 for b in calibration["bins"]], [b["n"] for b in calibration["bins"]], width=.085, color="#9aaab3")
    counts.set(xlabel="Confidence (10 equal-width bins)", ylabel="N", xlim=(0, 1))
    fig.text(.5, .01, "No temperature scaling fitted. Empty bins have no accuracy estimate.", ha="center", fontsize=9)
    fig.tight_layout(rect=(0, .03, 1, 1))
    return _save(fig, output, "reliability_diagram.png")


def embedding_figure(rows, coordinates, metadata, output):
    import numpy as np
    plt = _plotting()
    fig, ax = plt.subplots(figsize=(10, 7))
    labels = np.asarray([r["true_label"] for r in rows])
    for c, color in zip(CLASSES, COLORS):
        mask = labels == c
        ax.scatter(coordinates[mask, 0], coordinates[mask, 1], s=10, alpha=.5, color=color, label=c, linewidths=0)
    method = metadata["method"]
    ax.set(title=f"Final CLS representation | {method} | N = {len(rows)}", xlabel=f"{method} dimension 1", ylabel=f"{method} dimension 2")
    if method == "PCA":
        ratios = metadata["explained_variance_ratio"]
        ax.set(xlabel=f"PC1 ({100*ratios[0]:.2f}% variance)", ylabel=f"PC2 ({100*ratios[1]:.2f}% variance)")
    ax.legend(frameon=False, fontsize=9, markerscale=2)
    ax.spines[["top", "right"]].set_visible(False)
    fig.text(.5, .015, "Colored by dataset label. A 2D projection illustrates learned representation, not psychological states.", ha="center", fontsize=9)
    fig.tight_layout(rect=(0, .04, 1, 1))
    return _save(fig, output, "vit_embedding_projection.png")


def attention_figure(selected, maps, output):
    import numpy as np
    from PIL import Image
    plt = _plotting()
    valid = [r for r in selected if r["sample_id"] in maps]
    if not valid:
        raise ValueError("No verified attention maps exist.")
    fig, axes = plt.subplots(len(valid), 2, figsize=(9, 3.6*len(valid)), squeeze=False)
    for i, row in enumerate(valid):
        _photo(axes[i, 0], row)
        _photo(axes[i, 1], row)
        with Image.open(row["filepath"]) as im:
            size = im.size
        grid = maps[row["sample_id"]]
        normalized = (grid-grid.min())/(grid.max()-grid.min()) if grid.max() > grid.min() else np.zeros_like(grid)
        resized = np.asarray(Image.fromarray(normalized).resize(size, resample=Image.Resampling.BILINEAR))
        axes[i, 1].imshow(resized, cmap="magma", vmin=0, vmax=1, alpha=.45)
        axes[i, 0].set_title(f"ORIGINAL | TRUE: {row['true_label']}", fontsize=11)
        axes[i, 1].set_title(f"ATTENTION | PRED: {row['predicted_label']} | p={row['confidence']:.3f}", fontsize=11)
    fig.suptitle("Residual attention rollout: 2 correct and 2 incorrect examples when available", fontsize=13, y=.995)
    fig.text(.5, .01, "Attention visualization is interpretative, not causal. Maps normalized independently; alpha = 0.45.", ha="center", fontsize=9)
    fig.tight_layout(rect=(0, .03, 1, .98))
    return _save(fig, output, "vit_attention_examples.png")


def _gif_frame(fig):
    import numpy as np
    from PIL import Image
    fig.canvas.draw()
    rgb = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
    return Image.fromarray(rgb).convert("P", palette=Image.Palette.ADAPTIVE, colors=256)


def inference_gif(examples, output):
    import numpy as np
    plt = _plotting()
    if not examples:
        raise ValueError("No real examples selected.")
    frames = []
    for i, row in enumerate(examples):
        fig = plt.figure(figsize=(12.8, 7.2), dpi=100)
        ax_image = fig.add_axes([.04, .20, .39, .62])
        _photo(ax_image, row)
        ax = fig.add_axes([.58, .20, .36, .62])
        probabilities = [row[f"p_{c}"] for c in CLASSES]
        colors = ["#647e8b" if c == row["predicted_label"] else "#cbd3d7" for c in CLASSES]
        ax.barh(CLASSES, probabilities, color=colors, height=.65)
        ax.invert_yaxis()
        ax.set(xlim=(0, 1.08), xlabel="Softmax probability", title="Seven-class model output")
        for j, p in enumerate(probabilities):
            ax.text(min(p+.015, 1.01), j, f"{p:.3f}", va="center", fontsize=11)
        ax.spines[["top", "right"]].set_visible(False)
        result = "CORRECT" if row["correct"] else "ERROR"
        fig.suptitle(f"ViT-FER inference | Example {i+1}/{len(examples)} | {result}", fontsize=20, y=.96)
        fig.text(.5, .10, f"TRUE: {row['true_label']}     PRED: {row['predicted_label']}     confidence = {row['confidence']:.3f}", ha="center", fontsize=16)
        fig.text(.5, .035, "AffectNetFiltered | Off-the-shelf, no fine-tuning | Confidence is uncalibrated.", ha="center", fontsize=11)
        frames.append(_gif_frame(fig)); plt.close(fig)
    path = output / "figures" / "vit_inference_demo.gif"
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=3500, loop=0, optimize=False, disposal=2)
    write_json(output / "figures" / "vit_inference_demo.json", {"sample_ids": [r["sample_id"] for r in examples],
        "n_correct": sum(r["correct"] for r in examples), "n_wrong": sum(not r["correct"] for r in examples),
        "frame_duration_ms": 3500, "dimensions": [1280, 720], "probabilities_source": "predictions/predictions_vit.csv",
        "attention_panel": "not included; priority 2 GIF precedes optional attention extraction"})
    return path


def confusion_gif(rows, metrics, output):
    import numpy as np
    plt = _plotting()
    fig, ax = plt.subplots(figsize=(9, 7), dpi=100)
    matrix = np.zeros((7, 7), dtype=int)
    image = ax.imshow(matrix, cmap="Blues", vmin=0, vmax=max(np.asarray(metrics["confusion_matrix"]).max(), 1))
    ax.set(xticks=range(7), yticks=range(7), xticklabels=CLASSES, yticklabels=CLASSES, xlabel="Predicted label", ylabel="True label")
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    texts = [[ax.text(j, i, "0", ha="center", va="center", fontsize=9) for j in range(7)] for i in range(7)]
    fig.tight_layout()
    frames, previous = [], 0
    for end in np.unique(np.linspace(0, len(rows), 13).astype(int)):
        for row in rows[previous:int(end)]:
            matrix[CLASSES.index(row["true_label"]), CLASSES.index(row["predicted_label"])] += 1
        image.set_data(matrix)
        for i in range(7):
            for j in range(7):
                texts[i][j].set_text(str(matrix[i, j]))
                texts[i][j].set_color("white" if matrix[i,j] > image.norm.vmax*.5 else "black")
        ax.set_title(f"Accumulating real predictions: {int(end)} / {len(rows)}")
        frames.append(_gif_frame(fig)); previous = int(end)
    if not np.array_equal(matrix, np.asarray(metrics["confusion_matrix"])):
        raise ValueError("Animated confusion matrix does not match final metrics.")
    path = output / "figures" / "confusion_matrix_build.gif"
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=[350]*(len(frames)-1)+[1800], loop=0, optimize=False)
    plt.close(fig)
    return path


def core_figure_tasks(rows, metrics, panels, output):
    return [
        ("dataset_examples.png", lambda: dataset_examples(panels, len(rows), output)),
        ("confusion_matrix_vit_normalized.png", lambda: plot_confusion(metrics, output/"figures/confusion_matrix_vit_normalized.png", True)),
        ("f1_per_class_vit.png", lambda: class_figures(metrics, output, "f1")),
        ("prediction_examples.png", lambda: prediction_examples(panels, output)),
        ("confusion_matrix_vit.png", lambda: plot_confusion(metrics, output/"figures/confusion_matrix_vit.png", False)),
        ("class_distribution.png", lambda: class_figures(metrics, output, "distribution")),
        ("paper_vs_current_accuracy.png", lambda: paper_comparison(metrics, output)),
        ("experiment_summary_16x9.png", lambda: summary_figure(metrics, output)),
        ("top_confusions.png", lambda: top_confusions(panels, output)),
        ("vit_inference_demo.gif", lambda: inference_gif(panels["gif"], output)),
    ]
