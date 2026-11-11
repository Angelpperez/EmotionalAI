"""Deterministic, logged example selection; no judgements about people."""
from __future__ import annotations

import shutil
from collections import defaultdict
from pathlib import Path

from config import CLASSES, SEED
from src.utils import write_csv, write_json


def confusion_rankings(metrics):
    counts, rates = metrics["confusion_matrix"], metrics["confusion_matrix_normalized_true"]
    pairs = [{"true_label": CLASSES[i], "predicted_label": CLASSES[j], "n": counts[i][j],
              "rate": rates[i][j], "support": sum(counts[i]), "true_index": i, "predicted_index": j}
             for i in range(7) for j in range(7) if i != j and counts[i][j] > 0]
    by_count = sorted(pairs, key=lambda p: (-p["n"], p["true_index"], p["predicted_index"]))
    by_rate = sorted(pairs, key=lambda p: (-p["rate"], p["true_index"], p["predicted_index"]))
    return by_count, by_rate


def balanced_examples(rows, n, seed=SEED):
    import numpy as np
    rng = np.random.default_rng(seed)
    groups = defaultdict(list)
    for row in sorted(rows, key=lambda r: r["sample_id"]):
        groups[row["true_label"], row["predicted_label"]].append(row)
    for key in groups:
        rng.shuffle(groups[key])
    result = []
    while len(result) < n:
        made_progress = False
        for key in sorted(groups, key=lambda k: (CLASSES.index(k[0]), CLASSES.index(k[1]))):
            if groups[key] and len(result) < n:
                result.append(groups[key].pop())
                made_progress = True
        if not made_progress:
            break
    return result


def analyze_errors(rows, metrics, output):
    import numpy as np
    correct = [r for r in rows if r["correct"]]
    incorrect = [r for r in rows if not r["correct"]]
    high = sorted(correct, key=lambda r: (-r["confidence"], r["sample_id"]))[:5]
    used = {r["sample_id"] for r in high}
    low = sorted((r for r in correct if r["sample_id"] not in used), key=lambda r: (r["confidence"], r["sample_id"]))[:5]
    wrong_high = sorted(incorrect, key=lambda r: (-r["confidence"], r["sample_id"]))[:5]
    by_count, by_rate = confusion_rankings(metrics)
    rng = np.random.default_rng(SEED)
    datasets = {}
    for c in CLASSES:
        eligible = sorted((r for r in rows if r["true_label"] == c), key=lambda r: r["sample_id"])
        indexes = rng.permutation(len(eligible))[:4]
        datasets[c] = [eligible[int(i)] for i in indexes]
    confusion_examples = []
    for pair in by_rate[:3]:
        eligible = sorted((r for r in incorrect if r["true_label"] == pair["true_label"]
                           and r["predicted_label"] == pair["predicted_label"]), key=lambda r: r["sample_id"])
        chosen = np.random.default_rng(SEED).permutation(len(eligible))[:3]
        confusion_examples.append({"pair": pair, "rows": [eligible[int(i)] for i in chosen]})
    selected_errors = balanced_examples(incorrect, 10)
    attention = balanced_examples(correct, 2) + balanced_examples(incorrect, 2)
    demo = balanced_examples(correct, 3) + balanced_examples(incorrect, 2)
    panels = {"dataset": datasets, "correct_high": high, "correct_low": low, "incorrect_high": wrong_high,
              "confusions": confusion_examples, "attention": attention, "gif": demo}
    f1_order = sorted(CLASSES, key=lambda c: (metrics["per_class"][c]["f1"], CLASSES.index(c)))
    best_recall = max(metrics["per_class"][c]["recall"] for c in CLASSES)
    summary = {"n_correct": len(correct), "n_incorrect": len(incorrect), "worst_two_f1": f1_order[:2],
               "best_f1": [c for c in CLASSES if metrics["per_class"][c]["f1"] == max(metrics["per_class"][k]["f1"] for k in CLASSES)],
               "best_recall": [c for c in CLASSES if metrics["per_class"][c]["recall"] == best_recall],
               "top_two_confusions_by_count": by_count[:2], "top_three_confusions_by_rate": by_rate[:3],
               "selection_seed": SEED, "note": "One multiclass error is FP for predicted class and FN for true class; collections can overlap."}
    selections = []

    def register(category, examples, rule):
        for index, row in enumerate(examples):
            selections.append({"category": category, "order": index, "sample_id": row["sample_id"],
                               "filepath": row["filepath"], "true_label": row["true_label"],
                               "predicted_label": row["predicted_label"], "confidence": row["confidence"],
                               "selection_rule": rule})

    for c, samples in datasets.items():
        register(f"dataset_{c}", samples, "seed 42 random sample per true class, sorted IDs")
    for category, samples in (("correct_high", high), ("correct_low", low), ("incorrect_high", wrong_high)):
        register(category, samples, "confidence rank; ties by sample ID; high/low correct samples disjoint")
    for i, entry in enumerate(confusion_examples):
        register(f"top_confusion_{i}", entry["rows"], "seed 42 random sample within the directed confusion")
    register("attention", attention, "balanced directed label pairs, seed 42; up to 2 correct and 2 wrong")
    register("gif", demo, "balanced directed label pairs, seed 42; up to 3 correct and 2 wrong")
    for category, samples in (("false_positives", selected_errors), ("false_negatives", selected_errors),
                              ("correct_high_confidence", high)):
        destination = output / "error_analysis" / category
        destination.mkdir(parents=True, exist_ok=True)
        for row in samples:
            source = Path(row["filepath"])
            shutil.copy2(source, destination / f"{row['sample_id']}{source.suffix.lower()}")
        register(category, samples, "balanced directed errors or confidence rank; original bytes copied")
    write_json(output / "error_analysis" / "summary.json", summary)
    write_json(output / "error_analysis" / "panels.json", panels)
    write_csv(output / "error_analysis" / "selected_examples.csv", selections,
              ["category", "order", "sample_id", "filepath", "true_label", "predicted_label", "confidence", "selection_rule"])
    return panels, summary
