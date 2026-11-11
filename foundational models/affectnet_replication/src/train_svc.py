from __future__ import annotations

from config import C_GRID, CLASSES, SEED
from src.metrics import save_evaluation
from src.utils import write_csv, write_json

SVC_SPECS = (("SVC Linear", "svc_linear", "linear", None),
             ("SVC Linear weighted", "svc_linear_weighted", "linear", "balanced"),
             ("SVC RBF", "svc_rbf", "rbf", None),
             ("SVC RBF weighted", "svc_rbf_weighted", "rbf", "balanced"))


def internal_cv_splits(labels, groups, requested_folds=5):
    """No final validation rows are accepted by the hyperparameter search."""
    import numpy as np
    from sklearn.model_selection import StratifiedGroupKFold
    y = np.asarray(labels)
    groups = np.asarray(groups)
    if set(y) != set(CLASSES):
        raise ValueError("SVC training requires observed examples of all seven classes.")
    n_folds = min(requested_folds, min(len(set(groups[y == c])) for c in CLASSES))
    if n_folds < 2:
        raise ValueError("Need at least two distinct training-image hashes per class for internal CV.")
    cv = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=SEED)
    splits = list(cv.split(np.zeros(len(y)), y, groups))
    for fitting, tuning in splits:
        if set(groups[fitting]) & set(groups[tuning]):
            raise ValueError("Internal CV image-content leakage.")
        if set(y[fitting]) != set(CLASSES) or set(y[tuning]) != set(CLASSES):
            raise ValueError("Grouped CV cannot preserve all seven classes in each fold for these data.")
    return splits


def train_svcs(output, feature_columns, cv_folds=5, n_jobs=1, scale=False):
    import numpy as np
    import pandas as pd
    from sklearn.model_selection import GridSearchCV
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC
    from joblib import dump
    frame = pd.read_csv(output / "openface_features.csv")
    train = frame[frame["split"] == "train"].copy()
    final = frame[frame["split"] == "validation"].copy()
    if train.empty or final.empty or not feature_columns:
        raise ValueError("SVC requires nonempty OpenFace train/validation and real AU columns.")
    if set(train["sha256"]) & set(final["sha256"]):
        raise ValueError("Train/validation leakage detected in feature hashes.")
    x_train = train[feature_columns].to_numpy(dtype=float)
    x_final = final[feature_columns].to_numpy(dtype=float)
    if not np.isfinite(x_train).all() or not np.isfinite(x_final).all():
        raise ValueError("Nonfinite features are never imputed as valid detections.")
    splits = internal_cv_splits(train["true_label"].to_numpy(), train["sha256"].to_numpy(), cv_folds)
    fold_rows = []
    for fold, (fitting, tuning) in enumerate(splits):
        for role, indexes in (("fit", fitting), ("tune", tuning)):
            for index in indexes:
                fold_rows.append({"fold": fold, "role": role, "filepath": train.iloc[index]["filepath"]})
    write_csv(output / "metrics" / "internal_cv_assignments.csv", fold_rows, ["fold", "role", "filepath"])
    results = {}
    (output / "models").mkdir(exist_ok=True)
    for name, slug, kernel, weight in SVC_SPECS:
        try:
            pipeline = Pipeline([("scale", StandardScaler() if scale else "passthrough"),
                                 ("svc", SVC(kernel=kernel, class_weight=weight,
                                             gamma="scale", probability=False, random_state=SEED))])
            search = GridSearchCV(pipeline, {"svc__C": list(C_GRID)}, scoring="accuracy", cv=splits,
                                  refit=True, n_jobs=n_jobs, error_score="raise", return_train_score=False)
            print(f"Fitting {name}: C={list(C_GRID)}, {len(splits)} internal folds, train={len(train)}", flush=True)
            search.fit(x_train, train["true_label"].to_numpy())
            # The final partition is first touched for prediction after C selection/refit.
            predicted = search.predict(x_final)
            rows = [{"filepath": path, "true_label": true, "predicted_label": pred, "confidence": None}
                    for path, true, pred in zip(final["filepath"], final["true_label"], predicted)]
            metadata = {"best_C": float(search.best_params_["svc__C"]), "kernel": kernel,
                        "class_weight": weight, "gamma": "scale", "scaling": "StandardScaler fitted within each fold" if scale else "none",
                        "selection_metric": "accuracy", "selection_partition": "train only; StratifiedGroupKFold by image SHA256",
                        "internal_cv_folds": len(splits), "C_grid": list(C_GRID),
                        "internal_cv_best_accuracy": float(search.best_score_),
                        "evaluation_population": "validation images successfully processed by OpenFace",
                        "n_training": len(train), "feature_columns": feature_columns,
                        "confidence_note": "SVC probabilities disabled; confidence is unavailable, never a fabricated probability"}
            metrics = save_evaluation(rows, output, slug, metadata)
            cv_rows = [{"C": float(p["svc__C"]), "mean_internal_accuracy": float(mean),
                        "std_internal_accuracy": float(std), "rank": int(rank)}
                       for p, mean, std, rank in zip(search.cv_results_["params"],
                            search.cv_results_["mean_test_score"], search.cv_results_["std_test_score"],
                            search.cv_results_["rank_test_score"])]
            write_json(output / "metrics" / f"cv_{slug}.json", cv_rows)
            dump(search.best_estimator_, output / "models" / f"{slug}.joblib")
            results[name] = {"status": "completed", "metrics": metrics}
        except Exception as exc:
            results[name] = {"status": "failed", "reason": f"{type(exc).__name__}: {exc}", "metrics": None}
    return results
