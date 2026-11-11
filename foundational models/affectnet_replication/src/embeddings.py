from __future__ import annotations

import importlib.util
from config import SEED
from src.utils import write_csv, write_json


def project_embeddings(rows, output):
    import numpy as np
    from sklearn.decomposition import PCA
    embeddings = np.load(output / "embeddings" / "vit_cls_embeddings.npy", allow_pickle=False)
    if len(embeddings) != len(rows) or not np.isfinite(embeddings).all():
        raise ValueError("Invalid CLS embeddings.")
    info = {"n_images": len(rows), "embedding_dimension": embeddings.shape[1],
            "supervised": False, "seed": SEED, "input_scaling": "none"}
    coordinates = None
    if importlib.util.find_spec("umap") is not None:
        try:
            from umap import UMAP
            params = dict(n_components=2, n_neighbors=15, min_dist=0.1, metric="euclidean", random_state=SEED)
            coordinates = UMAP(**params).fit_transform(embeddings)
            info.update(method="UMAP", parameters=params)
        except Exception as exc:
            info["umap_error"] = f"{type(exc).__name__}: {exc}"
    if coordinates is None:
        reducer = PCA(n_components=2, svd_solver="full", whiten=False, random_state=SEED)
        coordinates = reducer.fit_transform(embeddings)
        info.update(method="PCA", parameters={"n_components": 2, "svd_solver": "full", "whiten": False},
                    explained_variance_ratio=reducer.explained_variance_ratio_.tolist(),
                    fallback_reason=info.get("umap_error", "UMAP is not installed in the frozen environment"))
    projection_rows = [{"sample_id": row["sample_id"], "prediction_index": i, "true_label": row["true_label"],
                        "x": float(coordinates[i, 0]), "y": float(coordinates[i, 1])} for i, row in enumerate(rows)]
    write_csv(output / "embeddings" / "projection_coordinates.csv", projection_rows,
              ["sample_id", "prediction_index", "true_label", "x", "y"])
    write_json(output / "embeddings" / "projection_metadata.json", info)
    return coordinates, info
