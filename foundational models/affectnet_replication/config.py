"""Shared protocol. Canonical IDs below are ours, never inferred dataset IDs."""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
REPOSITORY_ROOT = PROJECT_ROOT.parent.parent
CLASSES = ("neutral", "fear", "anger", "happiness", "sadness", "disgust", "surprise")
SEED = 42
C_GRID = (0.01, 0.1, 1, 10, 100)
MODEL_ID = "trpakov/vit-face-expression"
PAPER_URL = "https://arxiv.org/html/2409.08907v1"
PAPER_ACCURACY_PERCENT = {
    "SVC Linear": 27.3, "SVC Linear weighted": 28.7,
    "SVC RBF": 28.5, "SVC RBF weighted": 29.7, "ViT-FER": 43.9,
}
PAPER_TRAIN_COUNTS = dict(zip(CLASSES, (74873, 6378, 24881, 134411, 25458, 3803, 14090)))
PAPER_VALIDATION_PER_CLASS = 500
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
LABEL_ALIASES = {name: name for name in CLASSES}
LABEL_ALIASES.update({"angry": "anger", "happy": "happiness", "sad": "sadness"})
SPLIT_ALIASES = {"train": "train", "training": "train", "val": "validation",
                 "valid": "validation", "validation": "validation"}
