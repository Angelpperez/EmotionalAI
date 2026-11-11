from __future__ import annotations

import csv
import hashlib
import importlib.metadata
import json
import os
import platform
import random
import shutil
import sys
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from config import CLASSES, LABEL_ALIASES, SEED


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write_csv(path, rows, fields):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def markdown_local_link(label, path, base):
    """Portable Markdown links: Windows drive paths otherwise look like URI schemes."""
    from urllib.parse import quote
    relative = Path(path).resolve().relative_to(Path(base).resolve()).as_posix()
    return f"[{label}]({quote(relative, safe='/')})"


def normalize_label(value, label_map=None):
    raw = str(value).strip()
    if label_map is not None:
        if raw not in label_map:
            raise ValueError(f"Label {raw!r} absent from the explicit label map.")
        mapped = label_map[raw]
        if mapped is None:
            return None  # Deliberately excluded class, recorded in the inspection report.
        raw = str(mapped).strip()
    label = LABEL_ALIASES.get(raw.lower())
    if label not in CLASSES:
        raise ValueError(f"Unverified label {raw!r}; supply --label-map. Numeric IDs are never guessed.")
    return label


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fingerprint_rows(rows):
    return hashlib.sha256(json.dumps(rows, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()


def seed_everything(seed=SEED):
    random.seed(seed)
    import numpy as np
    np.random.seed(seed)
    try:
        import torch
    except ImportError:
        return
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)


def environment_metadata():
    packages = {}
    for name in ("torch", "torchvision", "transformers", "scikit-learn", "pandas", "numpy", "matplotlib", "pillow", "tqdm", "umap-learn", "huggingface-hub", "safetensors"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    result = {"python_version": sys.version, "python_executable": sys.executable,
              "python_311": sys.version_info[:2] == (3, 11), "platform": platform.platform(),
              "packages": packages, "cuda_available": None, "gpu": None,
              "seed": SEED, "captured_at_utc": utc_now()}
    result.update(cpu=platform.processor(), cpu_logical_cores=os.cpu_count())
    if sys.platform == "win32":
        import ctypes
        import winreg
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0") as key:
            result["cpu"] = winreg.QueryValueEx(key, "ProcessorNameString")[0].strip()
        class MemoryStatus(ctypes.Structure):
            _fields_ = [("length", ctypes.c_ulong), ("load", ctypes.c_ulong),
                        ("total_physical", ctypes.c_ulonglong), ("available_physical", ctypes.c_ulonglong),
                        ("total_pagefile", ctypes.c_ulonglong), ("available_pagefile", ctypes.c_ulonglong),
                        ("total_virtual", ctypes.c_ulonglong), ("available_virtual", ctypes.c_ulonglong),
                        ("available_extended", ctypes.c_ulonglong)]
        memory = MemoryStatus()
        memory.length = ctypes.sizeof(memory)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(memory)):
            result["ram_bytes"] = memory.total_physical
    try:
        gpu_info = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader,nounits"],
                                  capture_output=True, text=True, timeout=10, check=False)
        result["nvidia_smi"] = gpu_info.stdout.strip()
    except (OSError, subprocess.TimeoutExpired):
        result["nvidia_smi"] = None
    try:
        import torch
        result.update(cuda_available=torch.cuda.is_available(), cuda_version=torch.version.cuda,
                      gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
        result["torch_flags"] = {"deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
                                 "cudnn_deterministic": torch.backends.cudnn.deterministic,
                                 "cudnn_benchmark": torch.backends.cudnn.benchmark,
                                 "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
                                 "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
                                 "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG")}
    except (ImportError, OSError, RuntimeError) as exc:
        result["torch_runtime_error"] = str(exc)
    return result


def freeze_environment(destination):
    result = subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True, check=True)
    Path(destination).write_text(result.stdout, encoding="utf-8")


def source_snapshot(root):
    root = Path(root)
    paths = [root / "config.py", root / "run_experiment.py", root / "requirements.txt"]
    paths += sorted((root / "src").glob("*.py")) + sorted((root / "configs").glob("*.json"))
    paths += sorted((root / "scripts").glob("*.py")) + sorted((root / "scripts").glob("*.ps1"))
    return {str(p.relative_to(root)): sha256_file(p) for p in paths}


def archive_sources(root, output, hashes, subdirectory="source_code"):
    root, output = Path(root), Path(output)
    for relative, digest in hashes.items():
        source = root / relative
        if sha256_file(source) != digest:
            raise ValueError(f"Source file changed after its hash was recorded: {relative}")
        destination = output / subdirectory / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists() and sha256_file(destination) != digest:
            raise ValueError(f"Refusing to overwrite a frozen source snapshot: {destination}")
        shutil.copy2(source, destination)
    return str(output / subdirectory)


def artifact_manifest(output):
    root = Path(output)
    entries = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.name != "artifact_manifest.json":
            entries.append({"path": str(path.resolve()), "relative_path": path.relative_to(root).as_posix(),
                            "n_bytes": path.stat().st_size, "sha256": sha256_file(path),
                            "type": path.suffix.lower(), "source_run": root.name})
    write_json(root / "artifact_manifest.json", {"generated_at_utc": utc_now(), "artifacts": entries})
    return entries


def walk_files(root):
    """Include git-ignored data but exclude virtual environments, links, and build trees."""
    skipped = {".git", ".vs", ".venv", "env", "env312", "__pycache__", "node_modules",
               "site-packages", "build"}
    for base, dirs, files in os.walk(root, followlinks=False):
        dirs[:] = sorted(d for d in dirs if d.lower() not in skipped
                         and not (Path(base) / d / "pyvenv.cfg").is_file()
                         and not (Path(base) / d).is_symlink())
        for name in sorted(files):
            yield Path(base) / name
