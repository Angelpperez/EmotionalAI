from __future__ import annotations

import json
import os
import shlex
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]


def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def resolve_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else (REPO_ROOT / path).resolve()


def command_text(command: Sequence[str | Path]) -> str:
    return subprocess.list2cmdline([str(item) for item in command])


def run(
    command: Sequence[str | Path],
    *,
    dry_run: bool,
    cwd: Path = REPO_ROOT,
    env: dict[str, str] | None = None,
) -> None:
    printable = command_text(command)
    print(f"$ {printable}")
    if dry_run:
        return
    subprocess.run(
        [str(item) for item in command],
        cwd=cwd,
        env=env,
        check=True,
    )


def require(path: Path, label: str) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"{label}: {path}")
    return path


def write_manifest(
    output_dir: Path,
    *,
    experiment: str,
    arguments: dict[str, object],
    commands: list[list[str]],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "experiment": experiment,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "repository": str(REPO_ROOT),
        "python": sys.version,
        "arguments": arguments,
        "commands": commands,
    }
    (output_dir / "run_manifest.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def existing_or_default(candidates: Sequence[str | Path], fallback: str) -> Path:
    for candidate in candidates:
        path = resolve_path(candidate)
        if path.exists():
            return path
    return Path(fallback)


def python_executable(preferred: str | Path | None = None) -> Path:
    if preferred:
        return require(resolve_path(preferred), "Python")
    return Path(sys.executable)


def child_env(temp_dir: Path | None = None) -> dict[str, str]:
    env = os.environ.copy()
    if temp_dir is not None:
        temp_dir.mkdir(parents=True, exist_ok=True)
        env["TEMP"] = str(temp_dir)
        env["TMP"] = str(temp_dir)
    return env
