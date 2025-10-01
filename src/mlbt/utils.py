# src/mlbt/utils.py
import hashlib
from pathlib import Path
from datetime import datetime, UTC
import os

PROJECT_SENTINELS = ("pyproject.toml", ".git")

def sha1_of_str(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()

def utc_now_iso() -> str:
    # ISO 8601 with 'Z' suffix
    return datetime.now(UTC).replace(microsecond=0).isoformat() + "Z"

def find_project_root(start: Path | None = None) -> Path:
    """
    Resolve the project root by:
      1) MLBT_ROOT env var, if set
      2) walking upward from `start` (or this file) until a sentinel is found
      3) fallback to current working directory
    """
    env = os.environ.get("MLBT_ROOT")
    if env:
        return Path(env).resolve()

    p = (start or Path(__file__)).resolve()
    for parent in [p] + list(p.parents):
        for s in PROJECT_SENTINELS:
            if (parent / s).exists():
                return parent
    return Path.cwd().resolve()
