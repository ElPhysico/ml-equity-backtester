# src/mlbt/io.py
from pathlib import Path
import json
import yaml
import pandas as pd
from typing import Dict, Any, Optional, Union

from mlbt.utils import find_project_root

def resolve_output_dir(
    universe_id: str,
    start_month: str,  # "YYYY-MM"
    end_month: str,    # "YYYY-MM"
    config_hash: str,
    base_out_dir: Optional[Path] = None
) -> Path:
    """
    Build a deterministic path where the panel artifacts live.
    Layout: outputs/feature_panel/<universe_id>/start=YYYY-MM__end=YYYY-MM/<config_hash>/
    """
    root = base_out_dir or (find_project_root() / "outputs" / "feature_panel")
    out = (root / universe_id / f"start={start_month}_end={end_month}" / config_hash)
    out.mkdir(parents=True, exist_ok=True)
    return out

def save_panel(
    features: pd.DataFrame,
    meta: Dict[str, Any],
    universe_id: str,
    start_month: str,
    end_month: str,
    config_hash: str,
    base_out_dir: Optional[Path] = None
) -> Path:
    """
    Write features.parquet and meta.json to the resolved directory.
    Returns the directory path used.
    """
    out_dir = resolve_output_dir(universe_id, start_month, end_month, config_hash, base_out_dir)
    # Parquet
    (out_dir / "features.parquet").unlink(missing_ok=True)
    features.to_parquet(out_dir / "features.parquet")
    # Meta
    with open(out_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    return out_dir

def maybe_load_panel(
    universe_id: str,
    start_month: str,
    end_month: str,
    config_hash: str,
    base_out_dir: Optional[Path] = None
) -> Optional[pd.DataFrame]:
    """
    If a panel with this signature exists, load and return it; else return None.
    """
    out_dir = resolve_output_dir(universe_id, start_month, end_month, config_hash, base_out_dir)
    feats_path = out_dir / "features.parquet"
    if feats_path.exists():
        return pd.read_parquet(feats_path)
    return None


def read_yaml(path: Union[str, Path]) -> Any:
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)