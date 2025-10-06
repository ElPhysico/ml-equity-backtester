# src/mlbt/io.py
"""
Input / Output utilities.

This module centralizes all read / write helpers for the project.

Sections
--------
1. Feature panels
   - resolve_features_output_dir
   - save_panel
   - maybe_load_panel
2. Label panels
   - resolve_labels_output_dir
   - save_labels
   - maybe_load_labels
3. YAML configs
   - read_yaml
   - write_yaml

Design principles
-----------------
- All save_*() functions create parent directories as needed.
- All maybe_load_*() functions are read-only and never create paths.
- Paths are resolved relative to the project root (via utils.find_project_root)
  unless a custom base_out_dir is provided.
- File formats:
  * Tabular data → Parquet (.parquet)
  * Metadata → JSON (.json)
  * Configs → YAML (.yaml)
"""
from pathlib import Path
import json
import yaml
import pandas as pd
from typing import Dict, Any, Optional, Union

from mlbt.utils import find_project_root


# features io
def resolve_features_output_dir(
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
    out = (root / universe_id / f"start={start_month}__end={end_month}" / config_hash)
    # out.mkdir(parents=True, exist_ok=True)
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
    out_dir = resolve_features_output_dir(universe_id, start_month, end_month, config_hash, base_out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
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
    out_dir = resolve_features_output_dir(universe_id, start_month, end_month, config_hash, base_out_dir)
    feats_path = out_dir / "features.parquet"
    if feats_path.exists():
        return pd.read_parquet(feats_path)
    return None


# labels io
def resolve_labels_output_dir(
    universe_id: str,
    start_month: str,  # "YYYY-MM"
    end_month: str,    # "YYYY-MM"
    config_hash: str,
    base_out_dir: Optional[Path] = None
) -> Path:
    """
    Build a deterministic path where the label artifacts live.
    Layout: outputs/labels/<universe_id>/start=YYYY-MM__end=YYYY-MM/<config_hash>/
    """
    root = base_out_dir or (find_project_root() / "outputs" / "labels")
    out = (root / universe_id / f"start={start_month}__end={end_month}" / config_hash)
    # out.mkdir(parents=True, exist_ok=True)
    return out

def save_labels(
    labels: pd.DataFrame,
    meta: Dict[str, Any],
    universe_id: str,
    start_month: str,
    end_month: str,
    config_hash: str,
    base_out_dir: Optional[Path] = None
) -> Path:
    """
    Write labels.parquet and meta.json to the resolved directory.
    Returns the directory path used.
    """
    out_dir = resolve_labels_output_dir(universe_id, start_month, end_month, config_hash, base_out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    # Parquet
    (out_dir / "labels.parquet").unlink(missing_ok=True)
    labels.to_parquet(out_dir / "labels.parquet")
    # Meta
    with open(out_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    return out_dir

def maybe_load_labels(
    universe_id: str,
    start_month: str,
    end_month: str,
    config_hash: str,
    base_out_dir: Optional[Path] = None
) -> Optional[pd.DataFrame]:
    """
    If labels with this signature exist, load and return them; else return None.
    """
    out_dir = resolve_labels_output_dir(universe_id, start_month, end_month, config_hash, base_out_dir)
    labels_path = out_dir / "labels.parquet"
    if labels_path.exists():
        return pd.read_parquet(labels_path)
    return None


# yaml io
def read_yaml(path: Union[str, Path]) -> Any:
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)
    
def write_yaml(path: Union[str, Path], data: Any) -> None:
    p = Path(path)
    with p.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)
