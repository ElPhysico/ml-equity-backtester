# src/mlbt/yaml_io.py
import yaml
from pathlib import Path
from typing import Any, Union


def read_yaml(path: Union[str, Path]) -> Any:
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)
