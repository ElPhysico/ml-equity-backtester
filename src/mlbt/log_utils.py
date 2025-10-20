# src/mlbt/log_utils.py
import logging


def setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.WARNING
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%d-%m-%Y %H:%M:%S",
        force=True
    )
    logging.getLogger('matplotlib').setLevel(logging.WARNING)