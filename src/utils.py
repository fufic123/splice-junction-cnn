import os
import random
import csv
import numpy as np


def set_global_seed(seed: int = 42) -> None:
    """Fix random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
        # Force deterministic ops where possible
        os.environ["TF_DETERMINISTIC_OPS"] = "1"
    except ImportError:
        pass


def append_result_row(path: str, row: dict) -> None:
    """Append a single result dict as a row to a CSV file."""
    file_exists = os.path.isfile(path)
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path
