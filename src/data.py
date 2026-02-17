import os
import re
import requests
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


DATA_DIR = os.path.join(os.path.dirname(__file__), os.pardir, "data")
RAW_FILE = "splice.data"
DATA_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/molecular-biology/splice-junction-gene-sequences/splice.data"

NUCLEOTIDE_MAP = {"A": 0, "C": 1, "G": 2, "T": 3}
LABEL_MAP = {"EI": 0, "IE": 1, "N": 2}  # exon-intron, intron-exon, neither
LABEL_NAMES = ["EI", "IE", "N"]
NUM_CLASSES = 3


def download_data(dest_dir: str | None = None) -> str:
    """Download splice.data from UCI if not already present. Return file path."""
    if dest_dir is None:
        dest_dir = DATA_DIR
    os.makedirs(dest_dir, exist_ok=True)
    filepath = os.path.join(dest_dir, RAW_FILE)
    if os.path.isfile(filepath):
        print(f"[data] File already exists: {filepath}")
        return filepath
    print(f"[data] Downloading from {DATA_URL} ...")
    resp = requests.get(DATA_URL, timeout=60)
    resp.raise_for_status()
    with open(filepath, "wb") as f:
        f.write(resp.content)
    print(f"[data] Saved to {filepath} ({len(resp.content)} bytes)")
    return filepath


def parse_splice_file(filepath: str) -> pd.DataFrame:
    """Parse splice.data into a DataFrame with columns: label, id, sequence."""
    rows = []
    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(",")
            if len(parts) < 3:
                continue
            label = parts[0].strip()
            seq_id = parts[1].strip()
            sequence = parts[2].strip().upper().replace(" ", "")
            rows.append({"label": label, "id": seq_id, "sequence": sequence})
    df = pd.DataFrame(rows)
    return df


def clean_sequences(df: pd.DataFrame) -> pd.DataFrame:
    """Remove rows with invalid chars; replace ambiguous nucleotides with random ACGT."""
    # Keep only rows whose labels are in LABEL_MAP
    df = df[df["label"].isin(LABEL_MAP)].copy()

    def clean_seq(seq: str) -> str:
        # Replace any character that is not ACGT with a random nucleotide
        def _replace(m):
            return np.random.choice(list("ACGT"))
        return re.sub(r"[^ACGT]", _replace, seq)

    df["sequence"] = df["sequence"].apply(clean_seq)
    return df.reset_index(drop=True)


def one_hot_encode(sequence: str) -> np.ndarray:
    """Encode a DNA sequence as (L, 4) float32 array."""
    arr = np.zeros((len(sequence), 4), dtype=np.float32)
    for i, ch in enumerate(sequence):
        idx = NUCLEOTIDE_MAP.get(ch)
        if idx is not None:
            arr[i, idx] = 1.0
    return arr


def encode_dataset(df: pd.DataFrame):
    """Return X (N, L, 4) and y (N,) integer labels."""
    X = np.stack(df["sequence"].apply(one_hot_encode).values)
    y = df["label"].map(LABEL_MAP).values.astype(np.int64)
    return X, y


def stratified_split(X, y, test_size=0.15, val_size=0.15, seed=42):
    """Stratified train / val / test split."""
    # First split: train+val vs test
    X_tmp, X_test, y_tmp, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=seed
    )
    # Second split: train vs val (from the remaining)
    val_frac = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_tmp, y_tmp, test_size=val_frac, stratify=y_tmp, random_state=seed
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def load_and_prepare(seed: int = 42):
    """Full pipeline: download -> parse -> clean -> encode -> split."""
    np.random.seed(seed)
    filepath = download_data()
    df = parse_splice_file(filepath)
    df = clean_sequences(df)
    X, y = encode_dataset(df)
    splits = stratified_split(X, y, seed=seed)
    return df, X, y, splits
