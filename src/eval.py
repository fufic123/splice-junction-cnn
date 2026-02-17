import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, f1_score, confusion_matrix, classification_report,
)


def evaluate_model(model, X_test, y_test, label_names=None):
    """Return dict with accuracy, macro_f1, confusion matrix, predictions."""
    y_prob = model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_prob, axis=1)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="macro")
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=label_names, digits=4)
    return {
        "accuracy": acc,
        "macro_f1": f1,
        "confusion_matrix": cm,
        "y_pred": y_pred,
        "y_prob": y_prob,
        "report": report,
    }


def plot_confusion_matrix(cm, label_names, title="Confusion Matrix", save_path=None):
    """Plot confusion matrix as a color-coded table."""
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(label_names)))
    ax.set_yticks(range(len(label_names)))
    ax.set_xticklabels(label_names)
    ax.set_yticklabels(label_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    # Annotate cells
    for i in range(len(label_names)):
        for j in range(len(label_names)):
            color = "white" if cm[i, j] > cm.max() / 2 else "black"
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color=color, fontsize=12)
    fig.colorbar(im, ax=ax)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_training_curves(history, title="Training curves", save_path=None):
    """Plot loss and accuracy curves."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    # Loss
    axes[0].plot(history.history["loss"], label="train")
    axes[0].plot(history.history["val_loss"], label="val")
    axes[0].set_title(f"{title} — Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    # Accuracy
    axes[1].plot(history.history["accuracy"], label="train")
    axes[1].plot(history.history["val_accuracy"], label="val")
    axes[1].set_title(f"{title} — Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_class_distribution(y, label_names, title="Class distribution", save_path=None):
    """Bar chart of class counts."""
    unique, counts = np.unique(y, return_counts=True)
    fig, ax = plt.subplots(figsize=(5, 3.5))
    bars = ax.bar([label_names[i] for i in unique], counts, color=["#4c72b0", "#dd8452", "#55a868"])
    ax.set_ylabel("Count")
    ax.set_title(title)
    for bar, c in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5,
                str(c), ha="center", va="bottom", fontsize=10)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_nucleotide_frequencies(df, title="Overall nucleotide frequencies", save_path=None):
    """Bar chart of nucleotide frequencies across all sequences."""
    from collections import Counter
    all_seq = "".join(df["sequence"])
    counts = Counter(all_seq)
    total = sum(counts.values())
    bases = ["A", "C", "G", "T"]
    freqs = [counts.get(b, 0) / total for b in bases]
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.bar(bases, freqs, color=["#e74c3c", "#3498db", "#2ecc71", "#f39c12"])
    ax.set_ylabel("Frequency")
    ax.set_title(title)
    ax.set_ylim(0, max(freqs) * 1.2)
    for i, (b, fr) in enumerate(zip(bases, freqs)):
        ax.text(i, fr + 0.005, f"{fr:.3f}", ha="center", fontsize=9)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_position_base_frequency(df, title="Per-position base frequency", save_path=None):
    """Heatmap of nucleotide frequency at each position (matplotlib only)."""
    seqs = df["sequence"].values
    seq_len = len(seqs[0])
    bases = ["A", "C", "G", "T"]
    freq_matrix = np.zeros((4, seq_len))
    for seq in seqs:
        for j, ch in enumerate(seq):
            idx = {"A": 0, "C": 1, "G": 2, "T": 3}.get(ch)
            if idx is not None:
                freq_matrix[idx, j] += 1
    freq_matrix /= len(seqs)

    fig, ax = plt.subplots(figsize=(12, 2.5))
    im = ax.imshow(freq_matrix, aspect="auto", cmap="YlOrRd", interpolation="nearest")
    ax.set_yticks(range(4))
    ax.set_yticklabels(bases)
    ax.set_xlabel("Position")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, shrink=0.8, label="Frequency")
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_param_effect(results_df, param_col, metric_col="test_macro_f1",
                      title=None, save_path=None):
    """Line plot showing the effect of one hyper-parameter on a metric."""
    grouped = results_df.groupby(param_col)[metric_col].agg(["mean", "std"]).reset_index()
    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.errorbar(grouped[param_col].astype(str), grouped["mean"], yerr=grouped["std"],
                marker="o", capsize=4, linewidth=1.5)
    ax.set_xlabel(param_col)
    ax.set_ylabel(metric_col)
    ax.set_title(title or f"Effect of {param_col} on {metric_col}")
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig
