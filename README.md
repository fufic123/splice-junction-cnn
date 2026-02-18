# CNN for DNA Splice-Junction Classification

Classification of DNA sequences into splice-junction types (EI, IE, N) using 1D Convolutional Neural Networks on the UCI "Molecular Biology (Splice-junction Gene Sequences)" dataset.

## Problem Description

Splice junctions are points on a DNA sequence where "superfluous" intron DNA is removed during RNA processing. This project classifies 60-nucleotide DNA sequences into three categories:

- **EI** — exon-intron boundary (donor site)
- **IE** — intron-exon boundary (acceptor site)
- **N** — neither (non-splice region)

## Why CNNs for DNA?

1D Conv filters of shape `(kernel_size, 4)` applied to one-hot encoded DNA are mathematically equivalent to **position weight matrices (PWMs)** — the standard tool in bioinformatics for scoring sequence motifs. The network learns motif detectors end-to-end:

- **kernel_size** controls motif length (e.g., 7 ≈ capturing splice-site consensus like GT...AG)
- **GlobalMaxPooling** provides translation invariance — a motif is detected regardless of position
- **Dropout** regularizes against overfitting on this small (~3,190 samples) dataset

## Models

| Model | Description |
|-------|-------------|
| **Model A** — Baseline CNN | Conv1D → ReLU → GlobalMaxPool → Dense → Softmax (article-inspired) |
| **Model B** — Dilated Residual CNN | Residual blocks with dilated convolutions (rates 1,2,4) for larger receptive field |

## Repository Structure

```
.
├── notebooks/
│   └── splice_cnn_project.ipynb    # Main notebook (run top-to-bottom)
├── src/
│   ├── data.py                     # Download, parse, encode, split
│   ├── models.py                   # Model A (baseline CNN) + Model B (dilated res CNN)
│   ├── train.py                    # Training loop with callbacks
│   ├── eval.py                     # Evaluation metrics + all plotting functions
│   └── utils.py                    # Seeding, logging, helpers
├── outputs/
│   ├── figures/                    # Saved plots (auto-generated)
│   ├── results.csv                 # Per-run results log
│   ├── model_comparison.csv        # Summary table
│   ├── hyperparam_study_A.csv      # Hyperparameter study — Model A
│   └── hyperparam_study_B.csv      # Hyperparameter study — Model B
├── data/                           # Auto-downloaded UCI data
├── README.md
├── presentation.md
└── requirements.txt
```

## Quick Start

```bash
# 1. Create virtual environment & install dependencies
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 2. Run the notebook
cd notebooks
jupyter notebook splice_cnn_project.ipynb
```

The notebook automatically downloads the dataset from UCI on first run.

## Requirements

- Python 3.10+
- TensorFlow >= 2.16 / Keras 3
- numpy, pandas, scikit-learn, matplotlib, requests

All experiments run on **CPU** (no GPU required). Full pipeline takes ~15-30 minutes.

## Experiments & Results

### Training Protocol

- **Split:** Stratified 70/15/15 (train/val/test)
- **Optimizer:** Adam (lr=1e-3) with ReduceLROnPlateau
- **Early stopping:** patience=10 on validation loss
- **Reproducibility:** 3 random seeds (42, 123, 777), reporting mean ± std

### Model Comparison

| Model | Accuracy (mean ± std) | Macro-F1 (mean ± std) |
|-------|----------------------|----------------------|
| Baseline CNN (A) | 0.784 ± 0.017 | 0.768 ± 0.018 |
| Dilated Res CNN (B) | **0.856 ± 0.009** | **0.851 ± 0.007** |

### Hyperparameter Study (Model A)

| Parameter | Values | Observation |
|-----------|--------|-------------|
| kernel_size | 5, 7, 11 | 11 performs best (0.825 acc); larger kernels capture longer splice motifs |
| n_blocks | 1, 2, 3 | 2 blocks optimal (0.785); 1 block underfits (0.745) |
| dropout | 0.0, 0.3, 0.5 | Minimal effect in this range; 0.0 slightly best (0.787) |

### Key Findings

1. Conv1D filters learn biologically meaningful splice-site motifs (GT/AG consensus)
2. Dropout is critical for this small dataset — removing it causes clear overfitting
3. Dilated convolutions modestly improve performance by capturing longer-range context
4. EI and IE classes are occasionally confused (both are splice sites with structural similarities)

## Dataset

- **Source:** [UCI ML Repository — Dataset #69](https://archive.ics.uci.edu/dataset/69/molecular+biology+splice+junction+gene+sequences)
- **Size:** 3,190 sequences × 60 nucleotides
- **Classes:** EI (767), IE (768), N (1,655)
- **Preprocessing:** ambiguous nucleotides (N, D, S, R, etc.) replaced with random ACGT

## References

- Noordewier, M.O., Towell, G.G., Shavlik, J.W. (1991). Training Knowledge-Based Neural Networks to Recognize Genes in DNA Sequences. *Advances in Neural Information Processing Systems*.
- UCI Machine Learning Repository: Molecular Biology (Splice-junction Gene Sequences) Data Set.
