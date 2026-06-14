# Seis-JEPA: Self-Supervised Seismic World Modelling for Wildlife Identification under Severe Class Imbalance

[![arXiv](https://img.shields.io/badge/arXiv-XXXX.XXXXX-b31b1b.svg)](https://arxiv.org/abs/XXXX.XXXXX)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)](https://pytorch.org/)

**Mohamed Waleed Fakhr**
School of Computing, Coventry University TKH Branch, Cairo, Egypt

---

## Overview

Inspired by the **golden mole** (*Chrysochloridae*) — a blind, deaf African mammal that
navigates its environment purely through seismic ground vibrations — Seis-JEPA is the
**first Joint Embedding Predictive Architecture (JEPA) applied to seismic wildlife data**.

The system learns representations of wildlife seismic signals without any label information,
applied to the [SeisSavanna dataset](https://github.com/AlexSzen/SeisSavanna)
(70,308 spectrograms, six species, **430:1 human-to-elephant imbalance ratio**).

---

## Key Contributions

### 1. Seis-JEPA Architecture
A JEPA encoder adapted for three-component seismic spectrograms (Z, N, E channels)
featuring:
- **4-layer transformer encoder** (D=256, 8 heads) with multi-scale frequency-band masking
- **Affine frequency target**: T(x) = w ⊙ x + b — a learned per-frequency gain and bias
  analogous to the golden mole's basilar membrane, acting as a shallow signal processing
  interface rather than a full momentum encoder
- **Low-frequency augmentation**: bins 0–31 upsampled and concatenated as extra channels,
  exploiting the surface wave band where species-discriminative information concentrates

### 2. Representation Learning Disparity at Extreme Imbalance
We characterise and quantify the **representation learning disparity** that emerges at
430:1 class imbalance — far beyond the ~100:1 ratio where standard SSL robustness holds.
Under uniform sampling, elephant recall drops to **30.0%** vs a supervised upper bound
of **95.0%** — a **65 percentage point supervision gap**.

### 3. Systematic Negative Result on Unsupervised Correction
We conduct the most thorough investigation to date of unsupervised rare-class detection
on severely imbalanced seismic data, testing **10 methods** across two feature spaces:

| Method | Feature space | Elephant/human ratio |
|--------|--------------|---------------------|
| K-means++ (K=500) | Mel spectrogram (264-d) | 1.24× |
| K-means++ (K=50) | Mel spectrogram (264-d) | 1.28× |
| HDBSCAN | Mel spectrogram (264-d) | 1.00× |
| Isolation Forest | Physical (20 feat.) | 1.69× |
| LOF (k=20) | Physical (20 feat.) | 1.19× |
| NN distance (k=10) | Physical (20 feat.) | 1.45× |
| IF + LOF + NN ensemble | Physical (28 feat.) | **1.76×** |
| GMMHMM (8 states, 3 mix.) | Raw seismogram (3C) | 1.02× |
| Multiplicative ensemble | Mel + raw combined | 1.39× |
| Fuzzy ART (ρ=0.60) | Physical (20 feat.) | 2.46× |
| **Supervised (upper bound)** | Labels | **378×** |

None exceeded the 3× threshold required for meaningful improvement.
This establishes a fundamental limit: **164 elephant samples at 20–150m sensor distance
is below the unsupervised geometric detection threshold**, regardless of method.

### 4. One-Shot Semi-Supervised Sampling (Condition D)
With just **6 labels** — one per class, 0.008% of the dataset — one-shot LDA-guided
sampling achieves **46.2% elephant recall** (+16.2 pp over uniform sampling),
establishing a practical minimum supervision threshold.

### 5. Biological Analogy: Affine Target as Signal Processing Interface
The affine frequency target mirrors the frequency-selective biological interfaces found
across taxa: the golden mole's basilar membrane, the elephant's saccule,
the fish's lateral line, and the snake's pit organ — all converge on the same
computational primitive: a frequency-selective affine transform.

---

## Results

| Condition | Sampler | Labels | Balanced Acc | Elephant Recall |
|-----------|---------|--------|-------------|-----------------|
| B — Supervised | Inverse-frequency | All (164) | **93.96%** | **95.0%** |
| C — Pure SSL | Uniform | None | 72.69% | 30.0% |
| D — One-shot | LDA-guided | 6 (0.008%) | 67.34% | 46.2% |

---

## Dataset

This code uses the [SeisSavanna dataset](https://github.com/AlexSzen/SeisSavanna)
by Mortimer et al. (2021). Download the following files from the
[Dropbox repository](https://www.dropbox.com/scl/fo/avwuo0jx9tjp6167auk2b/):

```
dset_allspec_150_spectrograms.nc   (13.89 GB) — mel spectrograms
dset_allspec_150_chunks_clean.nc   (27.49 GB) — raw seismograms
dset_allspec_150_chunks.csv        (6.21 MB)  — event metadata
```

Place all files in your configured `SEIS_DIR` path.

---

## Installation

```bash
git clone https://github.com/YOUR_USERNAME/Seis-JEPA.git
cd Seis-JEPA
pip install torch torchvision xarray netCDF4 numpy scipy scikit-learn artlib
```

**Requirements:**
- Python 3.10+
- PyTorch 2.0+ with CUDA (A100 / L4 recommended)
- Google Colab Pro+ (used for development, ~30 compute units per full run)

---

## Repository Structure

```
Seis-JEPA/
├── S_JEPA_V1_SSL.ipynb       # Condition B — supervised sampler (upper bound)
├── S_JEPA_V2_SSL.ipynb       # Condition C — uniform sampler (SSL baseline)
├── S_JEPA_V3_SSL.ipynb       # Condition D — one-shot LDA sampler
├── paper/
│   ├── seis_jepa.tex         # LaTeX source
│   └── seis_jepa.pdf         # Compiled paper
└── README.md
```

---

## Notebook Overview

Each notebook is self-contained and follows the same cell structure:

| Cell | Description |
|------|-------------|
| Cell 1 | Drive mount, imports, logging |
| Cell 2 | Configuration (architecture, training hyperparameters) |
| Cell 3 | Data loading with smart cache, train/val/test split |
| Cell 2b | Unsupervised sampler construction (Condition D only) |
| Cells 4–9 | Architecture: AffineFrequencyTarget, StructuredBayesianGMM, ContextEncoder, SJEPAPredictor, masking, loss |
| Cell 10 | Training loop (100 epochs, cosine LR decay, checkpointing) |
| Cell 11 | Evaluation: KNN (K=5) on frozen embeddings, per-class recall, CDNV |
| Cell 12 | Final test set evaluation and results JSON |

---

## Configuration

Key hyperparameters (Cell 2):

```python
# Architecture
D_MODEL        = 256      # transformer hidden dimension
N_HEADS        = 8        # attention heads
N_LAYERS       = 4        # encoder depth
PATCH_SIZE     = 8        # spectrogram patch size (8×8 pixels)
N_LOWFREQ_BINS = 32       # low-frequency augmentation band

# Training
TOTAL_EPOCHS   = 100
BATCH_SIZE     = 64
LR             = 3e-4     # AdamW, cosine decay
LAMBDA_GMM     = 0.0      # GMM loss weight (disabled for Conditions B/C/D)
AFFINE_MU_REC  = 0.0      # affine reconstruction loss (disabled)

# Data
VIABLE_CLASSES = [0, 1, 2, 5, 6, 9]   # noise, human, elephant, numididae, bovidae, leporidae
TRAIN_FRAC     = 0.75
VAL_FRAC       = 0.10
```

---

## One-Shot Sampler (Cell 2b — Condition D)

The one-shot LDA-guided sampler uses **1 labelled sample per class** (6 labels total):

1. **Medoid selection**: find the most representative sample per class in a 20-dimensional
   physical feature space (10 raw seismogram + 10 mel spectrogram features)
2. **One-shot LDA**: compute between-class scatter matrix from 6 medoid vectors,
   project all 70,308 samples onto 5 discriminant directions
3. **Prototype distance weighting**: score each sample by
   `dist_to_human / dist_to_elephant` in the combined 13-dimensional space
4. **Ensemble anomaly scoring**: multiply by Isolation Forest, LOF, and NN distance scores
5. **Inverse-prior weights**: convert scores to sampling weights (WEIGHT_CAP = N/10)

The 20 physical features include: distance-normalised PGV, spectral entropy,
cepstral gait period, polarisation linearity/planarity, instantaneous frequency,
Z-N cross-correlation lag, LF/HF energy ratio (raw), plus surface/body wave ratio,
inter-component coherence, decay time, temporal kurtosis, autocorrelation gait proxy,
and LF/HF contrast (mel).

---

## Citing

If you use this code or paper, please cite:

```bibtex
@article{fakhr2025seisjepa,
  title   = {Seis-JEPA: Self-Supervised Seismic World Modelling for Wildlife
             Identification under Severe Class Imbalance},
  author  = {Fakhr, Mohamed Waleed},
  journal = {arXiv preprint arXiv:XXXX.XXXXX},
  year    = {2025}
}
```

Please also cite the SeisSavanna dataset:

```bibtex
@article{mortimer2021classifying,
  title   = {Classifying elephant behaviour through seismic waves: a performance
             comparison of machine learning and signal processing approaches},
  author  = {Mortimer, Beth and Rees, William L and Koelemeijer, Paula and
             Nissen-Meyer, Tarje},
  journal = {Methods in Ecology and Evolution},
  volume  = {12},
  number  = {12},
  pages   = {2479--2491},
  year    = {2021}
}
```

---

## License

MIT License — see [LICENSE](LICENSE) for details.

---

## Acknowledgements

This work uses the SeisSavanna dataset graciously provided by Mortimer et al. (2021).
Experiments were conducted on Google Colab Pro+ with A100/L4 GPU.
The artlib library (Melton et al., 2025) was used for ART-based clustering experiments.
