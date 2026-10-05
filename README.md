# Multiscale Entropy of Recurrence Microstates for ECG Arrhythmia Detection

This repository contains the source code and precomputed feature datasets associated with the paper:

> **Multiscale Entropy of Recurrence Microstates and Machine Learning for Cardiac Arrhythmia Detection in ECG Signals**

submitted to the **26th IEEE International Conference on Bioinformatics and Bioengineering (BIBE 2026)**.

The project investigates the use of **Multiscale Recurrence Microstate Entropy (RMEn)** as a nonlinear representation of ECG signals for automated cardiac rhythm classification using machine learning.

---

## Overview

Electrocardiography (ECG) contains rich temporal information about cardiac electrical activity. While conventional ECG analysis often focuses on waveform morphology and clinically established temporal or frequency-domain features, nonlinear measures can provide complementary information about the complexity and temporal organization of cardiac dynamics.

This work investigates a **recurrence-microstate-based entropy measure** at multiple temporal scales and evaluates its ability to discriminate cardiac rhythms using machine learning classifiers.

The proposed computational pipeline can be summarized as:

```text
                     ECG Signal
                         │
                         ▼
                ┌─────────────────┐
                │   ECG Leads     │
                └────────┬────────┘
                         │
                         ▼
                ┌─────────────────┐
                │ Normalization   │
                │     [0, 1]      │
                └────────┬────────┘
                         │
                         ▼
                ┌──────────────────┐
                │  Coarse-Graining │
                │  Scale τ         │
                └────────┬─────────┘
                         │
                         ▼
                ┌─────────────────┐
                │   Recurrence    │
                │   Microstates   │
                └────────┬────────┘
                         │
                         ▼
                ┌──────────────────┐
                │ Maximum Entropy  │
                │ Threshold Search │
                └────────┬─────────┘
                         │
                         ▼
                ┌──────────────────┐
                │      RMEn        │
                │  Feature Vector  │
                └────────┬─────────┘
                         │
              ┌──────────┴──────────┐
              │                     │
              ▼                     ▼
       Single Scale          Multiple Scales
              │                     │
              └──────────┬──────────┘
                         ▼
                ┌──────────────────┐
                │ Machine Learning │
                │ Classification   │
                └────────┬─────────┘
                         │
                         ▼
                 Rhythm Detection
```

The current implementation supports experiments using individual scales as well as combinations of multiple scales.

---

## Method

### 1. ECG preprocessing

For each ECG recording, the available leads are processed independently.

Constant signals are discarded because normalization and entropy estimation are not meaningful for a signal with zero dynamic range.

Each non-constant lead is normalized to the interval:

\[
[0,1]
\]

The implementation performs this normalization independently for each ECG lead.

---

### 2. Multiscale coarse-graining

To investigate cardiac dynamics at different temporal resolutions, the ECG signal is coarse-grained using a scale factor \(\tau\).

For a given scale, the signal is divided into non-overlapping blocks of length \(\tau\), and the mean value of each block is calculated:

\[
y_j^{(\tau)}
=

\frac{1}{\tau}
\sum_{i=(j-1)\tau+1}^{j\tau} x_i
\]

This produces a coarse-grained representation of the original ECG signal at scale \(\tau\).

The repository currently provides precomputed datasets for:

- Scale 1
- Scale 10
- Scale 20
- Scale 30

The extraction script accepts the scale as a command-line argument.

---

## Recurrence Microstates

For each coarse-grained ECG sequence, pairs of positions are randomly selected and compared using a recurrence threshold.

A \(3 \times 3\) block is used to construct a **recurrence microstate**.

For each pair of positions, the algorithm evaluates whether corresponding samples satisfy:

\[
|x_i-y_j| \leq \epsilon
\]

where \(\epsilon\) is the recurrence threshold.

Each of the nine positions in the \(3 \times 3\) block is represented as a binary state. Therefore, the resulting microstate space contains:

\[
2^{3\times3}=2^9=512
\]

possible recurrence microstates.

The frequency distribution of these microstates is then used to calculate Shannon entropy.

---

## Maximum Recurrence Entropy

Rather than fixing the recurrence threshold to a single value, the implementation searches for the threshold that produces the maximum microstate entropy.

The algorithm evaluates multiple threshold values using an adaptive search strategy.

For a microstate probability distribution \(p_i\), the entropy is calculated as:

\[
S=-\sum_i p_i\log(p_i)
\]

The maximum entropy value is subsequently normalized by the maximum possible entropy of the \(3\times3\) binary microstate space:

\[
RMEn =
\frac{S_{\max}}
{9\log(2)}
\]

The implementation also retains the threshold associated with the maximum entropy.

---

## Feature Representation

For each ECG lead and scale, two values are generated:

1. **RMEn** — normalized maximum recurrence microstate entropy;
2. **\(\epsilon\)** — recurrence threshold associated with the maximum entropy.

The features are stored in an interleaved representation:

```text
[RMEn₁, ε₁, RMEn₂, ε₂, ..., RMEnₖ, εₖ]
```

where each pair corresponds to a lead/scale combination.

For a 12-lead ECG, a single scale therefore produces:

\[
12 \times 2 = 24
\]

features.

When multiple scales are combined, the corresponding feature matrices are concatenated horizontally.

For example:

```text
Scale 10  → (N, 24)
Scale 20  → (N, 24)

Combined → (N, 48)
```

and:

```text
Scale 1   → (N, 24)
Scale 10  → (N, 24)
Scale 20  → (N, 24)

Combined → (N, 72)
```

The multiscale classification script implements this feature concatenation directly.

---

# Dataset

The repository contains the precomputed feature matrices and corresponding diagnostic labels used by the machine learning experiments.

### Feature files

```text
Data_S_msrec_scale_1.npy
Data_S_msrec_scale_10.npy
Data_S_msrec_scale_20.npy
Data_S_msrec_scale_30.npy
```

The `Data_S` files contain the RMEn-based feature matrices for the corresponding scales.

### Label files

```text
Data_L_msrec_scale_1.npy
Data_L_msrec_scale_10.npy
Data_L_msrec_scale_20.npy
Data_L_msrec_scale_30.npy
```

The `Data_L` files contain the diagnostic class labels.

The repository currently includes these precomputed `.npy` datasets.

---

## Cardiac Rhythm Classes

The diagnostic labels used by the feature extraction pipeline are:

| Label | Rhythm |
| ---: | --- |
| 0 | SR |
| 1 | SB |
| 2 | AFIB |
| 3 | ST |
| 4 | SVT |
| 5 | AF |
| 6 | SI |
| 7 | AT |
| 8 | AVNRT |
| 9 | AVRT |
| 10 | SAAWR |

The same class mapping is used by the classification pipeline.

---

# Machine Learning

The repository provides separate pipelines for **binary** and **multiclass** classification.

## Binary Classification

The binary classification pipeline evaluates:

```text
SR vs. target rhythm
```

where `SR` is class 0 and the target rhythm is selected from the remaining diagnostic groups.

Three classifiers are currently implemented:

- Random Forest
- SVM with RBF kernel
- Artificial Neural Network (MLP)

The default classifier is **SVM-RBF**.

### Evaluation metrics

The binary classification pipeline reports:

- Accuracy
- Balanced Accuracy
- Macro-F1
- Sensitivity
- Specificity
- Precision

The final results are reported as mean ± standard deviation across the cross-validation folds.

---

## Multiclass Classification

The multiclass pipeline evaluates the simultaneous classification of the cardiac rhythm classes.

Multiple scales can be selected and concatenated before classification.

For example:

```bash
python multi_class_multiscale.py 1 10 20
```

creates a multiscale feature representation using scales 1, 10, and 20.

The resulting feature blocks are concatenated along the feature dimension.

### Available classifiers

The multiclass implementation supports:

- K-Nearest Neighbors (KNN)
- Decision Tree
- Random Forest
- SVM-RBF
- Artificial Neural Network (MLP)

The default classifier is SVM-RBF.

---

# Cross-Validation

The classification experiments use **stratified 10-fold cross-validation**.

The random seed is fixed to:

```text
1001
```

to improve reproducibility.

The stratification procedure preserves the class distribution across the folds.

For the multiclass experiments, the implementation reports:

- Accuracy
- Balanced Accuracy
- Macro-F1
- Macro Sensitivity
- Macro Specificity
- Macro Precision

The aggregated predictions from the held-out folds are also used to calculate per-class performance and the aggregated confusion matrix.

---

# Installation

Clone the repository:

```bash
git clone https://github.com/andrecoimbra/ecg-multiscale-erm.git
cd ecg-multiscale-erm
```

Create a Python virtual environment:

```bash
python -m venv .venv
```

Activate it on macOS/Linux:

```bash
source .venv/bin/activate
```

On Windows:

```powershell
.venv\Scripts\activate
```

Install the required Python packages:

```bash
pip install numpy scikit-learn tqdm
```

---

# Feature Extraction

The feature extraction script is:

```text
evaluate_multiscale_entropy_ECG.py
```

It expects three auxiliary files:

```text
filename.dat
diagnostics.dat
block_list.dat
```

and ECG recordings located according to the path configured in the script:

```text
../ecg-signal-classification-entropies/database/ECGDataDenoised/
```

The ECG recordings are expected as CSV files.

---

## Generate Features for a Scale

Run:

```bash
python evaluate_multiscale_entropy_ECG.py <scale>
```

For example:

```bash
python evaluate_multiscale_entropy_ECG.py 10
```

This generates:

```text
Data_S_msrec_scale_10.npy
Data_L_msrec_scale_10.npy
```

Similarly:

```bash
python evaluate_multiscale_entropy_ECG.py 20
```

generates:

```text
Data_S_msrec_scale_20.npy
Data_L_msrec_scale_20.npy
```

The extraction process uses multiprocessing through Python's `ProcessPoolExecutor` and displays a progress bar using `tqdm`.

---

# Binary Classification

The binary classification script is:

```text
binary_class_multiscale.py
```

The general syntax is:

```bash
python binary_class_multiscale.py <group> <scale1> [scale2 ...] [-c <classifier>]
```

For example, to compare SR with class 1 using scale 10:

```bash
python binary_class_multiscale.py 1 10
```

Using scales 10 and 20:

```bash
python binary_class_multiscale.py 1 10 20
```

Using scales 1, 10, and 20:

```bash
python binary_class_multiscale.py 1 1 10 20
```

### Classifier selection

The classifier is selected using `-c` or `--classifier`.

```text
0 → Random Forest
1 → SVM-RBF
2 → ANN (MLP)
```

Examples:

```bash
python binary_class_multiscale.py 1 10 20 -c 0
```

```bash
python binary_class_multiscale.py 1 10 20 -c 1
```

```bash
python binary_class_multiscale.py 1 10 20 -c 2
```

The SVM and MLP pipelines use feature standardization fitted only on the training partition of each cross-validation fold, avoiding information leakage.

---

# Multiclass Classification

The multiclass classification script is:

```text
multi_class_multiscale.py
```

The general syntax is:

```bash
python multi_class_multiscale.py <scale1> [scale2 ...] [-c <classifier>]
```

Examples:

```bash
python multi_class_multiscale.py 20
```

```bash
python multi_class_multiscale.py 10 20
```

```bash
python multi_class_multiscale.py 1 10 20
```

The order of the scales provided on the command line determines the order in which the corresponding feature blocks are concatenated.

---

# Reproducibility

The main parameters controlling the nonlinear feature extraction are:

| Parameter | Value |
| --- | ---: |
| Recurrence block size | \(3 \times 3\) |
| Possible microstates | 512 |
| Maximum sampled pairs | 10,000 |
| Random seed | 1001 |
| Cross-validation folds | 10 |
| Feature normalization | [0, 1] |
| ECG leads | 12 |
| Features per scale | 24 |

The feature extraction code uses a fixed random seed before multiprocessing, while the classification pipelines also use seed `1001`.

---

# Repository Structure

```text
ecg-multiscale-erm/
│
├── Data_L_msrec_scale_1.npy
├── Data_L_msrec_scale_10.npy
├── Data_L_msrec_scale_20.npy
├── Data_L_msrec_scale_30.npy
│
├── Data_S_msrec_scale_1.npy
├── Data_S_msrec_scale_10.npy
├── Data_S_msrec_scale_20.npy
├── Data_S_msrec_scale_30.npy
│
├── binary_class_multiscale.py
├── multi_class_multiscale.py
├── evaluate_multiscale_entropy_ECG.py
│
├── filename.dat
├── diagnostics.dat
├── block_list.dat
│
├── LICENSE
├── .gitignore
└── README.md
```

The structure above reflects the current public repository.

---

# Paper

This repository supports the following manuscript:

> **Multiscale Entropy of Recurrence Microstates and Machine Learning for Cardiac Arrhythmia Detection in ECG Signals**

**Conference:**  
26th IEEE International Conference on Bioinformatics and Bioengineering (BIBE 2026)

**Year:** 2026

BIBE 2026 is scheduled for **November 27–29, 2026, in Shanghai, China**. The conference focuses on interdisciplinary research spanning bioinformatics, bioengineering, biomedicine, and related computational approaches.

---

# License

This project is released under the **MIT License**.

See the [`LICENSE`](LICENSE) file for details. The GitHub repository currently identifies the project as using the MIT license.