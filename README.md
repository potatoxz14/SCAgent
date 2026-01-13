# Rethinking Side-Channel Analysis: Automated Discovery and Analysis of Side-Channel Leakage with LLM-Assisted Agents

This repository contains the source code for the paper **Rethinking Side-Channel Analysis: Automated Discovery and Analysis of Side-Channel Leakage with LLM-Assisted Agents**. The framework provides tools for time-series feature extraction, dimensionality reduction, classification using TabPFN, and LLM-based side-channel vector proposal/verification.

## 1. Data Structure Organization

To ensure the scripts run correctly, the dataset must be organized in a hierarchical directory structure. The root folder should contain subdirectories for each class (e.g., website labels), and each subdirectory should contain the CSV samples.

**Required Structure:**

Plaintext

```
/path/to/dataset_root/
    ├── www.google.com/
    │   ├── 1.csv
    │   ├── 2.csv
    │   └── ...
    ├── www.facebook.com/
    │   ├── 1.csv
    │   └── ...
    └── [Other Classes]/
```

*Note: In the scripts, the `CSV_ROOT_FOLDER` variable must be updated to point to this root directory.*

## 2. Requirements

- **Python 3.8+**
- **Key Libraries:** `numpy`, `pandas`, `scikit-learn`, `sktime`, `tabpfn`, `joblib`
- **GPU Support:** `cupy`, `cuml` (required for PCA utilities and GPU acceleration)
- **LLM Modules:** `google-generativeai`

To install the primary classifier dependency:

Bash

```
pip install tabpfn
```

## 3. Configuration & Common Variables

Most scripts share the following configuration variables at the top of the file:

- **`CSV_ROOT_FOLDER`**: The path to your dataset directory.
- **`TARGET_COLUMN_INDICES`**: A list of indices (0-based) representing the specific data columns (features) to be extracted from the CSV files.
- **`TARGET_LENGTH`**: The target length for time-series normalization (e.g., 640 or 1000).

------

## 4. Usage Guide

### A. Main Classification Pipeline

**File:** `main_multi_feature.py`

This script performs the primary website fingerprinting analysis. It executes the following pipeline:

1. **Preprocessing:** Loads data from `CSV_ROOT_FOLDER` and standardizes length.
2. **Feature Extraction:** Uses **MiniRocket** (Multivariate) to extract 10,000 kernels.
3. **Dimensionality Reduction:** Applies PCA (via `pca_utils.py`) to reduce features to 250 dimensions.
4. **Classification:** Trains a **TabPFN** classifier (One-vs-Rest) and evaluates accuracy.

**Usage:**

Bash

```
python main_multi_feature.py
```

### B. Sampling Rate Analysis

**File:** `main_multi_diff_frequency.py`

Analyzes the robustness of the attack across different sampling frequencies. It downsamples the original data (defined by `ORIGINAL_FREQ`) to various target frequencies defined in `TEST_FREQUENCIES` (e.g., 5Hz, 10Hz, 50Hz) and reports accuracy for each.

**Usage:**

Bash

```
python main_multi_diff_frequency.py
```

### C. Noise Robustness Analysis

**File:** `main_diff_gaussian_noise.py`

Evaluates the model's performance under noisy conditions. It injects Gaussian noise at varying ratios (defined in `TEST_NOISE_RATIOS`) relative to the feature standard deviation and outputs accuracy metrics for each noise level.

**Usage:**

Bash

```
python main_diff_gaussian_noise.py
```

### D. Device Transferability

**File:** `main_transfability.py`

Tests the model's ability to generalize across different hardware devices (e.g., training on iPhone 13, testing on iPhone 14).

- **Configuration:** Update `FOLDER_IPHONE_13` and `FOLDER_IPHONE_14` paths before running.
- **Output:** Accuracy scores for both the source and target devices.

**Usage:**

Bash

```
python main_transfability.py
```

------

## 5. LLM-Assisted Side-Channel Discovery

This module utilizes Large Language Models (LLMs) to analyze system internals and propose new side-channel vectors.

### A. Vector Proposal

**File:** `proposal_ios.py`

Generates novel side-channel attack proposals by analyzing academic papers.

- **Setup:**
  1. Set your Google API Key in `os.environ["GOOGLE_API_KEY"]`.
  2. Set `PAPERS_DIR` to the folder containing reference PDFs.
  3. Adjust the `system_prompt` if necessary to refine the agent's focus.

**Usage:**

Bash

```
python proposal_ios.py
```

### B. Vector Verification

**File:** `verify_ios.py`

Validates the feasibility of the proposed vectors (from `proposal_ios.py`) using an LLM agent acting as an iOS security researcher.

- **Setup:**
  1. Set your Google API Key.
  2. Populate the `PROPOSALS` list with the output text from the proposal script.

**Usage:**

Bash

```
python verify_ios.py
```

------

## 6. Utilities

**File:** `pca_utils.py`

A helper module handling GPU-accelerated PCA operations using `cuml` and `cupy`.

- *Note: This file is not intended to be run directly; it is imported by the main training scripts.*
