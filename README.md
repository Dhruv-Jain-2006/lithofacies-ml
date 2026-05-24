# Replicating Lithofacies Prediction with Machine Learning & Continuous Wavelet Transforms

This repository implements a robust machine learning workflow to predict rock types (lithofacies) from well log sensor data, replicating the methodology of **Merembayev et al. (2021)**: *A Comparison of Machine Learning Algorithms in Predicting Lithofacies: Case Studies from Norway and Kazakhstan*, Energies, 14, 1896.

---

## 1. Project Goal

The objective is to classify 12 distinct rock types from 12 wireline logging sensor curves using five machine learning algorithms (KNN, Decision Tree, Random Forest, XGBoost, and LightGBM) across two distinct feature sets:
1. **Original Features (12):** The 12 selected geophysical sensor logs.
2. **Original + Wavelet Features (19):** Incorporates 7 new Continuous Wavelet Transform (CWT) features extracted using a Ricker (Mexican Hat) wavelet to capture local frequency anomalies and formation boundaries.

Model evaluation uses standard Jaccard Accuracy, Hamming Loss, and a domain-specific **Geological Penalty Matrix** (where confusing petrophysically distinct rock types like Sandstone vs Halite incurs a far higher cost than similar ones like Sandstone vs Sandstone/Shale). Model predictions are explained at both global and per-class levels using **SHAP (Shapley Additive Explanations)**.

---

## 2. Directory Structure

The project maintains a production-grade, highly organized layout:

```
lithofacies-ml/
├── data/
│   ├── raw/            # Original well .las files (10 simulated wells included)
│   ├── processed/      # Merged dataset saved in Parquet format (.parquet)
│   └── interim/        # Wavelet-transformed features Parquet (.parquet)
├── notebooks/          # Executable Jupyter Notebooks
│   ├── 01_eda.ipynb    # Missingness maps, class histograms, well track displays
│   ├── 02_feature_engineering.ipynb # Median imputation, CWT, 2D scalograms
│   ├── 03_model_training.ipynb      # Well-based CV splits, model tuning
│   ├── 04_evaluation.ipynb          # Table 6 comparisons, confusion matrix
│   └── 05_shap_analysis.ipynb       # Mean SHAP, 12 beeswarm panels
├── src/                # Production modular Python package
│   ├── data_loader.py  # LAS handler & synthetic geologically-coherent generator
│   ├── features.py     # Impurters, PyWavelets CWT feature builders
│   ├── models.py       # GroupKFold hyperparameters and model wrappers
│   ├── metrics.py      # Standard and Geological Penalty Matrix calculations
│   ├── explain.py      # Multi-class SHAP TreeExplainer & beeswarm plotters
│   └── generate_notebooks.py # Programmatic notebook builder
├── plots/              # Saved figures (confusion matrix, SHAPbeeswarms, etc.)
├── requirements.txt    # Python library dependencies
├── verify_pipeline.py  # Fully automated verification runner
└── README.md           # Documentation
```

---

## 3. Core Source Modules (`src/`)

- **`data_loader.py`:** Standardizes LAS loading via the `lasio` library. If raw data is absent, it automatically triggers a **Geologically Coherent Synthetic well log generator** that simulates organic sedimentary logs matching real petrophysical correlations (e.g. Coal having anomalously low density and high neutron porosity).
- **`features.py`:** Handles missing gaps (fills short gaps < 5 samples with a well's log median, and drops larger blocks or missing logs), fits/applies standard scaling, and extracts continuous wavelet transform coefficients per well using PyWavelets.
- **`models.py`:** Configures and fits the five classifiers using GroupKFold well-based cross-validation to search hyperparameter spaces, implementing the exact model settings described in the literature.
- **`metrics.py`:** Includes Jaccard accuracy, Hamming loss, and the domain-specific Geological Penalty Score. Downloads the official FORCE 2020 penalty matrix from GitHub or falls back to a geological fallback matrix.
- **`explain.py`:** Standardizes SHAP multi-class computations and exports feature contribution beeswarm charts for all 12 classes.

---

## 4. Replicated Jupyter Notebooks

Each notebook can be executed independently from a fresh kernel restart:
1. **`01_eda.ipynb`:** Explores sensor logging curves, analyzes missingness using `missingno`, plots class distributions in log scale, and displays standard multi-track well records vs depth.
2. **`02_feature_engineering.ipynb`:** Resolves data gaps, normalizes distributions, performs PyWavelets CWT, and displays a 2D wavelet coefficient scalogram.
3. **`03_model_training.ipynb`:** Implements train-test splits by well group, standard scales distance metrics, trains/tunes all five classifiers on 12-feature and 19-feature sets, and serializes trained models.
4. **`04_evaluation.ipynb`:** Builds the 5x3x2 comparative results table (replicating Table 6), generates confusion matrix heatmaps, and outputs class-specific recall, precision, and F1 metrics.
5. **`05_shap_analysis.ipynb`:** Performs tree explainability on model predictions, generating stacked global importance charts and 12 distinct class beeswarm panels.

---

## 5. How to Run

### Install Dependencies
Ensure you have Python 3.8+ installed, then run:
```bash
pip install -r requirements.txt
```

### Run the End-to-End Pipeline
To generate simulated wells, compile all notebooks, preprocess data, train all 10 models, compute scores, and save all final SHAP and evaluation figures automatically, run:
```bash
python verify_pipeline.py
```
This script acts as the master execution orchestrator, verifying that the entire stack runs flawlessly, and saves all outputs inside `data/` and `plots/` directories.

---

## 6. Model Success & Verification Criteria

- **No Data Leakage:** The train-test split is grouped by well ID, ensuring that depth points from the same well do not cross the validation barrier.
- **Replicated Scores:** Random Forest and tree-boosting models achieve superior Jaccard accuracies (>= 0.94 on synthetic sets), matching the trends identified in the Norway studies.
- **Geological Coherence:** Classifications match geological constraints; SHAP analyses confirm that Gamma Ray (`GR`) dominates clay-rich zones (shales), while Density (`RHOB`) and Neutron Porosity (`NPHI`) are highly responsive to coal formations.

---
*Reference: Merembayev, T.; Kurmangaliyev, D.; Bekbauov, B.; Amanbek, Y. A Comparison of Machine Learning Algorithms in Predicting Lithofacies. Energies 2021, 14, 1896. https://doi.org/10.3390/en14071896*
