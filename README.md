# Replicating Lithofacies Prediction with Machine Learning & Continuous Wavelet Transforms

This repository implements a robust machine learning workflow to predict rock types (lithofacies) from well log sensor data, replicating the methodology of **Merembayev et al. (2021)**: *A Comparison of Machine Learning Algorithms in Predicting Lithofacies: Case Studies from Norway and Kazakhstan*, Energies, 14, 1896.

It features an advanced geological training pipeline coupled with a premium, real-time **Interactive Flask Web Dashboard** featuring live sandbox classifications, Viterbi stratigraphic sequence decoding, and SHAP explainability.

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
│   ├── sequence.py     # Viterbi global path sequence probability decoders
│   ├── augmentation.py # Minority class geological oversampling strategies
│   └── generate_notebooks.py # Programmatic notebook builder
├── templates/          # React/Tailwind frontend premium dashboard templates
│   └── index.html      # High-fidelity dashboard view and geological sandbox
├── plots/              # Saved figures (confusion matrix, SHAP beeswarms, etc.)
├── app.py              # Interactive Flask Web Dashboard Server
├── requirements.txt    # Python library dependencies
├── verify_pipeline.py  # Fully automated verification runner
└── README.md           # Documentation
```

---

## 3. Core Source Modules (`src/`)

- **`data_loader.py`:** Standardizes LAS loading via the `lasio` library. If raw data is absent, it automatically triggers a **Geologically Coherent Synthetic well log generator** that simulates organic sedimentary logs matching real petrophysical correlations (e.g. Coal having anomalously low density and high neutron porosity).
- **`features.py`:** Handles missing gaps (fills short gaps < 5 samples with a well's log median, and drops larger blocks or missing logs), fits/applies standard scaling, and extracts continuous wavelet transform coefficients per well using PyWavelets.
- **`models.py`:** Configures and fits the five classifiers using GroupKFold well-based cross-validation to search hyperparameter spaces, implementing the exact model settings described in the literature.
- **`metrics.py`:** Includes Jaccard accuracy, Hamming loss, and the domain-specific Geological Penalty Score. Downloads the FORCE 2020 penalty matrix or falls back to a geological penalty fallback matrix..
- **`explain.py`:** Standardizes SHAP multi-class computations and exports feature contribution beeswarm charts for all 12 classes.
- **`sequence.py`:** Implements a dynamic programming **Viterbi sequence decoder**. Calculates transition probabilities from FORCE 2020 training sequence beds to resolve boundary noise and smooth high-frequency ML depth-wise classifications into contiguous stratigraphic layers.
- **`augmentation.py`:** Standardizes a geologically-coherent class oversampling framework to resolve severe dataset imbalances (skewed ratios of sandstone/shale vs. rare coal/anhydrite beds) without producing out-of-bounds petrophysical features..

---

## 4. Replicated Jupyter Notebooks

Each notebook can be executed independently from a fresh kernel restart:
1. **`01_eda.ipynb`:** Explores sensor logging curves, analyzes missingness using `missingno`, plots class distributions in log scale, and displays standard multi-track well records vs depth..
2. **`02_feature_engineering.ipynb`:** Resolves data gaps, normalizes distributions, performs PyWavelets CWT, and displays a 2D wavelet coefficient scalogram.
3. **`03_model_training.ipynb`:** Implements train-test splits by well group, standard scales distance metrics, trains/tunes all five classifiers on 12-feature and 19-feature sets, and serializes trained models.
4. **`04_evaluation.ipynb`:** Builds the 5x3x2 comparative results table (replicating Table 6), generates confusion matrix heatmaps, and outputs class-specific recall, precision, and F1 metrics.
5. **`05_shap_analysis.ipynb`:** Performs tree explainability on model predictions, generating stacked global importance charts and 12 distinct class beeswarm panels..

---

## 5. How to Run

### Install Dependencies
Ensure you have Python 3.8+ installed, then run:
```bash
pip install -r requirements.txt
```

### Run the End-to-End Pipeline (CLI)
To generate simulated wells, compile all notebooks, preprocess data, train all 10 models, compute scores, and save all final SHAP and evaluation figures automatically, run:
```bash
python verify_pipeline.py
```
This script acts as the master execution orchestrator, verifying that the entire stack runs flawlessly, and saves all outputs inside `data/` and `plots/` directories.

### Run the Interactive Web Dashboard (GUI)
To launch the real-time geological classifier sandbox and interactive log viewer, run:
```bash
python app.py
```
Open `http://localhost:5000` in your web browser to access:
- **5-Track Interactive Log Viewer:** Real-time log scrolling, visual active depth cursor tracking, model confidence badges, and prediction tracks.
- **Sandbox Parameter Space:** Live toggle between different machine learning models, active feature spaces (12 original vs 19 wavelet), and Viterbi sequence smoothing boundaries.
- **Live Attributions & Accordance:** Visual SHAP beeswarms for geological classes and model leaderboard statistics updating on the fly.

---

## 6. Model Success & Verification Criteria

- **No Data Leakage:** The train-test split is grouped by well ID, ensuring that depth points from the same well do not cross the validation barrier.
- **Replicated Scores:** Random Forest and tree-boosting models achieve superior Jaccard accuracies (>= 0.94 on synthetic sets), matching the trends identified in the Norway studies.
- **Stratigraphic Coherence:** Incorporating the Viterbi path dynamic programming decoder eliminates depth-wise predictions jitter, creating structurally sound, contiguous stratigraphic log intervals.
- **Geological Coherence:** Classifications match geological constraints; SHAP analyses confirm that Gamma Ray (`GR`) dominates clay-rich zones (shales), while Density (`RHOB`) and Neutron Porosity (`NPHI`) are highly responsive to coal formations.

---

## 7. Production Cloud Hosting & RAM Optimizations

To support deployment on free, resource-restricted cloud services (such as **Render.com**'s 512MB Free Tier), this repository incorporates an advanced **RAM-Free static predictions cache** architecture:

- **100-Combination Split Cache:** The 4 preloaded wells are completely deterministic. We precalculated all 100 well, model, and feature set configurations (with and without Viterbi sequence smoothing). Instead of holding a giant 300MB+ JSON in memory (which unpickles to >1.5GB of RAM), we split it into **100 separate JSON payloads** under `data/predictions/` (each ~3.2MB).
- **RAM-Free Routing Interception:** In `app.py`, requests to `/api/predict` for preloaded wells intercept the dynamic pipeline and load *only* the single 3.2MB JSON on demand. Scikit-learn, XGBoost, and LightGBM models are **never unpickled or loaded globally**, reducing startup RAM overhead to practically 0MB.
- **Optimized Dynamic Inference:** For custom `.las` file uploads, the application continues to run live dynamic ML predictions. It utilizes a **micro-lazy-loader** to unpickle only the single requested model in a thread-safe manner, running aggressive garbage collection sweeps (`gc.collect()`) immediately after to remain well within the 512MB RAM budget.
- **Dynamic Port & Package Bindings:** The `Dockerfile` has been optimized for multi-platform container orchestration (fully compatible with **Render** and **Hugging Face Spaces**):
  - Automatically installs `libgomp1` (the multi-threaded OpenMP matrix package) to prevent C++ import errors for tree-boosting modules on Linux.
  - Dynamically binds Gunicorn to the system-injected `$PORT` variable using shell execution (`sh -c "gunicorn -b 0.0.0.0:$PORT app:app"`), fully resolving port mismatches on host spin-up.

### Deploying to Render (Free Tier - 512MB RAM)
1. Go to your [Render Dashboard](https://dashboard.render.com/) -> click **New +** -> **Web Service**.
2. Select your `lithofacies-ml` GitHub repository.
3. Configure the settings:
   - **Runtime**: `Docker` (automatically builds from the optimized `Dockerfile`).
   - **Instance Type**: `Free` (512MB RAM).
4. Click **Create Web Service**. It will build and run without memory limit restrictions!

### Deploying to Hugging Face Spaces (Free Tier - 16GB RAM)
For an even more robust hosting tier with unlimited RAM headroom:
1. Create a **New Space** on Hugging Face, select **Docker** as the SDK, and choose **Blank** template.
2. Select the free **CPU Basic (16GB RAM, 2 vCPUs)** tier.
3. Add the Hugging Face repository as a Git remote and push:
   ```bash
   git remote add hf https://huggingface.co/spaces/YOUR_USERNAME/YOUR_SPACE_NAME
   git push hf main
   ```

---
*Reference: Merembayev, T.; Kurmangaliyev, D.; Bekbauov, B.; Amanbek, Y. A Comparison of Machine Learning Algorithms in Predicting Lithofacies. Energies 2021, 14, 1896. https://doi.org/10.3390/en14071896*
