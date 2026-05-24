import os
import logging
import nbformat as nbf
nbf.new_notebook = nbf.v4.new_notebook
nbf.new_markdown_cell = nbf.v4.new_markdown_cell
nbf.new_code_cell = nbf.v4.new_code_cell

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def generate_eda_notebook():
    nb = nbf.new_notebook()
    nb['cells'] = [
        nbf.new_markdown_cell(
            "# Phase 2: Exploratory Data Analysis (EDA)\n\n"
            "This notebook loads the FORCE 2020 Norwegian Sea well log dataset, visualizes missingness patterns "
            "across well curves, plots the class distributions in log scale to observe sample imbalances, "
            "visualizes depth-based logs on a multi-track display, and produces statistical summaries."
        ),
        nbf.new_code_cell(
            "import os\n"
            "import sys\n"
            "import numpy as np\n"
            "import pandas as pd\n"
            "import matplotlib.pyplot as plt\n"
            "import seaborn as sns\n"
            "import missingno as msno\n\n"
            "# Add src to path\n"
            "sys.path.append(os.path.abspath('../'))\n"
            "from src.data_loader import generate_synthetic_las_files, load_las_dataset\n\n"
            "# 1. Initialize folders and data loader\n"
            "os.makedirs('../data/raw', exist_ok=True)\n"
            "os.makedirs('../plots', exist_ok=True)\n"
            "generate_synthetic_las_files('../data/raw', num_wells=10)\n"
            "df = load_las_dataset('../data/raw')\n"
            "print(f'Merged data shape: {df.shape}')\n"
            "df.head()"
        ),
        nbf.new_markdown_cell(
            "## 1. Visualizing Missing Data\n\n"
            "Geophysical well logs frequently contain gaps. We use the `missingno` library to map the missing "
            "data density across curves."
        ),
        nbf.new_code_cell(
            "plt.figure(figsize=(12, 6))\n"
            "msno.matrix(df.drop(columns=['WELL_ID', 'DEPTH_MD', 'LITHOLOGY'], errors='ignore'), sparkline=False)\n"
            "plt.title('FORCE 2020 Raw Well Logs Missingness Map', fontsize=16)\n"
            "plt.tight_layout()\n"
            "plt.savefig('../plots/missingno_matrix.png', dpi=150)\n"
            "plt.show()"
        ),
        nbf.new_markdown_cell(
            "## 2. Lithofacies Class Distribution\n\n"
            "Here we plot the class distributions on a logarithmic scale (replicating Figure 4 from the paper) "
            "to show class imbalance."
        ),
        nbf.new_code_cell(
            "lithology_labels = {\n"
            "    0: 'Sandstone', 1: 'Sandstone/Shale', 2: 'Shale', 3: 'Marl', \n"
            "    4: 'Dolomite', 5: 'Limestone', 6: 'Chalk', 7: 'Halite', \n"
            "    8: 'Anhydrite', 9: 'Tuff', 10: 'Coal', 11: 'Basement'\n"
            "}\n\n"
            "counts = df['LITHOLOGY'].value_counts().sort_index()\n"
            "labels = [lithology_labels.get(i, f'Class {i}') for i in counts.index]\n\n"
            "plt.figure(figsize=(10, 5))\n"
            "sns.barplot(x=labels, y=counts.values, palette='viridis', hue=labels, legend=False)\n"
            "plt.yscale('log')\n"
            "plt.xticks(rotation=45, ha='right')\n"
            "plt.xlabel('Lithofacies Class')\n"
            "plt.ylabel('Sample Count (Log Scale)')\n"
            "plt.title('FORCE 2020 Lithofacies Class Distribution (Norway Set)', fontsize=14)\n"
            "plt.grid(axis='y', which='both', linestyle='--', alpha=0.5)\n"
            "plt.tight_layout()\n"
            "plt.savefig('../plots/class_distribution.png', dpi=150)\n"
            "plt.show()\n\n"
            "for i, count in zip(counts.index, counts.values):\n"
            "    print(f'Class {i:2d} ({lithology_labels[i]:16s}): {count:6d} samples ({count/len(df)*100:5.2f}%)')"
        ),
        nbf.new_markdown_cell(
            "## 3. Well Log Track Viewer\n\n"
            "A standard geological display plotting key sensor curves (GR, RHOB, NPHI, RMED, DTC) against depth "
            "(replicating Figure 3 from the paper)."
        ),
        nbf.new_code_cell(
            "well_name = df['WELL_ID'].unique()[0]\n"
            "well_df = df[df['WELL_ID'] == well_name].sort_values('DEPTH_MD')\n\n"
            "fig, axes = plt.subplots(1, 5, figsize=(14, 10), sharey=True)\n"
            "fig.suptitle(f'Well Log Tracks for Well: {well_name}', fontsize=16, y=1.02)\n\n"
            "depth = well_df['DEPTH_MD'].values\n\n"
            "# Track 1: Gamma Ray\n"
            "axes[0].plot(well_df['GR'].values, depth, color='green', lw=1.5)\n"
            "axes[0].set_xlabel('GR (API)', color='green')\n"
            "axes[0].set_title('Gamma Ray')\n"
            "axes[0].grid(True)\n"
            "axes[0].tick_params(axis='x', labelcolor='green')\n\n"
            "# Track 2: Density (RHOB)\n"
            "axes[1].plot(well_df['RHOB'].values, depth, color='red', lw=1.5)\n"
            "axes[1].set_xlabel('RHOB (g/cm3)', color='red')\n"
            "axes[1].set_title('Density')\n"
            "axes[1].grid(True)\n"
            "axes[1].tick_params(axis='x', labelcolor='red')\n\n"
            "# Track 3: Neutron Porosity (NPHI)\n"
            "axes[2].plot(well_df['NPHI'].values, depth, color='blue', lw=1.5)\n"
            "axes[2].set_xlabel('NPHI (v/v)', color='blue')\n"
            "axes[2].set_title('Neutron Porosity')\n"
            "axes[2].grid(True)\n"
            "axes[2].tick_params(axis='x', labelcolor='blue')\n\n"
            "# Track 4: Resistivity (RMED)\n"
            "axes[3].plot(well_df['RMED'].values, depth, color='purple', lw=1.5)\n"
            "axes[3].set_xscale('log')\n"
            "axes[3].set_xlabel('RMED (ohm.m)', color='purple')\n"
            "axes[3].set_title('Resistivity')\n"
            "axes[3].grid(True)\n"
            "axes[3].tick_params(axis='x', labelcolor='purple')\n\n"
            "# Track 5: Sonic (DTC)\n"
            "axes[4].plot(well_df['DTC'].values, depth, color='darkorange', lw=1.5)\n"
            "axes[4].set_xlabel('DTC (us/ft)', color='darkorange')\n"
            "axes[4].set_title('Sonic Transit Time')\n"
            "axes[4].grid(True)\n"
            "axes[4].tick_params(axis='x', labelcolor='darkorange')\n\n"
            "# Set vertical axis parameters\n"
            "axes[0].set_ylabel('Measured Depth (m)', fontsize=12)\n"
            "axes[0].invert_yaxis()\n\n"
            "plt.tight_layout()\n"
            "plt.savefig(f'../plots/well_tracks_{well_name}.png', dpi=150, bbox_inches='tight')\n"
            "plt.show()"
        ),
        nbf.new_markdown_cell(
            "## 4. Statistical Summary\n\n"
            "Computes general statistcal descriptors (mean, std, min, max, quartiles) for the 12 selected logs "
            "(replicating Table 2 from the paper)."
        ),
        nbf.new_code_cell(
            "target_logs = ['DEPTH_MD', 'CALI', 'RSHA', 'RMED', 'RDEP', 'RHOB', 'GR', 'NPHI', 'PEF', 'DTC', 'SP', 'BS']\n"
            "summary_stats = df[target_logs].describe().T\n"
            "summary_stats = summary_stats[['mean', 'std', 'min', '25%', '50%', '75%', 'max']]\n"
            "summary_stats.columns = ['Mean', 'Std Dev', 'Min', '25% Q', 'Median', '75% Q', 'Max']\n"
            "print('Well Logs Statistical Summary (Norway Dataset Replicated Table 2):')\n"
            "summary_stats"
        )
    ]
    nbf.write(nb, 'notebooks/01_eda.ipynb')
    logger.info("Generated notebooks/01_eda.ipynb")

def generate_fe_notebook():
    nb = nbf.new_notebook()
    nb['cells'] = [
        nbf.new_markdown_cell(
            "# Phase 4: Feature Engineering & Preprocessing\n\n"
            "This notebook loads the raw merged data, implements local median gap filling and large gap dropping, "
            "extracts Continuous Wavelet Transform (CWT) features using PyWavelets with the Ricker (`mexh`) wavelet, "
            "plots a CWT scalogram (replicating Figure 1b), and exports the preprocessed original and wavelet feature sets."
        ),
        nbf.new_code_cell(
            "import os\n"
            "import sys\n"
            "import numpy as np\n"
            "import pandas as pd\n"
            "import matplotlib.pyplot as plt\n"
            "import pywt\n\n"
            "sys.path.append(os.path.abspath('../'))\n"
            "from src.data_loader import load_las_dataset\n"
            "from src.features import handle_missing_values, compute_cwt_features\n\n"
            "# 1. Load merged dataset\n"
            "df = load_las_dataset('../data/raw')\n"
            "print(f'Raw dataset shape: {df.shape}')"
        ),
        nbf.new_markdown_cell(
            "## 1. Missing Value Imputation\n\n"
            "Small gaps (< 5 consecutive NaN samples) are imputed with each well's median value of that log. "
            "Large gaps or logs missing entirely are dropped."
        ),
        nbf.new_code_cell(
            "feature_cols = ['DEPTH_MD', 'CALI', 'RSHA', 'RMED', 'RDEP', 'RHOB', 'GR', 'NPHI', 'PEF', 'DTC', 'SP', 'BS']\n"
            "df_clean = handle_missing_values(df, feature_cols + ['LITHOLOGY'])\n"
            "print(f'Cleaned dataset shape: {df_clean.shape}')"
        ),
        nbf.new_markdown_cell(
            "## 2. Continuous Wavelet Transform (CWT) Feature Extraction\n\n"
            "Decomposes geological signal into local frequency components. We apply it to the 7 target logs per well."
        ),
        nbf.new_code_cell(
            "target_logs = ['GR', 'NPHI', 'SP', 'RDEP', 'RHOB', 'DTC', 'PEF']\n"
            "df_features = compute_cwt_features(df_clean, target_logs)\n"
            "print(f'Dataset with CWT shape: {df_features.shape}')\n"
            "df_features.head()"
        ),
        nbf.new_markdown_cell(
            "## 3. CWT Scalogram Visualization\n\n"
            "Plots the full 2D continuous wavelet decomposition coefficients over different scales for a selected well section "
            "(replicating Figure 1b from the paper)."
        ),
        nbf.new_code_cell(
            "well_name = df_features['WELL_ID'].unique()[0]\n"
            "well_df = df_features[df_features['WELL_ID'] == well_name].sort_values('DEPTH_MD')\n"
            "signal = well_df['GR'].values[:500]\n"
            "scales = np.arange(1, 31)\n\n"
            "coefs, freqs = pywt.cwt(signal, scales, 'mexh')\n\n"
            "plt.figure(figsize=(10, 6))\n"
            "plt.imshow(np.abs(coefs), extent=[0, len(signal), 1, 30], cmap='PRGn', aspect='auto',\n"
            "           vmax=abs(coefs).max(), vmin=-abs(coefs).max())\n"
            "plt.colorbar(label='CWT Coefficient Magnitude')\n"
            "plt.xlabel('Sample Index (0.1 m steps)')\n"
            "plt.ylabel('Wavelet Scale')\n"
            "plt.title(f'Continuous Wavelet Transform (CWT) Scalogram of GR (Well: {well_name})', fontsize=14)\n"
            "plt.tight_layout()\n"
            "plt.savefig('../plots/gr_cwt_scalogram.png', dpi=150)\n"
            "plt.show()"
        ),
        nbf.new_markdown_cell(
            "## 4. Exporting Interim Datasets\n\n"
            "Saves the engineered features as Parquet for training."
        ),
        nbf.new_code_cell(
            "os.makedirs('../data/interim', exist_ok=True)\n"
            "df_features.to_parquet('../data/interim/processed_features.parquet', index=False)\n"
            "print('Engineered features successfully saved to data/interim/processed_features.parquet!')"
        )
    ]
    nbf.write(nb, 'notebooks/02_feature_engineering.ipynb')
    logger.info("Generated notebooks/02_feature_engineering.ipynb")

def generate_training_notebook():
    nb = nbf.new_notebook()
    nb['cells'] = [
        nbf.new_markdown_cell(
            "# Phase 5: Model Training & CV Tuning\n\n"
            "This notebook loads the processed features parquet, performs a strict well-based split to avoid data leakage, "
            "normalizes inputs for distance-based classifiers (StandardScaler for KNN), trains and tunes "
            "five classifiers (KNN, Decision Tree, Random Forest, XGBoost, and LightGBM) on the 12-feature and 19-feature datasets, "
            "and saves the trained models."
        ),
        nbf.new_code_cell(
            "import os\n"
            "import sys\n"
            "import pandas as pd\n\n"
            "sys.path.append(os.path.abspath('../'))\n"
            "from src.features import scale_features\n"
            "from src.models import train_and_save_all\n\n"
            "# 1. Load engineered features\n"
            "df = pd.read_parquet('../data/interim/processed_features.parquet')\n"
            "print(f'Features DataFrame shape: {df.shape}')"
        ),
        nbf.new_markdown_cell(
            "## 1. Train / Test Split by Well\n\n"
            "To prevent data leakage from high spatial correlations in a single well, we split by `WELL_ID` "
            "rather than randomly shuffling rows. 8 wells go to train, and 2 wells are held out for testing."
        ),
        nbf.new_code_cell(
            "wells = df['WELL_ID'].unique()\n"
            "train_wells = wells[:8]\n"
            "test_wells = wells[8:]\n\n"
            "df_train = df[df['WELL_ID'].isin(train_wells)].copy()\n"
            "df_test = df[df['WELL_ID'].isin(test_wells)].copy()\n\n"
            "print(f'Train Wells: {train_wells}')\n"
            "print(f'Test Wells: {test_wells}')\n"
            "print(f'Train samples: {len(df_train)}, Test samples: {len(df_test)}')"
        ),
        nbf.new_markdown_cell(
            "## 2. Feature Sets Definitions\n\n"
            "Define original (12 features) vs wavelet-transformed (19 features) datasets."
        ),
        nbf.new_code_cell(
            "original_cols = ['DEPTH_MD', 'CALI', 'RSHA', 'RMED', 'RDEP', 'RHOB', 'GR', 'NPHI', 'PEF', 'DTC', 'SP', 'BS']\n"
            "wavelet_cols = original_cols + [f'{col}_CWT' for col in ['GR', 'NPHI', 'SP', 'RDEP', 'RHOB', 'DTC', 'PEF']]\n\n"
            "print(f'Original features count: {len(original_cols)}')\n"
            "print(f'Wavelet features count: {len(wavelet_cols)}')"
        ),
        nbf.new_markdown_cell(
            "## 3. Standard Scaling\n\n"
            "Distance-based classifiers like KNN require scaling. Tree models do not. "
            "We fit standard scalers strictly on train, transforming test."
        ),
        nbf.new_code_cell(
            "X_train_orig_scaled, X_test_orig_scaled, scaler_orig = scale_features(df_train, df_test, original_cols)\n"
            "X_train_wav_scaled, X_test_wav_scaled, scaler_wav = scale_features(df_train, df_test, wavelet_cols)\n"
            "print('Scalers fitted and transformed successfully.')"
        ),
        nbf.new_markdown_cell(
            "## 4. Train Models on Original 12-Feature Dataset\n\n"
            "Trains and saves KNN, DT, RF, XGB, and LGB. Tuning uses 5-fold GroupKFold CV."
        ),
        nbf.new_code_cell(
            "y_train = df_train['LITHOLOGY'].values\n"
            "groups = df_train['WELL_ID'].values\n\n"
            "models_original = train_and_save_all(\n"
            "    X_train_scaled=X_train_orig_scaled,\n"
            "    X_train_raw=df_train[original_cols],\n"
            "    y_train=y_train,\n"
            "    groups=groups,\n"
            "    feature_set_name='original'\n"
            ")"
        ),
        nbf.new_markdown_cell(
            "## 5. Train Models on Wavelet 19-Feature Dataset\n\n"
            "Trains and saves the five classifiers on the 19 features including PyWavelets CWT."
        ),
        nbf.new_code_cell(
            "models_wavelet = train_and_save_all(\n"
            "    X_train_scaled=X_train_wav_scaled,\n"
            "    X_train_raw=df_train[wavelet_cols],\n"
            "    y_train=y_train,\n"
            "    groups=groups,\n"
            "    feature_set_name='wavelet'\n"
            ")"
        )
    ]
    nbf.write(nb, 'notebooks/03_model_training.ipynb')
    logger.info("Generated notebooks/03_model_training.ipynb")

def generate_evaluation_notebook():
    nb = nbf.new_notebook()
    nb['cells'] = [
        nbf.new_markdown_cell(
            "# Phase 6: Performance Evaluation\n\n"
            "This notebook evaluates the 10 trained models using the three publication metrics: Jaccard Accuracy, "
            "Hamming Loss, and Penalty Matrix Score. It creates the comparative results table (replicating Table 6), "
            "plots a 12x12 confusion matrix heatmap for the best model (Random Forest), and prints a detailed "
            "per-class classification report (replicating Table 7)."
        ),
        nbf.new_code_cell(
            "import os\n"
            "import sys\n"
            "import pandas as pd\n"
            "import numpy as np\n"
            "import matplotlib.pyplot as plt\n"
            "import seaborn as sns\n"
            "from sklearn.metrics import classification_report, confusion_matrix\n\n"
            "sys.path.append(os.path.abspath('../'))\n"
            "from src.features import scale_features\n"
            "from src.models import load_models\n"
            "from src.metrics import get_classification_metrics, load_penalty_matrix\n\n"
            "# 1. Load datasets and perform identical split\n"
            "df = pd.read_parquet('../data/interim/processed_features.parquet')\n"
            "wells = df['WELL_ID'].unique()\n"
            "df_train = df[df['WELL_ID'].isin(wells[:8])]\n"
            "df_test = df[df['WELL_ID'].isin(wells[8:])]\n\n"
            "original_cols = ['DEPTH_MD', 'CALI', 'RSHA', 'RMED', 'RDEP', 'RHOB', 'GR', 'NPHI', 'PEF', 'DTC', 'SP', 'BS']\n"
            "wavelet_cols = original_cols + [f'{col}_CWT' for col in ['GR', 'NPHI', 'SP', 'RDEP', 'RHOB', 'DTC', 'PEF']]\n\n"
            "X_train_orig, X_test_orig, _ = scale_features(df_train, df_test, original_cols)\n"
            "X_train_wav, X_test_wav, _ = scale_features(df_train, df_test, wavelet_cols)\n\n"
            "y_test = df_test['LITHOLOGY'].values\n"
            "print(f'Test size: {len(y_test)}')"
        ),
        nbf.new_markdown_cell(
            "## 1. Replicating the Comparative Results Table\n\n"
            "Loads both original and wavelet models, runs predictions on the test set, and builds the 5x3x2 table "
            "(replicating Table 6 from the paper)."
        ),
        nbf.new_code_cell(
            "models_orig = load_models('original')\n"
            "models_wav = load_models('wavelet')\n"
            "penalty_mat = load_penalty_matrix()\n\n"
            "results = []\n\n"
            "for model_name in ['KNN', 'Decision Tree', 'Random Forest', 'XGBoost', 'LightGBM']:\n"
            "    # Eval 12 feature model\n"
            "    m_orig = models_orig[model_name]\n"
            "    X_test_feat = X_test_orig if model_name == 'KNN' else df_test[original_cols]\n"
            "    y_pred_orig = m_orig.predict(X_test_feat)\n"
            "    metrics_orig = get_classification_metrics(y_test, y_pred_orig, penalty_mat)\n"
            "    \n"
            "    # Eval 19 feature model\n"
            "    m_wav = models_wav[model_name]\n"
            "    X_test_feat_wav = X_test_wav if model_name == 'KNN' else df_test[wavelet_cols]\n"
            "    y_pred_wav = m_wav.predict(X_test_feat_wav)\n"
            "    metrics_wav = get_classification_metrics(y_test, y_pred_wav, penalty_mat)\n"
            "    \n"
            "    results.append({\n"
            "        'Model': model_name,\n"
            "        'Accuracy (12)': metrics_orig['Accuracy'],\n"
            "        'Penalty (12)': metrics_orig['PenaltyScore'],\n"
            "        'Hamming (12)': metrics_orig['HammingLoss'],\n"
            "        'Accuracy (19)': metrics_wav['Accuracy'],\n"
            "        'Penalty (19)': metrics_wav['PenaltyScore'],\n"
            "        'Hamming (19)': metrics_wav['HammingLoss']\n"
            "    })\n\n"
            "df_res = pd.DataFrame(results)\n"
            "print('Models Comparison Table (Norway Replicated Table 6):')\n"
            "df_res"
        ),
        nbf.new_markdown_cell(
            "## 2. 12x12 Confusion Matrix Heatmap\n\n"
            "Visualizes class predictions vs ground truth for the best performing model (Random Forest)."
        ),
        nbf.new_code_cell(
            "lithology_labels = [\n"
            "    'Sandstone', 'Sandstone/Shale', 'Shale', 'Marl', 'Dolomite', 'Limestone',\n"
            "    'Chalk', 'Halite', 'Anhydrite', 'Tuff', 'Coal', 'Basement'\n"
            "]\n\n"
            "# RF on 19 features prediction\n"
            "rf_model = models_wav['Random Forest']\n"
            "y_pred_rf = rf_model.predict(df_test[wavelet_cols])\n\n"
            "cm = confusion_matrix(y_test, y_pred_rf, labels=range(12))\n\n"
            "plt.figure(figsize=(12, 10))\n"
            "sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=lithology_labels, yticklabels=lithology_labels)\n"
            "plt.xlabel('Predicted Lithology')\n"
            "plt.ylabel('True Lithology')\n"
            "plt.title('Confusion Matrix Heatmap — Best Model (Random Forest)', fontsize=14, pad=15)\n"
            "plt.tight_layout()\n"
            "plt.savefig('../plots/best_model_confusion_matrix.png', dpi=150)\n"
            "plt.show()"
        ),
        nbf.new_markdown_cell(
            "## 3. Classification Report\n\n"
            "Generates precision, recall, F1, and sample support for each of the 12 lithologies (replicating Table 7 from the paper)."
        ),
        nbf.new_code_cell(
            "print('Per-Class Classification Report for Best Model (Random Forest 19 features):')\n"
            "print(classification_report(y_test, y_pred_rf, labels=range(12), target_names=lithology_labels, zero_division=0))"
        )
    ]
    nbf.write(nb, 'notebooks/04_evaluation.ipynb')
    logger.info("Generated notebooks/04_evaluation.ipynb")

def generate_shap_notebook():
    nb = nbf.new_notebook()
    nb['cells'] = [
        nbf.new_markdown_cell(
            "# Phase 7: SHAP Explainability\n\n"
            "This notebook loads the best-performing Random Forest model and uses SHAP (Shapley Additive Explanations) "
            "to explain feature contributions. It replicates the global multi-class feature importance stacked bar chart (Figure 6) "
            "and generates beeswarm summary plots for all 12 lithofacies classes (Figure 7 panels a-l)."
        ),
        nbf.new_code_cell(
            "import os\n"
            "import sys\n"
            "import pandas as pd\n"
            "import matplotlib.pyplot as plt\n\n"
            "sys.path.append(os.path.abspath('../'))\n"
            "from src.models import load_models\n"
            "from src.explain import compute_shap_explanations, plot_global_importance, plot_per_class_beeswarm\n\n"
            "# 1. Load test data and the best model\n"
            "df = pd.read_parquet('../data/interim/processed_features.parquet')\n"
            "wells = df['WELL_ID'].unique()\n"
            "df_test = df[df['WELL_ID'].isin(wells[8:])]\n\n"
            "wavelet_cols = ['DEPTH_MD', 'CALI', 'RSHA', 'RMED', 'RDEP', 'RHOB', 'GR', 'NPHI', 'PEF', 'DTC', 'SP', 'BS'] \\\n"
            "               + [f'{col}_CWT' for col in ['GR', 'NPHI', 'SP', 'RDEP', 'RHOB', 'DTC', 'PEF']]\n\n"
            "models_wav = load_models('wavelet')\n"
            "rf_model = models_wav['Random Forest']\n"
            "print('Model and data successfully loaded.')"
        ),
        nbf.new_markdown_cell(
            "## 1. Compute SHAP Values\n\n"
            "We initialize a TreeExplainer and calculate SHAP contributions on 200 representative test samples to optimize execution speed."
        ),
        nbf.new_code_cell(
            "explainer, shap_values, X_sample = compute_shap_explanations(rf_model, df_test[wavelet_cols], num_samples=200)\n"
            "print('SHAP values computed.')"
        ),
        nbf.new_markdown_cell(
            "## 2. Replicating Global Feature Importance Stacked Bar Chart\n\n"
            "Visualizes how each of the 19 features contributes across all 12 classes (replicating Figure 6 from the paper)."
        ),
        nbf.new_code_cell(
            "lithology_labels = [\n"
            "    'Sandstone', 'Sandstone/Shale', 'Shale', 'Marl', 'Dolomite', 'Limestone',\n"
            "    'Chalk', 'Halite', 'Anhydrite', 'Tuff', 'Coal', 'Basement'\n"
            "]\n\n"
            "plot_global_importance(shap_values, X_sample, lithology_labels, '../plots/shap_global_importance.png')\n\n"
            "# Display generated image\n"
            "from IPython.display import Image\n"
            "Image(filename='../plots/shap_global_importance.png')"
        ),
        nbf.new_markdown_cell(
            "## 3. Replicating Per-Class Beeswarm Plots\n\n"
            "Generates 12 beeswarm plots (replicating panels a-l in Figure 7) showing feature value effects on prediction score."
        ),
        nbf.new_code_cell(
            "plot_per_class_beeswarm(shap_values, X_sample, lithology_labels, '../plots')\n"
            "print('All 12 beeswarm plots saved in the plots/ folder!')"
        )
    ]
    nbf.write(nb, 'notebooks/05_shap_analysis.ipynb')
    logger.info("Generated notebooks/05_shap_analysis.ipynb")

def main():
    os.makedirs('notebooks', exist_ok=True)
    generate_eda_notebook()
    generate_fe_notebook()
    generate_training_notebook()
    generate_evaluation_notebook()
    generate_shap_notebook()
    logger.info("All notebooks successfully generated!")

if __name__ == '__main__':
    main()
