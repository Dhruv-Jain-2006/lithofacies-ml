import os
import sys
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

# Add current directory and src to the path
sys.path.append(os.path.abspath('.'))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Make sure matplotlib runs headless
plt.switch_backend('Agg')

def run_pipeline():
    logger.info("==============================================")
    logger.info("STARTING LITHOFACIES ML PIPELINE VERIFICATION")
    logger.info("==============================================")
    
    # ---------------------------------------------------------
    # PHASE 1: DIRECTORIES & DATA LOAD/GENERATION
    # ---------------------------------------------------------
    logger.info("\n--- PHASE 1 & 2: DATA ACQUISITION & EDA ---")
    from src import data_loader
    data_loader.main()
    
    # ---------------------------------------------------------
    # INTERACTIVE NOTEBOOKS SETUP
    # ---------------------------------------------------------
    logger.info("\n--- GENERATING INTERACTIVE NOTEBOOKS ---")
    from src import generate_notebooks
    generate_notebooks.main()
    
    # ---------------------------------------------------------
    # PHASE 3 & 4: FEATURE ENGINEERING & DATA PREPARATION
    # ---------------------------------------------------------
    logger.info("\n--- PHASE 3 & 4: DATA PREPARATION & FEATURES ---")
    from src.features import handle_missing_values, compute_cwt_features, scale_features
    
    # Load raw merged parquet
    raw_df = pd.read_parquet('data/processed/merged_raw.parquet')
    logger.info(f"Loaded merged raw dataset: {raw_df.shape}")
    
    # Define columns
    original_cols = ['DEPTH_MD', 'CALI', 'RSHA', 'RMED', 'RDEP', 'RHOB', 'GR', 'NPHI', 'PEF', 'DTC', 'SP', 'BS']
    wavelet_logs = ['GR', 'NPHI', 'SP', 'RDEP', 'RHOB', 'DTC', 'PEF']
    wavelet_cols = original_cols + [f"{col}_CWT" for col in wavelet_logs]
    
    # Impute missing values
    clean_df = handle_missing_values(raw_df, original_cols + ['LITHOLOGY'])
    
    # Compute Continuous Wavelet Transform
    features_df = compute_cwt_features(clean_df, wavelet_logs)
    
    # Save engineered dataset
    features_df.to_parquet('data/interim/processed_features.parquet', index=False)
    logger.info("Preprocessed features successfully saved to data/interim/processed_features.parquet")
    
    # Train / Test split by well (to prevent spatial leakage)
    wells = features_df['WELL_ID'].unique()
    train_wells = wells[:8]
    test_wells = wells[8:]
    logger.info(f"Train wells (80%): {train_wells.tolist()}")
    logger.info(f"Test wells (20%): {test_wells.tolist()}")
    
    df_train = features_df[features_df['WELL_ID'].isin(train_wells)].copy()
    df_test = features_df[features_df['WELL_ID'].isin(test_wells)].copy()
    
    y_train = df_train['LITHOLOGY'].values
    y_test = df_test['LITHOLOGY'].values
    groups = df_train['WELL_ID'].values
    
    # Normalize features
    X_train_orig_scaled, X_test_orig_scaled, _ = scale_features(df_train, df_test, original_cols)
    X_train_wav_scaled, X_test_wav_scaled, _ = scale_features(df_train, df_test, wavelet_cols)
    
    # ---------------------------------------------------------
    # PHASE 5: MODEL TRAINING
    # ---------------------------------------------------------
    logger.info("\n--- PHASE 5: MODEL TRAINING & HYPERPARAMETER TUNING ---")
    from src.models import train_and_save_all, load_models
    
    # Train standard models
    models_orig = train_and_save_all(
        X_train_scaled=X_train_orig_scaled,
        X_train_raw=df_train[original_cols],
        y_train=y_train,
        groups=groups,
        feature_set_name="original"
    )
    
    # Train wavelet-transformed models
    models_wav = train_and_save_all(
        X_train_scaled=X_train_wav_scaled,
        X_train_raw=df_train[wavelet_cols],
        y_train=y_train,
        groups=groups,
        feature_set_name="wavelet"
    )
    
    # ---------------------------------------------------------
    # PHASE 6: PERFORMANCE EVALUATION
    # ---------------------------------------------------------
    logger.info("\n--- PHASE 6: EVALUATION & RESULTS ---")
    from src.metrics import get_classification_metrics, load_penalty_matrix
    
    penalty_mat = load_penalty_matrix()
    comparison_results = []
    
    for model_name in ['KNN', 'Decision Tree', 'Random Forest', 'XGBoost', 'LightGBM']:
        # Eval 12-feature
        m_orig = models_orig[model_name]
        X_test_f = X_test_orig_scaled if model_name == 'KNN' else df_test[original_cols]
        y_pred_orig = m_orig.predict(X_test_f)
        met_orig = get_classification_metrics(y_test, y_pred_orig, penalty_mat)
        
        # Eval 19-feature
        m_wav = models_wav[model_name]
        X_test_f_wav = X_test_wav_scaled if model_name == 'KNN' else df_test[wavelet_cols]
        y_pred_wav = m_wav.predict(X_test_f_wav)
        met_wav = get_classification_metrics(y_test, y_pred_wav, penalty_mat)
        
        comparison_results.append({
            "Model": model_name,
            "Accuracy (12)": met_orig["Accuracy"],
            "Penalty (12)": met_orig["PenaltyScore"],
            "Hamming (12)": met_orig["HammingLoss"],
            "Accuracy (19)": met_wav["Accuracy"],
            "Penalty (19)": met_wav["PenaltyScore"],
            "Hamming (19)": met_wav["HammingLoss"]
        })
        
    df_comparison = pd.DataFrame(comparison_results)
    
    logger.info("\n=======================================================")
    logger.info("REPLICATED COMPARATIVE RESULTS TABLE (Table 6)")
    logger.info("=======================================================")
    print(df_comparison.to_string(index=False))
    logger.info("=======================================================\n")
    
    # Save confusion matrix for best model (Random Forest, 19 features)
    rf_best = models_wav["Random Forest"]
    y_pred_rf = rf_best.predict(df_test[wavelet_cols])
    
    lithology_labels = [
        'Sandstone', 'Sandstone/Shale', 'Shale', 'Marl', 'Dolomite', 'Limestone',
        'Chalk', 'Halite', 'Anhydrite', 'Tuff', 'Coal', 'Basement'
    ]
    
    cm = confusion_matrix(y_test, y_pred_rf, labels=range(12))
    
    os.makedirs('plots', exist_ok=True)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=lithology_labels, yticklabels=lithology_labels)
    plt.xlabel('Predicted Lithology')
    plt.ylabel('True Lithology')
    plt.title('Confusion Matrix Heatmap — Best Model (Random Forest)', fontsize=14, pad=15)
    plt.tight_layout()
    plt.savefig('plots/best_model_confusion_matrix.png', dpi=150)
    plt.close()
    
    logger.info("Best model (Random Forest) confusion matrix plot successfully saved to plots/best_model_confusion_matrix.png.")
    
    # Print per-class classification report
    logger.info("\n=======================================================")
    logger.info("BEST MODEL CLASSIFICATION REPORT (Table 7)")
    logger.info("=======================================================")
    print(classification_report(y_test, y_pred_rf, labels=range(12), target_names=lithology_labels, zero_division=0))
    logger.info("=======================================================\n")
    
    # ---------------------------------------------------------
    # PHASE 7: SHAP EXPLAINABILITY
    # ---------------------------------------------------------
    logger.info("\n--- PHASE 7: SHAP INTERPRETABILITY ---")
    from src.explain import compute_shap_explanations, plot_global_importance, plot_per_class_beeswarm
    
    # Compute SHAP on a downsampled subset of the test data (for fast & robust run)
    explainer, shap_vals, X_samp = compute_shap_explanations(rf_best, df_test[wavelet_cols], num_samples=200)
    
    # Global multi-class stacked bar chart (Figure 6)
    plot_global_importance(shap_vals, X_samp, lithology_labels, 'plots/shap_global_importance.png')
    
    # Per-class beeswarm plots (Figure 7 panels a-l)
    plot_per_class_beeswarm(shap_vals, X_samp, lithology_labels, 'plots')
    
    logger.info("==============================================")
    logger.info("LITHOFACIES ML PIPELINE RUN SUCCESSFULLY!")
    logger.info("==============================================")

if __name__ == "__main__":
    run_pipeline()
