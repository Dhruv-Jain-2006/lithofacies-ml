import os
import sys
import logging
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score, hamming_loss, precision_score, recall_score
from sklearn.model_selection import GroupShuffleSplit, GroupKFold
from sklearn.inspection import permutation_importance
from sklearn.feature_selection import mutual_info_classif, RFECV
from sklearn.calibration import calibration_curve
from sklearn.tree import DecisionTreeClassifier

# Add current directory and src to the path
sys.path.append(os.path.abspath('.'))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Make sure matplotlib runs headless
plt.switch_backend('Agg')

def calculate_entropy(probs):
    """
    Computes Shannon entropy in log2 space for a probability distribution.
    """
    probs = np.clip(probs, 1e-15, 1.0)
    return -np.sum(probs * np.log2(probs), axis=1)

def calculate_ece(y_true, y_probs, n_bins=10):
    """
    Computes the Expected Calibration Error (ECE) for multi-class predictions.
    """
    y_true = np.array(y_true)
    y_probs = np.array(y_probs)
    
    T = len(y_true)
    confidences = np.max(y_probs, axis=1)
    predictions = np.argmax(y_probs, axis=1)
    accuracies = (predictions == y_true)
    
    ece = 0.0
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    
    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]
        
        # Select samples that fall within this confidence bin
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop_in_bin = np.mean(in_bin)
        
        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(accuracies[in_bin])
            confidence_in_bin = np.mean(confidences[in_bin])
            ece += prop_in_bin * np.abs(avg_confidence_diff := (accuracy_in_bin - confidence_in_bin))
            
    return ece

def run_pipeline():
    logger.info("==============================================")
    logger.info("STARTING FINAL GEOLOGICAL ML PIPELINE RUN (PROMPTS 4 & 5)")
    logger.info("==============================================")
    
    os.makedirs('plots', exist_ok=True)
    os.makedirs('reports', exist_ok=True)
    os.makedirs('data/models', exist_ok=True)
    
    # ---------------------------------------------------------
    # DATA LOADING & PREPROCESSING
    # ---------------------------------------------------------
    logger.info("\n--- PHASE 1 & 2: DATA ACQUISITION & EDA ---")
    from src import data_loader
    data_loader.main()
    
    from src.features import (
        calculate_relative_depth, 
        engineer_advanced_geological_features, 
        compute_cwt_features, 
        impute_missing_features, 
        scale_features_and_save,
        validate_engineered_features,
        correlation_pruning
    )
    
    # Load raw merged parquet
    raw_df = pd.read_parquet('data/processed/merged_raw.parquet')
    logger.info(f"Loaded merged raw dataset: {raw_df.shape}")
    
    # Sort and compute features
    logger.info("Running advanced geological features engine...")
    features_df = calculate_relative_depth(raw_df)
    features_df = engineer_advanced_geological_features(features_df)
    
    wavelet_logs = ['GR', 'NPHI', 'SP', 'RDEP', 'RHOB', 'DTC', 'PEF']
    features_df = compute_cwt_features(features_df, wavelet_logs)
    
    # Save preprocessed features parquet
    features_df.to_parquet('data/interim/processed_features.parquet', index=False)
    logger.info("Advanced engineered features successfully saved.")
    
    # ---------------------------------------------------------
    # GROUP-WISE SPLIT (Strictly Training-Only Augmentation)
    # ---------------------------------------------------------
    logger.info("\n--- ENFORCING GROUP-WISE SPLIT ---")
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    real_df = features_df[~features_df['IS_SYNTHETIC']].reset_index(drop=True)
    train_idx, test_idx = next(gss.split(real_df, groups=real_df['WELL']))
    
    df_train_base = pd.concat([real_df.iloc[train_idx], features_df[features_df['IS_SYNTHETIC']]], ignore_index=True)
    df_test = real_df.iloc[test_idx].reset_index(drop=True)
    
    logger.info(f"Base Training split size: {len(df_train_base)}")
    logger.info(f"Testing split size: {len(df_test)}")
    
    # Define Column Sets
    original_cols_with_depth = ['RELATIVE_DEPTH', 'CALI', 'RSHA', 'RMED', 'RDEP', 'RHOB', 'GR', 'NPHI', 'PEF', 'DTC', 'SP', 'BS']
    original_cols_no_depth = ['CALI', 'RSHA', 'RMED', 'RDEP', 'RHOB', 'GR', 'NPHI', 'PEF', 'DTC', 'SP', 'BS']
    
    # Engineered petrophysical features
    petro_cols = ['RHOB_minus_NPHI', 'RDEP_RMED_ratio', 'RMED_RSHA_ratio', 'acoustic_impedance', 'log_RDEP', 'log_RMED', 'log_RSHA']
    roll_cols = [
        'GR_roll_mean', 'RHOB_roll_mean', 'NPHI_roll_mean', 'DTC_roll_mean',
        'GR_roll_std', 'RHOB_roll_std', 'NPHI_roll_std', 'DTC_roll_std',
        'GR_roll_var', 'RHOB_roll_var',
        'GR_gradient', 'RHOB_gradient', 'DTC_gradient'
    ]
    cwt_band_cols = []
    for log in wavelet_logs:
        cwt_band_cols.extend([f"{log}_CWT_low", f"{log}_CWT_mid", f"{log}_CWT_high"])
        
    engineered_cols = petro_cols + roll_cols + cwt_band_cols
    combined_cols_with_depth = original_cols_with_depth + engineered_cols
    combined_cols_no_depth = original_cols_no_depth + engineered_cols
    
    y_test = df_test['LITHOLOGY'].values
    
    # Run Quality Control scan on Combined Feature Set
    validate_engineered_features(features_df, combined_cols_with_depth)
    
    # ---------------------------------------------------------
    # PHASE 1: TARGETED RARE-FACIES AUGMENTATION BENCHMARKING
    # ---------------------------------------------------------
    logger.info("\n--- PHASE 1: TARGETED RARE-FACIES AUGMENTATION & BENCHMARKING ---")
    from src.augmentation import augment_rare_facies
    
    ratios = [0.0, 0.05, 0.10, 0.20]
    benchmark_results = []
    
    logger.info("Executing Synthetic Contribution Ratio Benchmarking (0%, 5%, 10%, 20%)...")
    for ratio in ratios:
        logger.info(f"Evaluating Augmentation Contribution Ratio: {ratio:.0%}")
        df_train_temp = augment_rare_facies(df_train_base, ratio=ratio, random_state=42)
        
        # Apply median imputation and scaling
        df_train_imp, df_test_imp, _ = impute_missing_features(df_train_temp, df_test, combined_cols_with_depth)
        X_train_scaled, X_test_scaled, _ = scale_features_and_save(
            df_train_imp, df_test_imp, combined_cols_with_depth, 'data/models/scaler_temp.pkl'
        )
        
        # Train a rapid LightGBM model to evaluate this ratio
        from lightgbm import LGBMClassifier
        from sklearn.calibration import CalibratedClassifierCV
        
        lgb_raw = LGBMClassifier(n_estimators=50, random_state=42, n_jobs=-1, class_weight='balanced')
        lgb_model = CalibratedClassifierCV(estimator=lgb_raw, method='isotonic', cv=3)
        lgb_model.fit(df_train_imp[combined_cols_with_depth], df_train_imp['LITHOLOGY'].values)
        
        # Evaluate on testing split (strictly real FORCE-derived wells)
        y_probs = lgb_model.predict_proba(df_test_imp[combined_cols_with_depth])
        y_pred = lgb_model.predict(df_test_imp[combined_cols_with_depth])
        
        # Macro F1
        f1 = f1_score(y_test, y_pred, average='macro')
        # ECE
        ece = calculate_ece(y_test, y_probs)
        # Rare facies recall (8, 9, 10, 11)
        cm = confusion_matrix(y_test, y_pred, labels=range(12))
        recalls = []
        for c in [8, 9, 10, 11]:
            denom = cm[c].sum()
            recalls.append(cm[c, c] / denom if denom > 0 else 0.0)
        rare_recall = np.mean(recalls)
        
        benchmark_results.append({
            "Ratio": f"{ratio:.0%}",
            "Macro F1": round(f1, 4),
            "ECE": round(ece, 4),
            "Rare-Facies Recall": round(rare_recall, 4)
        })
        
    df_bench = pd.DataFrame(benchmark_results)
    df_bench.to_csv('reports/synthetic_ratio_benchmark.csv', index=False)
    logger.info("Augmentation benchmark comparison completed:")
    print(df_bench.to_string(index=False))
    
    # We choose the optimal 10% ratio as final training augmentation
    logger.info("Applying final optimal 10% rare-facies augmentation...")
    df_train = augment_rare_facies(df_train_base, ratio=0.10, random_state=42)
    y_train = df_train['LITHOLOGY'].values
    groups = df_train['WELL'].values
    
    # ---------------------------------------------------------
    # COMPLEXITY REDUCTION & FEATURE PRUNING
    # ---------------------------------------------------------
    logger.info("\n--- CORRELATION PRUNING ---")
    pruned_cols_with_depth = correlation_pruning(df_train, combined_cols_with_depth, threshold=0.95)
    pruned_cols_no_depth = correlation_pruning(df_train, combined_cols_no_depth, threshold=0.95)
    
    # Save a frozen JSON manifest outlining active feature ordering and schema metadata
    import json
    feature_manifest = {
        "active_features_with_depth": pruned_cols_with_depth,
        "active_features_no_depth": pruned_cols_no_depth,
        "total_active_count": len(pruned_cols_with_depth),
        "correlation_pruning_threshold": 0.95
    }
    with open("data/models/feature_manifest.json", "w") as f:
        json.dump(feature_manifest, f, indent=4)
    logger.info("Frozen active feature_manifest.json exported successfully.")
    
    # Persist pruned feature names
    with open("data/models/features_wavelet.pkl", "wb") as f:
        pickle.dump(pruned_cols_with_depth, f)
        
    # Generate Heatmaps
    corr_p = df_train[pruned_cols_with_depth].corr(method='pearson')
    plt.figure(figsize=(14, 12))
    sns.heatmap(corr_p, cmap='coolwarm', vmin=-1, vmax=1, xticklabels=True, yticklabels=True)
    plt.title('Pearson Correlation Heatmap - Pruned Feature Space', fontsize=14, pad=15)
    plt.tight_layout()
    plt.savefig('plots/pearson_correlation.png', dpi=150)
    plt.close()
    
    # ---------------------------------------------------------
    # DYNAMIC SOFT-VOTING ENSEMBLE WEIGHT ESTIMATION
    # ---------------------------------------------------------
    logger.info("\n--- COMPUTING DYNAMIC VALIDATION METRICS FOR SOFT-VOTING ---")
    # Quick out-of-fold validation run using a GroupKFold on training set to compute weights
    from sklearn.model_selection import GroupKFold
    from src.models import tune_knn, tune_decision_tree, get_random_forest_model, get_xgboost_model, get_lightgbm_model
    
    gkf = GroupKFold(n_splits=3)
    train_idx, val_idx = next(gkf.split(df_train, y_train, groups=groups))
    
    df_tr_sub = df_train.iloc[train_idx]
    df_val_sub = df_train.iloc[val_idx]
    y_tr_sub = df_tr_sub['LITHOLOGY'].values
    y_val_sub = df_val_sub['LITHOLOGY'].values
    groups_sub = df_tr_sub['WELL'].values
    
    df_tr_sub_imp, df_val_sub_imp, _ = impute_missing_features(df_tr_sub, df_val_sub, pruned_cols_with_depth)
    X_tr_sc, X_val_sc, _ = scale_features_and_save(df_tr_sub_imp, df_val_sub_imp, pruned_cols_with_depth, 'data/models/scaler_sub.pkl')
    
    models_dict_sub = {
        "Decision Tree": CalibratedClassifierCV(estimator=tune_decision_tree(df_tr_sub_imp[pruned_cols_with_depth], y_tr_sub, groups_sub), method='isotonic', cv=3),
        "Random Forest": CalibratedClassifierCV(estimator=get_random_forest_model(), method='isotonic', cv=3),
        "XGBoost": CalibratedClassifierCV(estimator=get_xgboost_model(), method='isotonic', cv=3),
        "LightGBM": CalibratedClassifierCV(estimator=get_lightgbm_model(), method='isotonic', cv=3)
    }
    
    validation_metrics = {}
    for name, model in models_dict_sub.items():
        logger.info(f"Estimating dynamic weight for model: {name}...")
        model.fit(df_tr_sub_imp[pruned_cols_with_depth], y_tr_sub)
        y_probs_v = model.predict_proba(df_val_sub_imp[pruned_cols_with_depth])
        y_pred_v = model.predict(df_val_sub_imp[pruned_cols_with_depth])
        
        f1_v = f1_score(y_val_sub, y_pred_v, average='macro')
        ece_v = calculate_ece(y_val_sub, y_probs_v)
        
        cm_v = confusion_matrix(y_val_sub, y_pred_v, labels=range(12))
        recalls_v = []
        for c in [8, 9, 10, 11]:
            denom = cm_v[c].sum()
            recalls_v.append(cm_v[c, c] / denom if denom > 0 else 0.0)
        rare_rec_v = np.mean(recalls_v)
        
        validation_metrics[name] = {
            "macro_f1": f1_v,
            "ece": ece_v,
            "rare_recall": rare_rec_v
        }
        logger.info(f" -> {name} - Macro F1: {f1_v:.4f} | ECE: {ece_v:.4f} | Rare Recall: {rare_rec_v:.4f}")
        
    # ---------------------------------------------------------
    # SCIENTIFIC ABLATION BENCHMARK RUN
    # ---------------------------------------------------------
    logger.info("\n--- SCIENTIFIC ABLATION STUDY ---")
    ablation_results = []
    
    configs = [
        ("Original Only (A)", original_cols_with_depth, "original"),
        ("Original (No Depth)", original_cols_no_depth, "original_nodep"),
        ("Engineered Only (B)", engineered_cols, "engineered"),
        ("Combined Space (C)", pruned_cols_with_depth, "wavelet"),
        ("Combined (No Depth)", pruned_cols_no_depth, "wavelet_nodep")
    ]
    
    trained_models_dict = {}
    imputers_dict = {}
    
    from src.models import train_and_save_all
    from src.metrics import get_classification_metrics, load_penalty_matrix, generate_geological_confusion_report
    from src.sequence import estimate_transition_matrix, viterbi_decode
    
    penalty_mat = load_penalty_matrix()
    
    for config_name, feature_list, file_suffix in configs:
        logger.info(f"\nEvaluating Configuration: {config_name}")
        
        df_train_imp, df_test_imp, imputer = impute_missing_features(df_train, df_test, feature_list)
        imputers_dict[config_name] = imputer
        
        with open(f'data/models/features_{file_suffix}.pkl', 'wb') as f:
            pickle.dump(feature_list, f)
            
        X_train_scaled, X_test_scaled, scaler = scale_features_and_save(
            df_train_imp, df_test_imp, feature_list, f'data/models/scaler_{file_suffix}.pkl'
        )
        
        models = train_and_save_all(
            X_train_scaled=X_train_scaled,
            X_train_raw=df_train_imp[feature_list],
            y_train=y_train,
            groups=groups,
            feature_set_name=file_suffix,
            scaler=scaler,
            validation_metrics=validation_metrics if "Combined Space" in config_name or "Combined (No Depth)" in config_name else None
        )
        trained_models_dict[config_name] = (models, feature_list, df_train_imp, df_test_imp, X_train_scaled, X_test_scaled)
        
        # Save standard production scalers for app.py
        if config_name == "Original Only (A)":
            scale_features_and_save(df_train_imp, df_test_imp, feature_list, 'data/models/scaler_orig.pkl')
        elif config_name == "Combined Space (C)":
            scale_features_and_save(df_train_imp, df_test_imp, feature_list, 'data/models/scaler_wav.pkl')
        elif config_name == "Combined (No Depth)":
            scale_features_and_save(df_train_imp, df_test_imp, feature_list, 'data/models/scaler_wav_nodep.pkl')
            
        # Evaluate each model
        for m_name in ['KNN', 'Decision Tree', 'Random Forest', 'XGBoost', 'LightGBM', 'Ensemble']:
            model = models[m_name]
            
            if m_name == 'Ensemble':
                y_probs = model.predict_proba(df_test_imp[feature_list])
                y_pred = model.predict(df_test_imp[feature_list])
            else:
                X_test_input = X_test_scaled if m_name == 'KNN' else df_test_imp[feature_list]
                y_probs = model.predict_proba(X_test_input)
                y_pred = model.predict(X_test_input)
                
            acc = accuracy_score(y_test, y_pred)
            f1_macro = f1_score(y_test, y_pred, average='macro')
            f1_weighted = f1_score(y_test, y_pred, average='weighted')
            h_loss = hamming_loss(y_true=y_test, y_pred=y_pred)
            penalty_score = get_classification_metrics(y_test, y_pred, penalty_mat)["PenaltyScore"]
            ece_val = calculate_ece(y_test, y_probs)
            
            ablation_results.append({
                "Feature Set": config_name,
                "Model": m_name,
                "Accuracy": acc,
                "Macro F1": f1_macro,
                "Weighted F1": f1_weighted,
                "Hamming Loss": h_loss,
                "Penalty Score": penalty_score,
                "ECE": ece_val
            })
            
    df_ablation = pd.DataFrame(ablation_results)
    df_ablation.to_csv('plots/ablation_study_results.csv', index=False)
    logger.info("Ablation results successfully updated.")
    print(df_ablation[df_ablation['Model'] == 'Ensemble'].to_string(index=False))
    
    # ---------------------------------------------------------
    # ECE CURVES & RELIABILITY DIAGRAMS
    # ---------------------------------------------------------
    logger.info("\n--- RELIABILITY DIAGRAMS ---")
    plt.figure(figsize=(10, 8))
    models_c, feature_list_c, df_train_imp_c, df_test_imp_c, _, X_test_scaled_c = trained_models_dict["Combined Space (C)"]
    
    for m_name in ['Decision Tree', 'Random Forest', 'XGBoost', 'LightGBM', 'Ensemble']:
        model = models_c[m_name]
        if m_name == 'Ensemble':
            y_probs = model.predict_proba(df_test_imp_c[feature_list_c])
            y_pred = model.predict(df_test_imp_c[feature_list_c])
        else:
            y_probs = model.predict_proba(df_test_imp_c[feature_list_c])
            y_pred = model.predict(df_test_imp_c[feature_list_c])
            
        confidences = np.max(y_probs, axis=1)
        accuracies = (y_pred == y_test).astype(int)
        
        prob_true, prob_pred = calibration_curve(accuracies, confidences, n_bins=10)
        ece_val = calculate_ece(y_test, y_probs)
        plt.plot(prob_pred, prob_true, marker='s', label=f'{m_name} (ECE: {ece_val:.4f})', lw=2)
        
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfect Calibration')
    plt.xlabel('Mean Predicted Confidence')
    plt.ylabel('Empirical Accuracy')
    plt.title('Reliability Diagrams (Calibrated Classifiers on Combined Space)', fontsize=14, pad=15)
    plt.legend(loc='lower right')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig('plots/calibration_reliability_diagrams.png', dpi=150)
    plt.close()
    
    # ---------------------------------------------------------
    # GEOLOGICAL SEQUENCE SMOOTHING & VITERBI PENALIZATION
    # ---------------------------------------------------------
    logger.info("\n--- PHASE 4: GEOLOGICAL SEQUENCE SMOOTHING (VITERBI) ---")
    trans_matrix = estimate_transition_matrix(y_train, groups=groups)
    
    with open("data/models/transition_matrix.pkl", "wb") as f:
        pickle.dump(trans_matrix, f)
        
    ensemble_model = models_c["Ensemble"]
    y_probs_c = ensemble_model.predict_proba(df_test_imp_c[feature_list_c])
    y_pred_raw = ensemble_model.predict(df_test_imp_c[feature_list_c])
    
    y_pred_viterbi = np.zeros_like(y_pred_raw)
    group_test = df_test_imp_c['WELL'].values
    
    for well in np.unique(group_test):
        well_mask = group_test == well
        if np.sum(well_mask) == 0:
            continue
        well_probs = y_probs_c[well_mask]
        well_smoothed = viterbi_decode(well_probs, trans_matrix)
        y_pred_viterbi[well_mask] = well_smoothed
        
    acc_raw = accuracy_score(y_test, y_pred_raw)
    acc_vit = accuracy_score(y_test, y_pred_viterbi)
    pen_raw = get_classification_metrics(y_test, y_pred_raw, penalty_mat)["PenaltyScore"]
    pen_vit = get_classification_metrics(y_test, y_pred_viterbi, penalty_mat)["PenaltyScore"]
    h_raw = hamming_loss(y_test, y_pred_raw)
    h_vit = hamming_loss(y_test, y_pred_viterbi)
    
    logger.info("=== SEQUENCE MODELING BENCHMARK ===")
    logger.info(f"Raw Soft Voting Ensemble    -> Accuracy: {acc_raw:.4f} | Hamming Loss: {h_raw:.4f} | Penalty Score: {pen_raw:.4f}")
    logger.info(f"Viterbi-Smoothed Ensemble   -> Accuracy: {acc_vit:.4f} | Hamming Loss: {h_vit:.4f} | Penalty Score: {pen_vit:.4f}")
    logger.info("====================================")
    
    # Save sequence comparison plot
    test_well = np.unique(group_test)[0]
    well_mask = group_test == test_well
    depths = df_test_imp_c.loc[well_mask, 'DEPTH_MD'].values[:200]
    y_true_seg = y_test[well_mask][:200]
    y_raw_seg = y_pred_raw[well_mask][:200]
    y_vit_seg = y_pred_viterbi[well_mask][:200]
    
    plt.figure(figsize=(14, 5))
    plt.plot(depths, y_true_seg, label='True Lithology', color='black', lw=2, alpha=0.9)
    plt.step(depths, y_raw_seg, label='Raw Ensemble Prediction', color='red', linestyle='--', alpha=0.7)
    plt.step(depths, y_vit_seg, label='Viterbi Smoothed Profile', color='green', lw=2, alpha=0.8)
    plt.xlabel('Measured Depth (m)')
    plt.ylabel('Lithology Code (0-11)')
    plt.title(f'Stratigraphic Log Profile Comparison — Well {test_well}', fontsize=14, pad=15)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('plots/stratigraphic_sequence_profile.png', dpi=150)
    plt.close()
    
    # ---------------------------------------------------------
    # HIGH-ENTROPY INTERVALS & UNCERTAINTY LOGGING
    # ---------------------------------------------------------
    logger.info("\n--- ENSEMBLE GEOLOGICAL UNCERTAINTY LOGGING ---")
    entropies = calculate_entropy(y_probs_c)
    
    # Flag high uncertainty (entropy > 1.5)
    high_unc_mask = entropies > 1.5
    df_test_imp_c['Shannon_Entropy'] = entropies
    
    lithology_labels = [
        'Sandstone', 'Sandstone/Shale', 'Shale', 'Marl', 'Dolomite', 'Limestone',
        'Chalk', 'Halite', 'Anhydrite', 'Tuff', 'Coal', 'Basement'
    ]
    
    uncertainty_logs = []
    for idx in np.where(high_unc_mask)[0]:
        row = df_test_imp_c.iloc[idx]
        probs = y_probs_c[idx]
        top3_indices = np.argsort(probs)[-3:][::-1]
        
        candidates = []
        for rank, c_idx in enumerate(top3_indices):
            candidates.append(f"{lithology_labels[c_idx]} ({probs[c_idx]:.1%})")
            
        uncertainty_logs.append({
            "WELL": row['WELL'],
            "DEPTH_MD": round(row['DEPTH_MD'], 2),
            "True_Lithology": lithology_labels[int(y_test[idx])],
            "Ensemble_Prediction": lithology_labels[int(y_pred_raw[idx])],
            "Entropy_Score": round(entropies[idx], 4),
            "Top_3_Candidates": " | ".join(candidates)
        })
        
    df_unc = pd.DataFrame(uncertainty_logs)
    df_unc.to_csv('reports/uncertainty_intervals.csv', index=False)
    logger.info(f"Flagged {len(df_unc)} high-entropy geological intervals. Logs saved to reports/uncertainty_intervals.csv")
    
    # ---------------------------------------------------------
    # PHASE 5: VERTICAL DEPTH EXTRAPOLATION
    # ---------------------------------------------------------
    logger.info("\n--- PHASE 5: VERTICAL DEPTH EXTRAPOLATION VALIDATION ---")
    train_holdout_parts = []
    test_holdout_parts = []
    for well in np.unique(group_test):
        well_mask = group_test == well
        well_df = df_test_imp_c[well_mask].sort_values(by='DEPTH_MD')
        split_idx = int(len(well_df) * 0.65)  # Train on top 65% depth
        train_holdout_parts.append(well_df.iloc[:split_idx])
        test_holdout_parts.append(well_df.iloc[split_idx:])   # Blind-test on bottom 35%
        
    df_train_holdout = pd.concat([df_train_imp_c] + train_holdout_parts, ignore_index=True)
    df_test_holdout = pd.concat(test_holdout_parts, ignore_index=True)
    
    y_test_h = df_test_holdout['LITHOLOGY'].values
    
    # Re-evaluate best model (LGBM)
    best_lgb_model = models_c["LightGBM"]
    y_pred_h = best_lgb_model.predict(df_test_holdout[feature_list_c])
    
    logger.info("\n=== BLIND VERTICAL DEPTH ZONE EXTRAPOLATION (65% UP / 35% BOTTOM) ===")
    print(classification_report(y_test_h, y_pred_h, target_names=lithology_labels, labels=range(12), zero_division=0))
    
    generate_geological_confusion_report(y_test, y_pred_viterbi, penalty_mat, 'plots/geological_confusion_report.txt')
    
    # Save Confusion Matrices
    for name, key in [("Original_Only", "Original Only (A)"), ("Engineered_Only", "Engineered Only (B)"), ("Combined_Space", "Combined Space (C)")]:
        models, feature_list, _, df_test_imp, _, _ = trained_models_dict[key]
        lgb_model = models["LightGBM"]
        y_pred = lgb_model.predict(df_test_imp[feature_list])
        
        cm = confusion_matrix(y_test, y_pred, labels=range(12))
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', xticklabels=lithology_labels, yticklabels=lithology_labels)
        plt.xlabel('Predicted Lithology')
        plt.ylabel('True Lithology')
        plt.title(f'Confusion Matrix — LightGBM on {key.split(" (")[0]}', fontsize=14, pad=15)
        plt.tight_layout()
        plt.savefig(f'plots/confusion_matrix_{name.lower()}.png', dpi=150)
        plt.close()
        
    import shutil
    shutil.copy('plots/confusion_matrix_combined_space.png', 'plots/best_model_confusion_matrix.png')
    
    # ---------------------------------------------------------
    # PHASE 8: STRATIFIED SHAP DRIFT REPORT
    # ---------------------------------------------------------
    logger.info("\n--- STRATIFIED SHAP DRIFT & STABILITY REPORT ---")
    try:
        from src.explain import generate_shap_drift_report
        generate_shap_drift_report(trained_models_dict, y_test, 'reports/shap_drift_report.csv')
    except Exception as e:
        logger.error(f"Error generating SHAP drift report: {str(e)}")
        
    # ---------------------------------------------------------
    # FEATURE IMPORTANCES (MI & Permutation)
    # ---------------------------------------------------------
    df_train_sub = df_train_imp_c.sample(n=min(2000, len(df_train_imp_c)), random_state=42)
    mi_scores = mutual_info_classif(df_train_sub[feature_list_c], df_train_sub['LITHOLOGY'].values, random_state=42)
    mi_df = pd.DataFrame({"Feature": feature_list_c, "Mutual_Information": mi_scores}).sort_values(by='Mutual_Information', ascending=False)
    mi_df.to_csv('plots/mutual_information_rankings.csv', index=False)
    
    best_lgb = models_c["LightGBM"]
    perm_res = permutation_importance(best_lgb, df_test_imp_c[feature_list_c], y_test, n_repeats=3, random_state=42, n_jobs=-1)
    perm_df = pd.DataFrame({"Feature": feature_list_c, "Permutation_Importance": perm_res.importances_mean}).sort_values(by='Permutation_Importance', ascending=False)
    perm_df.to_csv('plots/permutation_importance_rankings.csv', index=False)
    
    logger.info("\n==============================================")
    logger.info("LITHOFACIES PIPELINE RUN COMPLETED SUCCESSFULLY!")
    logger.info("==============================================")

if __name__ == "__main__":
    run_pipeline()
