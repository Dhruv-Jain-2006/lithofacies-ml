import os
import logging
import pickle
import numpy as np
import pandas as pd
import pywt
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def calculate_relative_depth(df):
    """
    Computes a normalized local relative depth feature per well/block to prevent depth leakage.
    RELATIVE_DEPTH = (DEPTH_MD - min_depth) / (max_depth - min_depth)
    """
    logger.info("Calculating relative depth features...")
    df_out = df.copy()
    
    # Calculate per well group/block to avoid cross-well leakage
    if 'WELL' in df_out.columns:
        df_out['RELATIVE_DEPTH'] = df_out.groupby('WELL')['DEPTH_MD'].transform(
            lambda x: (x - x.min()) / (x.max() - x.min() + 1e-8)
        )
    else:
        df_out['RELATIVE_DEPTH'] = df_out.groupby('WELL_ID')['DEPTH_MD'].transform(
            lambda x: (x - x.min()) / (x.max() - x.min() + 1e-8)
        )
    return df_out

def impute_missing_features(df_train, df_test, feature_cols):
    """
    Implements standardized data cleaning. Handles missing values (NaNs, remaining infinities)
    using SimpleImputer(strategy='median') fit ONLY on df_train and applied to both df_train and df_test
    to prevent train-test data leakage.
    """
    logger.info("Imputing missing values using SimpleImputer(strategy='median')...")
    
    # Extract feature matrices
    X_train = df_train[feature_cols].copy()
    X_test = df_test[feature_cols].copy()
    
    # Replace any leftover infinities with NaN (just in case)
    X_train = X_train.replace([np.inf, -np.inf], np.nan)
    X_test = X_test.replace([np.inf, -np.inf], np.nan)
    
    # Initialize and fit imputer on training set only
    imputer = SimpleImputer(strategy='median')
    X_train_imputed = imputer.fit_transform(X_train)
    X_test_imputed = imputer.transform(X_test)
    
    # Put back into dataframes with correct columns and indices
    df_train_out = df_train.copy()
    df_test_out = df_test.copy()
    
    df_train_out[feature_cols] = X_train_imputed
    df_test_out[feature_cols] = X_test_imputed
    
    logger.info("Imputation completed successfully without data leakage.")
    return df_train_out, df_test_out, imputer

def handle_missing_values(df, feature_cols):
    """
    Backward-compatibility wrapper for missing value handling.
    Fills NaNs with column medians.
    """
    logger.info("Running backward-compatible handle_missing_values...")
    df_out = df.copy()
    for col in feature_cols:
        if col in df_out.columns:
            median_val = df_out[col].median()
            if pd.isna(median_val):
                median_val = 0.0
            df_out[col] = df_out[col].fillna(median_val)
    return df_out

def engineer_advanced_geological_features(df):
    """
    Implements advanced petrophysical and contextual statistical learning features.
    Strictly preserves row sequence and ensures rolling statistics NEVER cross well/block boundaries.
    """
    logger.info("Engineering advanced geological and petrophysical features...")
    df_out = df.copy()
    
    # Ensure sorted order to prevent temporal/spatial row drift
    df_out = df_out.sort_values(by=['WELL_ID', 'WELL', 'DEPTH_MD']).reset_index(drop=True)
    
    # ---------------------------------------------------------
    # 1. PETROPHYSICAL ENGINEERED FEATURES
    # ---------------------------------------------------------
    # Density-Neutron Separation (porosity / shale-sand indicator)
    df_out['RHOB_minus_NPHI'] = df_out['RHOB'] - df_out['NPHI']
    
    # Resistivity Ratios (invasion profile characterization)
    df_out['RDEP_RMED_ratio'] = df_out['RDEP'] / (df_out['RMED'] + 1e-5)
    df_out['RMED_RSHA_ratio'] = df_out['RMED'] / (df_out['RSHA'] + 1e-5)
    
    # Acoustic Impedance Proxy (compaction / hardness proxy)
    df_out['acoustic_impedance'] = df_out['RHOB'] * (304800 / (df_out['DTC'] + 1e-5))
    
    # Log-Scaled Resistivities (handle log-normality)
    df_out['log_RDEP'] = np.log10(df_out['RDEP'].clip(lower=1e-5) + 1e-5)
    df_out['log_RMED'] = np.log10(df_out['RMED'].clip(lower=1e-5) + 1e-5)
    df_out['log_RSHA'] = np.log10(df_out['RSHA'].clip(lower=1e-5) + 1e-5)
    
    # ---------------------------------------------------------
    # 2. CONTEXTUAL ROLLING FEATURES & LOCAL GRADIENTS
    # ---------------------------------------------------------
    # We will compute statistics block-by-block using 'WELL' or 'WELL_ID'
    group_col = 'WELL' if 'WELL' in df_out.columns else 'WELL_ID'
    
    rolling_cols = ['GR', 'RHOB', 'NPHI', 'DTC']
    var_cols = ['GR', 'RHOB']
    gradient_cols = ['GR', 'RHOB', 'DTC']
    
    # Initialize rolling columns
    for col in rolling_cols:
        df_out[f"{col}_roll_mean"] = 0.0
        df_out[f"{col}_roll_std"] = 0.0
    for col in var_cols:
        df_out[f"{col}_roll_var"] = 0.0
    for col in gradient_cols:
        df_out[f"{col}_gradient"] = 0.0
        
    window_size = 21 # ±10 neighboring samples
    
    logger.info(f"Computing contextual rolling statistics (window={window_size}) and local gradients per well-block...")
    for block in df_out[group_col].unique():
        block_mask = df_out[group_col] == block
        block_df = df_out.loc[block_mask].sort_values(by='DEPTH_MD')
        
        if len(block_df) == 0:
            continue
            
        # Apply rolling windows
        for col in rolling_cols:
            if col in block_df.columns:
                roll = block_df[col].rolling(window=window_size, center=True, min_periods=1)
                df_out.loc[block_mask, f"{col}_roll_mean"] = roll.mean()
                df_out.loc[block_mask, f"{col}_roll_std"] = roll.std().fillna(0.0)
                
        for col in var_cols:
            if col in block_df.columns:
                roll = block_df[col].rolling(window=window_size, center=True, min_periods=1)
                df_out.loc[block_mask, f"{col}_roll_var"] = roll.var().fillna(0.0)
                
        # Compute local gradients using numpy.gradient with respect to DEPTH_MD
        for col in gradient_cols:
            if col in block_df.columns and len(block_df) > 1:
                try:
                    depths = block_df['DEPTH_MD'].values
                    values = block_df[col].values
                    # Handle NaNs temporarily for gradient math to avoid crash
                    val_series = pd.Series(values)
                    val_filled = val_series.fillna(val_series.median() if not val_series.isna().all() else 0.0).values
                    
                    grad = np.gradient(val_filled, depths)
                    df_out.loc[block_mask, f"{col}_gradient"] = grad
                except Exception as e:
                    logger.warning(f"Error computing gradient for block {block}, col {col}: {str(e)}")
                    df_out.loc[block_mask, f"{col}_gradient"] = 0.0
                    
    logger.info("Geological feature engineering completed successfully.")
    return df_out

def compute_cwt_features(df, target_logs, scales=None):
    """
    Computes Continuous Wavelet Transform (CWT) features using the Ricker wavelet (mexh)
    decomposed into LOW frequency (scales 8-10), MID frequency (scales 4-7), and HIGH frequency (scales 1-3) bands.
    Also computes the global CWT feature for backward compatibility.
    All computations are performed strictly per well/block to prevent boundary leakage.
    """
    if scales is None:
        scales = np.arange(1, 11)
        
    logger.info(f"Extracting multi-scale CWT features for logs: {target_logs}...")
    df_out = df.copy()
    
    group_col = 'WELL' if 'WELL' in df_out.columns else 'WELL_ID'
    
    # Initialize CWT feature columns
    for log in target_logs:
        df_out[f"{log}_CWT"] = 0.0
        df_out[f"{log}_CWT_low"] = 0.0
        df_out[f"{log}_CWT_mid"] = 0.0
        df_out[f"{log}_CWT_high"] = 0.0
        
    # Define frequency scale bands
    low_scales = [8, 9, 10]
    mid_scales = [4, 5, 6, 7]
    high_scales = [1, 2, 3]
    
    # Process block-by-block
    for block in df_out[group_col].unique():
        block_mask = df_out[group_col] == block
        block_df = df_out.loc[block_mask].sort_values(by='DEPTH_MD')
        
        for log in target_logs:
            if log not in block_df.columns:
                continue
                
            signal = block_df[log].values
            if len(signal) == 0:
                continue
                
            # Continuous Wavelet Transform using Ricker (Mexican hat) wavelet
            try:
                # Handle cases where signal might be entirely NaNs before imputer
                if pd.Series(signal).isna().all():
                    continue
                    
                # Fill temporary NaNs with median just for wavelet calculation to avoid CWT crash
                temp_series = pd.Series(signal)
                temp_filled = temp_series.fillna(temp_series.median() if not temp_series.isna().all() else 0.0).values
                
                # Perform Continuous Wavelet Transform
                coefs, freqs = pywt.cwt(temp_filled, scales, 'mexh')
                
                # Extract global mean absolute CWT for backward compatibility
                mean_abs_coef = np.mean(np.abs(coefs), axis=0)
                df_out.loc[block_mask, f"{log}_CWT"] = mean_abs_coef
                
                # Extract band-specific absolute coefficients
                coefs_abs = np.abs(coefs)
                # Note: axis=0 represents scales (1 to 10). Scales are 1-indexed in pywt.cwt, so:
                # low: indices 7..9 (scales 8-10)
                # mid: indices 3..6 (scales 4-7)
                # high: indices 0..2 (scales 1-3)
                df_out.loc[block_mask, f"{log}_CWT_low"] = np.mean(coefs_abs[7:10], axis=0)
                df_out.loc[block_mask, f"{log}_CWT_mid"] = np.mean(coefs_abs[3:7], axis=0)
                df_out.loc[block_mask, f"{log}_CWT_high"] = np.mean(coefs_abs[0:3], axis=0)
                
            except Exception as e:
                logger.error(f"Error computing CWT for block {block}, log {log}: {str(e)}")
                # Fill with absolute value fallback
                fallback = np.abs(np.nan_to_num(signal))
                df_out.loc[block_mask, f"{log}_CWT"] = fallback
                df_out.loc[block_mask, f"{log}_CWT_low"] = fallback
                df_out.loc[block_mask, f"{log}_CWT_mid"] = fallback
                df_out.loc[block_mask, f"{log}_CWT_high"] = fallback
                
    logger.info("Multi-scale CWT features successfully extracted.")
    return df_out

def validate_engineered_features(df, feature_cols):
    """
    Scans the engineered feature columns for Quality Control issues:
    - NaN explosion
    - Infinite values
    - Near-zero variance
    - Extreme skewness
    - Redundancy / Extreme correlation
    Outputs warnings/logs for any flagged issues.
    """
    logger.info("--- STARTING FEATURE ENGINEERING QUALITY CONTROL CHECK ---")
    df_check = df[feature_cols].copy()
    
    issues_found = 0
    
    for col in feature_cols:
        # Check NaNs
        nan_count = df_check[col].isna().sum()
        if nan_count > 0:
            logger.warning(f"QC WARN: Feature '{col}' contains {nan_count} NaNs ({nan_count/len(df)*100:.2f}%)!")
            issues_found += 1
            
        # Check Infs
        inf_count = np.isinf(df_check[col]).sum()
        if inf_count > 0:
            logger.warning(f"QC WARN: Feature '{col}' contains {inf_count} infinite values!")
            issues_found += 1
            
        # Check Near-zero variance
        val_range = df_check[col].max() - df_check[col].min()
        std_val = df_check[col].std()
        if std_val < 1e-4 or pd.isna(std_val):
            logger.warning(f"QC WARN: Feature '{col}' has near-zero variance (std={std_val:.6f})! Might be useless.")
            issues_found += 1
            
        # Check Skewness
        skew_val = df_check[col].skew()
        if abs(skew_val) > 10:
            logger.info(f"QC INFO: Feature '{col}' has extreme skewness (skew={skew_val:.2f}). Scaling recommended.")
            
    # Check redundancy / correlation
    corr_matrix = df_check.corr().abs()
    upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    highly_corr = [column for column in upper_tri.columns if any(upper_tri[column] > 0.98)]
    if len(highly_corr) > 0:
        logger.info(f"QC INFO: Found {len(highly_corr)} highly redundant/correlated features (>0.98 threshold): {highly_corr}")
        
    if issues_found == 0:
        logger.info("QC SUCCESS: All engineered features passed validation cleanly.")
    else:
        logger.info(f"QC COMPLETED: Found {issues_found} feature quality control alerts.")
        
    logger.info("--- FEATURE ENGINEERING QUALITY CONTROL COMPLETED ---")
    return issues_found

def scale_features(df_train, df_test, feature_cols):
    """
    Applies StandardScaler to feature columns. Fits the scaler only on df_train
    and transforms both df_train and df_test.
    """
    logger.info("Standardizing feature scales...")
    scaler = StandardScaler()
    
    # Extract feature matrices
    X_train = df_train[feature_cols].copy()
    X_test = df_test[feature_cols].copy()
    
    # Fit and transform
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Convert back to DataFrame to preserve column names
    X_train_df = pd.DataFrame(X_train_scaled, columns=feature_cols, index=df_train.index)
    X_test_df = pd.DataFrame(X_test_scaled, columns=feature_cols, index=df_test.index)
    
    return X_train_df, X_test_df, scaler

def scale_features_and_save(df_train, df_test, feature_cols, save_path=None):
    """
    Fits StandardScaler strictly on df_train, transforms both, and persists the scaler to save_path.
    """
    X_train_df, X_test_df, scaler = scale_features(df_train, df_test, feature_cols)
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'wb') as f:
            pickle.dump(scaler, f)
        logger.info(f"Saved scaler to {save_path}")
        
    return X_train_df, X_test_df, scaler

def correlation_pruning(df_train, feature_cols, threshold=0.95):
    """
    Identifies features in df_train that are highly correlated (|r| > threshold).
    Returns a list of clean feature columns with redundancy removed,
    and saves a redundancy clustering/diagnostic report to plots/redundancy_report.txt.
    """
    logger.info(f"Running correlation-based feature pruning on training split (threshold={threshold})...")
    
    # Compute correlation matrix on training set only to prevent leakage
    corr_matrix = df_train[feature_cols].corr().abs()
    
    # Find highly correlated pairs
    upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    
    to_drop = set()
    pruning_log = []
    
    for col in upper_tri.columns:
        # Find features correlated with this one
        correlated_with = upper_tri.index[upper_tri[col] > threshold].tolist()
        if len(correlated_with) > 0:
            # We want to keep the one that has the higher variance or first alphabetically
            # For simplicity, we drop the current column and keep the correlated_with ones,
            # unless the correlated_with ones are already marked to be dropped.
            active_correlated = [f for f in correlated_with if f not in to_drop]
            if len(active_correlated) > 0:
                to_drop.add(col)
                pruning_log.append(f"Drop '{col}' because it is highly correlated with {active_correlated} (r > {threshold})")
                
    clean_cols = [c for c in feature_cols if c not in to_drop]
    
    # Save redundancy cluster/diagnostic report
    os.makedirs('plots', exist_ok=True)
    report_path = 'plots/redundancy_report.txt'
    with open(report_path, 'w') as f:
        f.write("=== GEOLOGICAL FEATURE REDUNDANCY CLUSTERING REPORT ===\n")
        f.write(f"Correlation Threshold: {threshold}\n")
        f.write(f"Original Feature Count: {len(feature_cols)}\n")
        f.write(f"Pruned Feature Count: {len(clean_cols)}\n")
        f.write(f"Dropped Feature Count: {len(to_drop)}\n\n")
        f.write("--- DROPPED FEATURES & RATIONALE ---\n")
        for line in pruning_log:
            f.write(line + "\n")
        f.write("\n--- PRUNED FEATURE SET ---\n")
        for col in clean_cols:
            f.write(f"- {col}\n")
            
    logger.info(f"Correlation pruning complete. Kept {len(clean_cols)} of {len(feature_cols)} features. Saved report to {report_path}")
    return clean_cols

