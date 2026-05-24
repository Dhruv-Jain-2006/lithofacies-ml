import logging
import numpy as np
import pandas as pd
import pywt
from sklearn.preprocessing import StandardScaler

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def handle_missing_values(df, feature_cols):
    """
    Handles missing values exactly as specified:
    - For small depth gaps (fewer than 5 consecutive NaN rows): fill with the median
      value of that log within the same well.
    - For large gaps or logs missing entirely in a well: drop those rows from training/testing.
    """
    logger.info("Handling missing values...")
    df_clean = df.copy()
    
    # Process each well independently
    for well in df_clean['WELL_ID'].unique():
        well_mask = df_clean['WELL_ID'] == well
        for col in feature_cols:
            if col not in df_clean.columns:
                continue
            
            # Get the curve series for this well
            series = df_clean.loc[well_mask, col].copy()
            
            is_nan = series.isna()
            if not is_nan.any():
                continue
                
            # Compute well median for this log
            well_median = series.median()
            if pd.isna(well_median):
                # Column is entirely missing in this well (e.g. PEF in Well 5). Keep as NaN.
                continue
                
            # Find lengths of contiguous NaN blocks
            # Mark changes from NaN to non-NaN (or vice-versa) to identify blocks
            nan_groups = (is_nan != is_nan.shift()).cumsum()
            nan_only_groups = nan_groups[is_nan]
            group_counts = nan_only_groups.value_counts()
            
            # Identify and fill groups smaller than 5 samples
            small_gap_groups = group_counts[group_counts < 5].index
            
            if len(small_gap_groups) > 0:
                fill_mask = is_nan & nan_groups.isin(small_gap_groups)
                series[fill_mask] = well_median
                df_clean.loc[well_mask, col] = series
                
    # Drop rows that still have NaNs in the feature columns (large gaps or missing logs)
    initial_shape = df_clean.shape
    df_clean = df_clean.dropna(subset=feature_cols)
    final_shape = df_clean.shape
    dropped_rows = initial_shape[0] - final_shape[0]
    
    logger.info(f"Missing values handled. Dropped {dropped_rows} rows due to large gaps / missing logs.")
    logger.info(f"Cleaned DataFrame shape: {final_shape}")
    
    return df_clean

def compute_cwt_features(df, target_logs, scales=None):
    """
    Computes Continuous Wavelet Transform (CWT) features using the Ricker wavelet (mexh).
    For each target log, it calculates CWT coefficients across scales, takes their absolute values,
    and returns the mean absolute coefficient as a new feature.
    CWT is computed per well to avoid boundary leakage.
    """
    if scales is None:
        scales = np.arange(1, 11)
        
    logger.info(f"Extracting CWT features for logs: {target_logs} using scales: {scales}...")
    df_out = df.copy()
    
    # Initialize new CWT feature columns with zeros
    cwt_cols = [f"{log}_CWT" for log in target_logs]
    for col in cwt_cols:
        df_out[col] = 0.0
        
    # Process well-by-well
    for well in df_out['WELL_ID'].unique():
        well_mask = df_out['WELL_ID'] == well
        well_df = df_out.loc[well_mask]
        
        for log in target_logs:
            if log not in well_df.columns:
                continue
                
            signal = well_df[log].values
            
            # Continuous Wavelet Transform using Ricker (Mexican hat) wavelet
            try:
                coefs, freqs = pywt.cwt(signal, scales, 'mexh')
                # Take mean absolute coefficient per depth sample (axis=0 is scales, axis=1 is samples)
                mean_abs_coef = np.mean(np.abs(coefs), axis=0)
                
                # Assign back to main DataFrame
                df_out.loc[well_mask, f"{log}_CWT"] = mean_abs_coef
            except Exception as e:
                logger.error(f"Error computing CWT for well {well}, log {log}: {str(e)}")
                # Fill with zeros or simple signal representation as fallback
                df_out.loc[well_mask, f"{log}_CWT"] = np.abs(signal)
                
    logger.info("CWT features successfully extracted.")
    return df_out

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
