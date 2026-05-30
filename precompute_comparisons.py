import os
import sys
import json
import logging
import pickle
import numpy as np
import pandas as pd

# Add current directory to path
sys.path.append(os.path.abspath('.'))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

from src.metrics import get_classification_metrics, load_penalty_matrix

def main():
    logger.info("Initializing precomputation of well comparisons...")
    
    # 1. Load Processed Dataset
    dataset_path = 'data/interim/processed_features.parquet'
    if not os.path.exists(dataset_path):
        logger.error(f"Processed dataset not found at {dataset_path}!")
        return
        
    df_processed = pd.read_parquet(dataset_path)
    logger.info(f"Loaded processed features dataset of shape: {df_processed.shape}")
    
    # 2. Get list of well IDs
    well_ids = sorted(df_processed['WELL_ID'].unique())
    logger.info(f"Found {len(well_ids)} well blocks: {well_ids}")
    
    # 3. Load feature definitions
    features_dict = {}
    features_dict['original'] = ['RELATIVE_DEPTH', 'CALI', 'RSHA', 'RMED', 'RDEP', 'RHOB', 'GR', 'NPHI', 'PEF', 'DTC', 'SP', 'BS']
    features_dict['wavelet'] = []
    
    # Load dynamically aligned feature lists if they exist
    for config in ['original', 'wavelet']:
        feat_path = f'data/models/features_{config}.pkl'
        if os.path.exists(feat_path):
            with open(feat_path, 'rb') as f:
                features_dict[config] = pickle.load(f)
                
    logger.info(f"Loaded original features count: {len(features_dict['original'])}")
    logger.info(f"Loaded wavelet features count: {len(features_dict['wavelet'])}")
    
    # 4. Load persisted scalers
    scalers = {}
    for config in ['original', 'wavelet']:
        scaler_path = f'data/models/scaler_{config}.pkl'
        if os.path.exists(scaler_path):
            with open(scaler_path, 'rb') as f:
                scalers[config] = pickle.load(f)
                
    # 5. Load all models
    from src.models import load_models
    logger.info("Loading original models...")
    models_orig = load_models('original')
    logger.info("Loading wavelet models...")
    models_wav = load_models('wavelet')
    
    # 6. Load penalty matrix
    penalty_matrix = load_penalty_matrix()
    
    # 7. Precompute comparisons well-by-well
    preloaded_comparisons = {}
    model_names = ['KNN', 'Random Forest', 'Decision Tree', 'XGBoost', 'LightGBM']
    
    for well_id in well_ids:
        logger.info(f"Precomputing comparisons for well: {well_id}")
        well_df = df_processed[df_processed['WELL_ID'] == well_id].sort_values(by='DEPTH_MD').copy()
        
        # Downsample metrics comparison just like original app.py
        if len(well_df) > 1000:
            well_df_metrics = well_df.iloc[::10].copy()
        else:
            well_df_metrics = well_df.copy()
            
        y_true = well_df_metrics['LITHOLOGY'].fillna(-1).values.astype(int)
        valid_mask = y_true >= 0
        
        if not valid_mask.any():
            preloaded_comparisons[well_id] = []
            logger.warning(f"No ground truth for well: {well_id}")
            continue
            
        y_true_valid = y_true[valid_mask]
        
        # Impute missing values with global training medians
        X_raw12_all = well_df_metrics[features_dict['original']].copy()
        medians12 = df_processed[features_dict['original']].median()
        X_raw12_all = X_raw12_all.fillna(medians12)
        
        X_raw19_all = well_df_metrics[features_dict['wavelet']].copy()
        medians19 = df_processed[features_dict['wavelet']].median()
        X_raw19_all = X_raw19_all.fillna(medians19)
        
        well_results = []
        for name in model_names:
            # 12F Original metrics
            acc12, pen12, ham12 = 0.0, 0.0, 0.0
            if name in models_orig:
                model12 = models_orig[name]
                X_in12 = scalers['original'].transform(X_raw12_all) if name == 'KNN' else X_raw12_all.values
                y_pred12 = model12.predict(X_in12).astype(int)[valid_mask]
                m12 = get_classification_metrics(y_true_valid, y_pred12, penalty_matrix)
                acc12 = float(m12["Accuracy"])
                pen12 = float(m12["PenaltyScore"])
                ham12 = float(m12["HammingLoss"])
                
            # 19F Wavelet metrics
            acc19, pen19, ham19 = 0.0, 0.0, 0.0
            if name in models_wav:
                model19 = models_wav[name]
                X_in19 = scalers['wavelet'].transform(X_raw19_all) if name == 'KNN' else X_raw19_all.values
                y_pred19 = model19.predict(X_in19).astype(int)[valid_mask]
                m19 = get_classification_metrics(y_true_valid, y_pred19, penalty_matrix)
                acc19 = float(m19["Accuracy"])
                pen19 = float(m19["PenaltyScore"])
                ham19 = float(m19["HammingLoss"])
                
            well_results.append({
                "name": name + " ★" if name == "Random Forest" else name,
                "acc12": round(acc12, 3),
                "pen12": round(pen12, 4),
                "ham12": round(ham12, 4),
                "acc19": round(acc19, 3),
                "pen19": round(pen19, 4),
                "ham19": round(ham19, 4)
            })
            
        preloaded_comparisons[well_id] = well_results
        logger.info(f"Successfully precomputed well {well_id}: {well_results[1]}") # Print Random Forest as sample
        
    # 8. Save to JSON file
    os.makedirs('data/models', exist_ok=True)
    json_path = 'data/models/preloaded_comparisons.json'
    with open(json_path, 'w') as f:
        json.dump(preloaded_comparisons, f, indent=4)
        
    logger.info(f"Saved all precomputed comparisons to {json_path} successfully!")

if __name__ == '__main__':
    main()
