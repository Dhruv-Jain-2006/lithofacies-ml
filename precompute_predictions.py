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
from src.sequence import viterbi_decode

# Lithology configuration
LITHOLOGY_LABELS = [
    'Sandstone', 'Shale', 'Limestone', 'Chalk', 'Halite', 
    'Dolomite', 'Coal', 'Marl', 'Anhydrite', 'Tuff', 'Basement', 'Sandstone/Shale'
]
LITHOLOGY_COLORS = [
    '#F5A623', '#555566', '#38B6FF', '#E8EDF2', '#FF4D6A', 
    '#00E5CC', '#111116', '#8B6914', '#C084FC', '#86EFAC', '#374151', '#D97706'
]

def calculate_uncertainty_metrics(probabilities):
    entropies = []
    top3_candidates = []
    low_confidence_flags = []
    
    for probs in probabilities:
        # Shannon Entropy
        eps = 1e-12
        p_clipped = np.clip(probs, eps, 1.0)
        entropy = -np.sum(probs * np.log2(p_clipped))
        entropies.append(float(entropy))
        
        # Get top-3 candidates
        top3_indices = np.argsort(probs)[::-1][:3]
        candidates = []
        for idx in top3_indices:
            candidates.append({
                "class_idx": int(idx),
                "name": LITHOLOGY_LABELS[idx],
                "probability": float(probs[idx]),
                "color": LITHOLOGY_COLORS[idx]
            })
        top3_candidates.append(candidates)
        
        # Low confidence flag
        highest_prob = float(np.max(probs))
        low_confidence_flags.append(bool(highest_prob < 0.35))
        
    return entropies, top3_candidates, low_confidence_flags

def load_single_model(config, model_name):
    safe_name = model_name.replace(' ', '_')
    path = f"data/models/models_{config}_{safe_name}.pkl"
    if not os.path.exists(path):
        giant_path = f"data/models/models_{config}.pkl"
        if os.path.exists(giant_path):
            with open(giant_path, 'rb') as f:
                models = pickle.load(f)
                return models.get(model_name)
        return None
    with open(path, 'rb') as f:
        return pickle.load(f)

def main():
    logger.info("Starting offline precomputation of sandbox predictions...")
    
    # 1. Load Processed Dataset
    dataset_path = 'data/interim/processed_features.parquet'
    if not os.path.exists(dataset_path):
        logger.error(f"Processed dataset not found at {dataset_path}!")
        return
        
    df_processed = pd.read_parquet(dataset_path)
    logger.info(f"Loaded dataset: {df_processed.shape}")
    
    well_ids = ['WELL_FORCE_2020_01', 'WELL_FORCE_2020_02', 'WELL_FORCE_2020_03', 'WELL_FORCE_2020_04']
    model_names = ['KNN', 'Random Forest', 'Decision Tree', 'XGBoost', 'LightGBM']
    feature_sets = ['original', 'original_nodep', 'engineered', 'wavelet', 'wavelet_nodep']
    
    # 2. Setup features config
    ORIGINAL_COLS = ['RELATIVE_DEPTH', 'CALI', 'RSHA', 'RMED', 'RDEP', 'RHOB', 'GR', 'NPHI', 'PEF', 'DTC', 'SP', 'BS']
    PETRO_COLS = ['GR_RHOB_RATIO', 'NPHI_RHOB_DIFF', 'LOG_RMED', 'LOG_RDEP']
    ROLL_COLS = ['GR_ROLL_MEAN_5', 'GR_ROLL_STD_5', 'RHOB_ROLL_MEAN_5', 'RHOB_ROLL_STD_5']
    CWT_BAND_COLS = ['CWT_GR_SCALE_1', 'CWT_GR_SCALE_2', 'CWT_GR_SCALE_3', 'CWT_RHOB_SCALE_1', 'CWT_RHOB_SCALE_2', 'CWT_RHOB_SCALE_3']
    
    features_dict = {
        'original': ORIGINAL_COLS,
        'original_nodep': [c for c in ORIGINAL_COLS if c != 'RELATIVE_DEPTH'],
        'engineered': PETRO_COLS + ROLL_COLS + CWT_BAND_COLS,
        'wavelet': ORIGINAL_COLS + PETRO_COLS + ROLL_COLS + CWT_BAND_COLS,
        'wavelet_nodep': [c for c in (ORIGINAL_COLS + PETRO_COLS + ROLL_COLS + CWT_BAND_COLS) if c != 'RELATIVE_DEPTH']
    }
    
    # Check for dynamically aligned pruned feature lists on disk
    for config in feature_sets:
        feat_path = f'data/models/features_{config}.pkl'
        if os.path.exists(feat_path):
            with open(feat_path, 'rb') as f:
                features_dict[config] = pickle.load(f)
                
    # 3. Load Scalers
    scalers_dict = {}
    for config in feature_sets:
        scaler_path = f'data/models/scaler_{config}.pkl'
        if os.path.exists(scaler_path):
            with open(scaler_path, 'rb') as f:
                scalers_dict[config] = pickle.load(f)
                
    # 4. Load Penalty & Transition Matrices
    penalty_matrix = load_penalty_matrix()
    transition_matrix = None
    transition_matrix_path = 'data/models/transition_matrix.pkl'
    if os.path.exists(transition_matrix_path):
        with open(transition_matrix_path, 'rb') as f:
            transition_matrix = pickle.load(f)
            
    fallback_trans = np.eye(12) * 0.95 + (1.0 - np.eye(12)) * (0.05 / 11.0)
    trans_mat = transition_matrix if transition_matrix is not None else fallback_trans
    
    # 5. Precompute
    preloaded_predictions = {}
    
    for well_id in well_ids:
        logger.info(f"Processing well: {well_id}")
        well_df = df_processed[df_processed['WELL_ID'] == well_id].sort_values(by='DEPTH_MD').copy()
        
        if len(well_df) == 0:
            logger.warning(f"Well {well_id} has no data!")
            continue
            
        y_true = well_df['LITHOLOGY'].fillna(-1).values.astype(int)
        valid_mask = y_true >= 0
        
        for feature_set in feature_sets:
            cols = features_dict[feature_set]
            scaler = scalers_dict.get(feature_set)
            
            # Prepare feature matrices
            X_raw = well_df[cols].copy()
            medians = df_processed[cols].median()
            X_raw = X_raw.fillna(medians)
            
            for model_name in model_names:
                logger.info(f"Predicting combination: {well_id} | {feature_set} | {model_name}")
                model = load_single_model(feature_set, model_name)
                if model is None:
                    logger.error(f"Model {model_name} not found in models dict for {feature_set}!")
                    continue
                    
                X_input = scaler.transform(X_raw) if model_name == 'KNN' else X_raw.values
                
                # Predictions
                predictions = model.predict(X_input).astype(int)
                
                # Probabilities & Uncertainty metrics
                try:
                    probabilities = model.predict_proba(X_input)
                    max_probs = [float(p[pred]) for p, pred in zip(probabilities, predictions)]
                    entropies, top3_candidates, low_confidence_flags = calculate_uncertainty_metrics(probabilities)
                except Exception as p_err:
                    logger.warning(f"Probabilities failed: {str(p_err)}")
                    probabilities = np.zeros((len(predictions), 12))
                    for idx, p_cls in enumerate(predictions):
                        probabilities[idx, int(p_cls)] = 1.0
                    max_probs = [1.0] * len(predictions)
                    entropies = [0.0] * len(predictions)
                    top3_candidates = [
                        [{
                            "class_idx": int(pred),
                            "name": LITHOLOGY_LABELS[int(pred)],
                            "probability": 1.0,
                            "color": LITHOLOGY_COLORS[int(pred)]
                        }] for pred in predictions
                    ]
                    low_confidence_flags = [False] * len(predictions)
                    
                # Viterbi
                try:
                    viterbi_predictions = viterbi_decode(probabilities, trans_mat).astype(int).tolist()
                except Exception as vex:
                    logger.error(f"Viterbi failed: {str(vex)}")
                    viterbi_predictions = predictions.astype(int).tolist()
                    
                # Metrics for active predictions under viterbi=False
                acc, ham, pen = 0.0, 0.0, 0.0
                if valid_mask.any():
                    m = get_classification_metrics(y_true[valid_mask], predictions[valid_mask], penalty_matrix)
                    acc = float(m["Accuracy"])
                    ham = float(m["HammingLoss"])
                    pen = float(m["PenaltyScore"])
                    
                # Metrics for active predictions under viterbi=True
                acc_vit, ham_vit, pen_vit = 0.0, 0.0, 0.0
                if valid_mask.any():
                    m_vit = get_classification_metrics(y_true[valid_mask], np.array(viterbi_predictions)[valid_mask], penalty_matrix)
                    acc_vit = float(m_vit["Accuracy"])
                    ham_vit = float(m_vit["HammingLoss"])
                    pen_vit = float(m_vit["PenaltyScore"])
                    
                mismatches = [int(t != p) if t >= 0 else 0 for t, p in zip(y_true, predictions)]
                mismatches_viterbi = [int(t != p) if t >= 0 else 0 for t, p in zip(y_true, viterbi_predictions)]
                
                # Store both Viterbi and non-Viterbi payload in the key to make it 100% precomputed!
                well_key = f"{well_id}_{model_name}_{feature_set}"
                preloaded_predictions[well_key] = {
                    "well_id": well_id,
                    "model_name": model_name,
                    "feature_set": feature_set,
                    "predictions": predictions.tolist(),
                    "viterbi_predictions": viterbi_predictions,
                    "probabilities": max_probs,
                    "entropies": entropies,
                    "top3_candidates": top3_candidates,
                    "low_confidence_flags": low_confidence_flags,
                    "mismatches": mismatches,
                    "mismatches_viterbi": mismatches_viterbi,
                    "metrics": {
                        "accuracy": acc,
                        "hamming_loss": ham,
                        "penalty_score": pen
                    },
                    "metrics_viterbi": {
                        "accuracy": acc_vit,
                        "hamming_loss": ham_vit,
                        "penalty_score": pen_vit
                    }
                }
            
            # Clean up memory immediately
            import gc
            gc.collect()
            
    # Save to disk
    json_path = 'data/models/preloaded_predictions.json'
    logger.info(f"Saving precomputed predictions to {json_path}...")
    with open(json_path, 'w') as f:
        json.dump(preloaded_predictions, f)
    logger.info("Precomputation completed successfully!")

if __name__ == '__main__':
    main()
