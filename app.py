import os
import sys
import pickle
import logging
import numpy as np
import pandas as pd
from flask import Flask, jsonify as _orig_jsonify, render_template, request, send_from_directory
from sklearn.preprocessing import StandardScaler

def sanitize_json_value(val):
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return None
    if isinstance(val, float) and np.isinf(val):
        return None
    if isinstance(val, dict):
        return {k: sanitize_json_value(v) for k, v in val.items()}
    if isinstance(val, (list, tuple, np.ndarray)):
        return [sanitize_json_value(x) for x in val]
    if isinstance(val, (np.integer, int)):
        return int(val)
    if isinstance(val, (np.floating, float)):
        return float(val)
    # Check pandas Null/NA values explicitly
    if pd.api.types.is_scalar(val) and pd.isna(val):
        return None
    return val

def jsonify(*args, **kwargs):
    if args:
        if len(args) == 1:
            data = args[0]
        else:
            data = list(args)
    else:
        data = kwargs
    return _orig_jsonify(sanitize_json_value(data))

# Add current directory and src to the path
sys.path.append(os.path.abspath('.'))

# Import new Prompt 3 research classes and sequence tools
from src.models import WeightedCalibratedEnsemble
from src.sequence import viterbi_decode

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


app = Flask(__name__, template_folder='templates', static_folder='static')

# ---------------------------------------------------------
# CONSTANTS & LITHOFACIES DEFINITIONS
# ---------------------------------------------------------
LITHOLOGY_LABELS = [
    'Sandstone', 'Sandstone/Shale', 'Shale', 'Marl', 'Dolomite', 'Limestone',
    'Chalk', 'Halite', 'Anhydrite', 'Tuff', 'Coal', 'Basement'
]

LITHOLOGY_COLORS = [
    '#FFE066',  # 0: Sandstone (Bright Gold)
    '#C7F9CC',  # 1: Sandstone/Shale (Sage Green)
    '#566573',  # 2: Shale (Slate Gray)
    '#A9DFBF',  # 3: Marl (Light Muted Green)
    '#138D75',  # 4: Dolomite (Teal)
    '#5DADE2',  # 5: Limestone (Sky Blue)
    '#F4F6F7',  # 6: Chalk (Pure Off-White)
    '#EBDEF0',  # 7: Halite (Lavender Pink)
    '#D2B4DE',  # 8: Anhydrite (Soft Purple)
    '#F5CBA7',  # 9: Tuff (Apricot Orange)
    '#1C2833',  # 10: Coal (Jet Charcoal Black)
    '#922B21'   # 11: Basement (Crimson Maroon)
]

ORIGINAL_COLS = ['RELATIVE_DEPTH', 'CALI', 'RSHA', 'RMED', 'RDEP', 'RHOB', 'GR', 'NPHI', 'PEF', 'DTC', 'SP', 'BS']
WAVELET_LOGS = ['GR', 'NPHI', 'SP', 'RDEP', 'RHOB', 'DTC', 'PEF']

# Advanced geological feature lists
PETRO_COLS = ['RHOB_minus_NPHI', 'RDEP_RMED_ratio', 'RMED_RSHA_ratio', 'acoustic_impedance', 'log_RDEP', 'log_RMED', 'log_RSHA']
ROLL_COLS = [
    'GR_roll_mean', 'RHOB_roll_mean', 'NPHI_roll_mean', 'DTC_roll_mean',
    'GR_roll_std', 'RHOB_roll_std', 'NPHI_roll_std', 'DTC_roll_std',
    'GR_roll_var', 'RHOB_roll_var',
    'GR_gradient', 'RHOB_gradient', 'DTC_gradient'
]
CWT_BAND_COLS = []
for log in WAVELET_LOGS:
    CWT_BAND_COLS.extend([f"{log}_CWT_low", f"{log}_CWT_mid", f"{log}_CWT_high"])

WAVELET_COLS = ORIGINAL_COLS + PETRO_COLS + ROLL_COLS + CWT_BAND_COLS

# ---------------------------------------------------------
# STARTUP LOADERS (DATA, MODELS, & SCALERS)
# ---------------------------------------------------------
df_processed = None
cached_uploaded_well_df = None

# Config-specific dictionaries for 5 configurations
models_dict = {}
scalers_dict = {}
features_dict = {}

# Backward compatible globally referenced models & scalers
models_orig = {}
models_wav = {}
scaler_orig = None
scaler_wav = None

penalty_matrix = None
transition_matrix = None

def initialize_app():
    global df_processed, models_dict, scalers_dict, features_dict, penalty_matrix, transition_matrix
    global models_orig, models_wav, scaler_orig, scaler_wav, WAVELET_COLS
    
    # 1. Load Processed Dataset
    dataset_path = 'data/interim/processed_features.parquet'
    if not os.path.exists(dataset_path):
        logger.error(f"Processed dataset not found at {dataset_path}! Please run pipeline first.")
        fallback_path = 'data/processed/merged_raw.parquet'
        if os.path.exists(fallback_path):
            logger.info("Falling back to raw merged dataset and engineering features dynamically...")
            from src.features import handle_missing_values, compute_cwt_features, calculate_relative_depth, engineer_advanced_geological_features
            raw_df = pd.read_parquet(fallback_path)
            raw_df = calculate_relative_depth(raw_df)
            clean_df = handle_missing_values(raw_df, ORIGINAL_COLS + ['LITHOLOGY'])
            clean_df = engineer_advanced_geological_features(clean_df)
            df_processed = compute_cwt_features(clean_df, WAVELET_LOGS)
        else:
            raise FileNotFoundError("No well log dataset found in data/ directories.")
    else:
        df_processed = pd.read_parquet(dataset_path)
        logger.info(f"Loaded processed features dataset: {df_processed.shape}")
        
    # Define features configs with strict fallback lists
    config_names = ['original', 'original_nodep', 'engineered', 'wavelet', 'wavelet_nodep']
    
    features_dict['original'] = ORIGINAL_COLS
    features_dict['original_nodep'] = [c for c in ORIGINAL_COLS if c != 'RELATIVE_DEPTH']
    features_dict['engineered'] = PETRO_COLS + ROLL_COLS + CWT_BAND_COLS
    features_dict['wavelet'] = ORIGINAL_COLS + PETRO_COLS + ROLL_COLS + CWT_BAND_COLS
    features_dict['wavelet_nodep'] = [c for c in features_dict['wavelet'] if c != 'RELATIVE_DEPTH']
    
    # 2. Check for pruned feature lists dynamically from disk
    for config in config_names:
        feat_path = f'data/models/features_{config}.pkl'
        if os.path.exists(feat_path):
            try:
                with open(feat_path, 'rb') as f:
                    features_dict[config] = pickle.load(f)
                logger.info(f"Dynamically aligned features list for '{config}': {len(features_dict[config])} features.")
            except Exception as e:
                logger.error(f"Error loading features list for {config}: {str(e)}")
                
    # Update WAVELET_COLS with the dynamically aligned one
    WAVELET_COLS = features_dict['wavelet']
    
    # 3. Load Persisted Scalers (strictly load, do NOT fit unless fallback needed)
    for config in config_names:
        scaler_path = f'data/models/scaler_{config}.pkl'
        if os.path.exists(scaler_path):
            try:
                with open(scaler_path, 'rb') as f:
                    scalers_dict[config] = pickle.load(f)
                logger.info(f"Loaded persisted scaler for '{config}' from disk.")
            except Exception as e:
                logger.error(f"Error loading scaler for {config}: {str(e)}")
        else:
            logger.warning(f"Persisted scaler for '{config}' not found! Fitting fallback scaler...")
            wells = sorted(df_processed['WELL_ID'].unique())
            train_wells = wells[:8]
            df_train = df_processed[df_processed['WELL_ID'].isin(train_wells)]
            scalers_dict[config] = StandardScaler().fit(df_train[features_dict[config]])
            
    # Set backward compatible global references
    scaler_orig = scalers_dict['original']
    scaler_wav = scalers_dict['wavelet']
    
    # 4. Load ML Models
    for config in config_names:
        try:
            from src.models import load_models
            models_dict[config] = load_models(config)
            logger.info(f"Loaded calibrated models for feature set '{config}'.")
        except Exception as e:
            logger.error(f"Error loading models for configuration '{config}': {str(e)}")
            
    # Set backward compatible global references
    models_orig = models_dict.get('original', {})
    models_wav = models_dict.get('wavelet', {})
    
    # 5. Load Geological Penalty Matrix & Stratigraphic Transition Matrix
    try:
        from src.metrics import load_penalty_matrix
        penalty_matrix = load_penalty_matrix()
        logger.info("Loaded geological penalty matrix.")
    except Exception as e:
        logger.error(f"Error loading penalty matrix: {str(e)}")
        
    try:
        transition_matrix_path = 'data/models/transition_matrix.pkl'
        if os.path.exists(transition_matrix_path):
            with open(transition_matrix_path, 'rb') as f:
                transition_matrix = pickle.load(f)
            logger.info("Loaded geological stratigraphic transition matrix successfully.")
        else:
            logger.warning("Geological transition matrix not found on disk yet.")
    except Exception as e:
        logger.error(f"Error loading geological transition matrix: {str(e)}")


# Trigger initialization on startup
startup_error = None
startup_traceback = None
try:
    initialize_app()
except Exception as ex:
    import traceback
    startup_error = str(ex)
    startup_traceback = traceback.format_exc()
    logger.error(f"Startup initialization failed: {str(ex)}")

# ---------------------------------------------------------
# FLASK ROUTING
# ---------------------------------------------------------

@app.route('/')
def index():
    """Serves the primary SPA dashboard page."""
    return render_template('index.html')

@app.route('/plots/<path:filename>')
def serve_plot(filename):
    """Serves static plot files directly from the plots directory."""
    return send_from_directory('plots', filename)

@app.route('/api/debug/logs', methods=['GET'])
def debug_logs():
    return jsonify({
        "df_processed_loaded": df_processed is not None,
        "models_orig_keys": list(models_orig.keys()),
        "models_wav_keys": list(models_wav.keys()),
        "penalty_matrix_loaded": penalty_matrix is not None,
        "startup_error": startup_error,
        "startup_traceback": startup_traceback
    })

@app.route('/api/status', methods=['GET'])
def get_status():
    """Returns general service configuration, available features, and state."""
    try:
        status_data = {
            "status": "ready" if df_processed is not None else "error",
            "models_loaded": {
                "original_12": list(models_orig.keys()),
                "wavelet_19": list(models_wav.keys())
            },
            "lithofacies": [
                {"class_idx": idx, "name": label, "color": LITHOLOGY_COLORS[idx]} 
                for idx, label in enumerate(LITHOLOGY_LABELS)
            ],
            "feature_sets": {
                "original": ORIGINAL_COLS,
                "wavelet": WAVELET_COLS
            },
            "dataset_info": {
                "total_rows": len(df_processed) if df_processed is not None else 0,
                "num_wells": int(df_processed['WELL_ID'].nunique()) if df_processed is not None else 0,
                "class_distribution": {int(k): int(v) for k, v in df_processed['LITHOLOGY'].value_counts().to_dict().items() if not pd.isna(k)} if df_processed is not None else {}
            }
        }
        return jsonify(status_data)
    except Exception as e:
        logger.error(f"Status API Error: {str(e)}")
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/api/wells', methods=['GET'])
def get_wells():
    """Lists all available wells with depth bounds and metadata."""
    if df_processed is None:
        return jsonify({"error": "Dataset not loaded"}), 500
        
    try:
        wells_list = []
        for well_id in sorted(df_processed['WELL_ID'].unique()):
            well_df = df_processed[df_processed['WELL_ID'] == well_id]
            wells_list.append({
                "well_id": well_id,
                "min_depth": float(well_df['DEPTH_MD'].min()),
                "max_depth": float(well_df['DEPTH_MD'].max()),
                "sample_count": len(well_df)
            })
        return jsonify(wells_list)
    except Exception as e:
        logger.error(f"Wells API Error: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/well/<well_id>', methods=['GET'])
def get_well_data(well_id):
    """Returns raw and wavelet features for the specified well."""
    if df_processed is None:
        return jsonify({"error": "Dataset not loaded"}), 500
        
    try:
        if well_id in ['WELL_CUSTOM', 'uploaded_well']:
            if cached_uploaded_well_df is None:
                return jsonify({"error": "No uploaded well in cache. Please upload a LAS file first."}), 400
            well_df = cached_uploaded_well_df.copy()
        else:
            well_df = df_processed[df_processed['WELL_ID'] == well_id].sort_values(by='DEPTH_MD')
            
        if len(well_df) == 0:
            return jsonify({"error": f"Well {well_id} not found"}), 404
            
        # Standardize curve response as dictionary of arrays for fast JSON transit
        payload = {}
        for col in well_df.columns:
            # Handle float columns and NaN conversions to None for JSON compliance
            if pd.api.types.is_numeric_dtype(well_df[col]):
                payload[col] = [None if pd.isna(x) else float(x) for x in well_df[col].values]
            else:
                payload[col] = well_df[col].values.tolist()
                
        return jsonify(payload)
    except Exception as e:
        logger.error(f"Well Data API Error: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/well/<well_id>/compare', methods=['GET'])
def get_well_model_comparison(well_id):
    """
    Computes accuracy, penalty score, and hamming loss for all 5 models 
    under both original (12F) and wavelet (19F) feature sets for the specified well.
    """
    if df_processed is None:
        return jsonify({"error": "Dataset not loaded"}), 500
        
    try:
        if well_id in ['WELL_CUSTOM', 'uploaded_well']:
            if cached_uploaded_well_df is None:
                return jsonify({"error": "No uploaded well in cache. Please upload a LAS file first."}), 400
            well_df = cached_uploaded_well_df.copy()
        else:
            well_df = df_processed[df_processed['WELL_ID'] == well_id].sort_values(by='DEPTH_MD').copy()
            
        if len(well_df) == 0:
            return jsonify({"error": f"Well {well_id} not found"}), 404
            
        # Downsample well logs for metrics comparison to avoid Gunicorn timeouts & OOM kills on Free Cloud Tiers
        if len(well_df) > 1000:
            well_df_metrics = well_df.iloc[::10].copy()
        else:
            well_df_metrics = well_df.copy()
            
        y_true = well_df_metrics['LITHOLOGY'].fillna(-1).values.astype(int)
        valid_mask = y_true >= 0
        
        # If no ground truth, we'll return an empty list or placeholders
        if not valid_mask.any():
            return jsonify([])
            
        y_true_valid = y_true[valid_mask]
        
        from src.metrics import get_classification_metrics
        
        results = []
        model_names = ['KNN', 'Random Forest', 'Decision Tree', 'XGBoost', 'LightGBM']
        
        # Impute missing values with global training medians to prevent KNN/other NaN crashes
        X_raw12_all = well_df_metrics[ORIGINAL_COLS].copy()
        medians12 = df_processed[ORIGINAL_COLS].median()
        X_raw12_all = X_raw12_all.fillna(medians12)
        
        X_raw19_all = well_df_metrics[WAVELET_COLS].copy()
        medians19 = df_processed[WAVELET_COLS].median()
        X_raw19_all = X_raw19_all.fillna(medians19)
        
        for name in model_names:
            try:
                # 12F Original metrics
                acc12, pen12, ham12 = 0.0, 0.0, 0.0
                if name in models_orig:
                    model12 = models_orig[name]
                    X_in12 = scaler_orig.transform(X_raw12_all) if name == 'KNN' else X_raw12_all.values
                    y_pred12 = model12.predict(X_in12).astype(int)[valid_mask]
                    m12 = get_classification_metrics(y_true_valid, y_pred12, penalty_matrix)
                    acc12 = float(m12["Accuracy"])
                    pen12 = float(m12["PenaltyScore"])
                    ham12 = float(m12["HammingLoss"])
                    
                # 19F Wavelet metrics
                acc19, pen19, ham19 = 0.0, 0.0, 0.0
                if name in models_wav:
                    model19 = models_wav[name]
                    X_in19 = scaler_wav.transform(X_raw19_all) if name == 'KNN' else X_raw19_all.values
                    y_pred19 = model19.predict(X_in19).astype(int)[valid_mask]
                    m19 = get_classification_metrics(y_true_valid, y_pred19, penalty_matrix)
                    acc19 = float(m19["Accuracy"])
                    pen19 = float(m19["PenaltyScore"])
                    ham19 = float(m19["HammingLoss"])
            except Exception as model_err:
                logger.error(f"Error predicting model {name}: {str(model_err)}")
                import traceback
                traceback.print_exc()
                return jsonify({"error": f"Model {name} failed: {str(model_err)}", "trace": traceback.format_exc()}), 500
                
            results.append({
                "name": name + " ★" if name == "Random Forest" else name,
                "acc12": round(acc12, 3),
                "pen12": round(pen12, 4),
                "ham12": round(ham12, 4),
                "acc19": round(acc19, 3),
                "pen19": round(pen19, 4),
                "ham19": round(ham19, 4)
            })
            
        return jsonify(results)
    except Exception as e:
        logger.error(f"Well Compare API Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

def calculate_uncertainty_metrics(probabilities):
    """
    Computes Shannon entropy, top-3 candidates with their probabilities,
    and flags low confidence predictions (highest prob < 0.35) for each sample.
    """
    entropies = []
    top3_candidates = []
    low_confidence_flags = []
    
    for probs in probabilities:
        # Shannon Entropy: -sum(p * log2(p))
        # Add epsilon to prevent log(0)
        eps = 1e-12
        p_clipped = np.clip(probs, eps, 1.0)
        entropy = -np.sum(probs * np.log2(p_clipped))
        entropies.append(float(entropy))
        
        # Get top-3 candidate classes
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

@app.route('/api/predict', methods=['POST'])
def run_prediction():
    """
    Runs ML model inference on the selected well with uncertainty estimation and Viterbi sequence smoothing.
    Payload: { "well_id": "...", "model_name": "...", "feature_set": "original"|"wavelet", "use_viterbi": bool }
    """
    if df_processed is None:
        return jsonify({"error": "Dataset not loaded"}), 500
        
    try:
        data = request.json or {}
        well_id = data.get('well_id')
        model_name = data.get('model_name')
        feature_set = data.get('feature_set', 'original')
        use_viterbi = data.get('use_viterbi', False)
        
        if not well_id or not model_name:
            return jsonify({"error": "Missing well_id or model_name parameter"}), 400
            
        if well_id in ['WELL_CUSTOM', 'uploaded_well']:
            if cached_uploaded_well_df is None:
                return jsonify({"error": "No uploaded well in cache. Please upload a LAS file first."}), 400
            well_df = cached_uploaded_well_df.copy()
        else:
            well_df = df_processed[df_processed['WELL_ID'] == well_id].sort_values(by='DEPTH_MD').copy()
            
        if len(well_df) == 0:
            return jsonify({"error": f"Well {well_id} not found"}), 404
            
        # Select active models dictionary
        models = models_dict.get(feature_set, models_dict.get('original', {}))
        cols = features_dict.get(feature_set, features_dict.get('original', ORIGINAL_COLS))
        scaler = scalers_dict.get(feature_set, scalers_dict.get('original'))
        
        if model_name not in models:
            return jsonify({"error": f"Model {model_name} not loaded for feature set {feature_set}"}), 400
            
        model = models[model_name]
        
        # Prepare feature matrix and handle missing values using global dataset medians
        X_raw = well_df[cols].copy()
        medians = df_processed[cols].median()
        X_raw = X_raw.fillna(medians)
        
        # Apply scaling if KNN
        if model_name == 'KNN':
            X_input = scaler.transform(X_raw)
        else:
            X_input = X_raw.values
            
        # Run predictions
        predictions = model.predict(X_input)
        
        # Compute probabilities and advanced geological uncertainty
        try:
            probabilities = model.predict_proba(X_input)
            max_probs = [float(p[pred]) for p, pred in zip(probabilities, predictions)]
            entropies, top3_candidates, low_confidence_flags = calculate_uncertainty_metrics(probabilities)
        except Exception as p_err:
            logger.warning(f"Probability estimation failed, using flat fallbacks: {str(p_err)}")
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
            
        # Run Viterbi sequence smoothing
        try:
            if transition_matrix is not None:
                viterbi_predictions = viterbi_decode(probabilities, transition_matrix).astype(int).tolist()
            else:
                fallback_trans = np.eye(12) * 0.95 + (1.0 - np.eye(12)) * (0.05 / 11.0)
                viterbi_predictions = viterbi_decode(probabilities, fallback_trans).astype(int).tolist()
        except Exception as vex:
            logger.error(f"Viterbi sequence decoding failed: {str(vex)}")
            viterbi_predictions = predictions.astype(int).tolist()
            
        # Active y_pred based on use_viterbi toggle
        y_pred = np.array(viterbi_predictions) if use_viterbi else predictions.astype(int)
        
        # Calculate evaluation metrics side-by-side with ground-truth
        y_true = well_df['LITHOLOGY'].fillna(-1).values.astype(int)
        
        # Filter out unmapped indicators (-1)
        valid_mask = y_true >= 0
        
        from src.metrics import get_classification_metrics
        if valid_mask.any():
            metrics_res = get_classification_metrics(y_true[valid_mask], y_pred[valid_mask], penalty_matrix)
            accuracy = float(metrics_res["Accuracy"])
            hamming_loss = float(metrics_res["HammingLoss"])
            penalty_score = float(metrics_res["PenaltyScore"])
        else:
            accuracy = 0.0
            hamming_loss = 0.0
            penalty_score = 0.0
            
        # Calculate depth-by-depth mismatch highlights (ignore unlabeled sections)
        mismatches = [int(t != p) if t >= 0 else 0 for t, p in zip(y_true, y_pred)]
        
        return jsonify({
            "well_id": well_id,
            "model_name": model_name,
            "feature_set": feature_set,
            "predictions": predictions.astype(int).tolist(),
            "viterbi_predictions": viterbi_predictions,
            "probabilities": max_probs,
            "entropies": entropies,
            "top3_candidates": top3_candidates,
            "low_confidence_flags": low_confidence_flags,
            "mismatches": mismatches,
            "viterbi_applied": bool(use_viterbi),
            "metrics": {
                "accuracy": accuracy,
                "hamming_loss": hamming_loss,
                "penalty_score": penalty_score
            }
        })
    except Exception as e:
        logger.error(f"Prediction API Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route('/api/sandbox/predict', methods=['POST'])
def run_sandbox_prediction():
    """
    Runs ML model inference on a single 12-dimensional log vector supplied by the sandbox.
    Payload: { "model_name": "...", "features": { "GR": 45.0, "RHOB": 2.35, ... } }
    """
    if df_processed is None:
        return jsonify({"error": "Dataset not loaded"}), 500
        
    try:
        data = request.json or {}
        model_name = data.get('model_name')
        feature_vals = data.get('features', {})
        
        if not model_name or not feature_vals:
            return jsonify({"error": "Missing model_name or features parameter"}), 400
            
        # Select active models dictionary (always original 12 features for manual sliders)
        models = models_orig
        cols = ORIGINAL_COLS
        scaler = scaler_orig
        
        if model_name not in models:
            return jsonify({"error": f"Model {model_name} not loaded for original feature set"}), 400
            
        model = models[model_name]
        
        # Build 1D raw vector matching columns exactly
        raw_vector = []
        for col in cols:
            raw_vector.append(float(feature_vals.get(col, 0.0)))
            
        raw_vector = np.array([raw_vector]) # shape (1, 12)
        
        # Scale if KNN
        if model_name == 'KNN':
            X_input = scaler.transform(pd.DataFrame(raw_vector, columns=cols))
        else:
            X_input = raw_vector
            
        # Run prediction
        prediction = int(model.predict(X_input)[0])
        
        # Get probability distribution
        try:
            probs = model.predict_proba(X_input)[0]
            probs_list = [float(p) for p in probs]
        except Exception:
            probs_list = [0.0] * 12
            probs_list[prediction] = 1.0
            
        return jsonify({
            "model_name": model_name,
            "prediction": prediction,
            "prediction_label": LITHOLOGY_LABELS[prediction],
            "prediction_color": LITHOLOGY_COLORS[prediction],
            "probabilities": probs_list
        })
    except Exception as e:
        logger.error(f"Sandbox Prediction API Error: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/upload', methods=['POST'])
def upload_las_file():
    """
    Accepts a raw LAS file upload, preprocesses it dynamically (imputing gaps,
    extracting CWT wavelets, and scaling features), runs real-time inference,
    and returns full vertical channel data for multi-track plotting.
    """
    if df_processed is None:
        return jsonify({"error": "Dataset not loaded"}), 500
        
    try:
        global cached_uploaded_well_df
        # Check if file was uploaded
        if 'file' not in request.files:
            return jsonify({"error": "No file uploaded"}), 400
            
        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "No file selected"}), 400
            
        # Securely save the temp file
        temp_dir = 'data/raw/uploads'
        os.makedirs(temp_dir, exist_ok=True)
        temp_path = os.path.join(temp_dir, 'temp_uploaded_well.las')
        file.save(temp_path)
        
        # Load and parse using lasio
        import lasio
        logger.info(f"Parsing uploaded LAS file: {file.filename}")
        las = lasio.read(temp_path)
        df = las.df().reset_index()
        
        # Standardize columns to uppercase first
        df.columns = [col.upper().strip() for col in df.columns]
        
        # Rename common depth mnemonics to DEPTH_MD if DEPTH_MD is not already present
        if 'DEPTH_MD' not in df.columns:
            if 'DEPT' in df.columns:
                df = df.rename(columns={'DEPT': 'DEPTH_MD'})
            elif 'DEPTH' in df.columns:
                df = df.rename(columns={'DEPTH': 'DEPTH_MD'})
                
        # Eliminate duplicate column names (e.g. duplicate DEPTH_MD from index + curve)
        df = df.loc[:, ~df.columns.duplicated()]
        
        # Ensure DEPTH_MD is present
        if 'DEPTH_MD' not in df.columns:
            return jsonify({"error": "Uploaded LAS file is missing a DEPTH, DEPT, or DEPTH_MD curve"}), 400
            
        # Calculate relative depth dynamically for the uploaded well
        df['RELATIVE_DEPTH'] = (df['DEPTH_MD'] - df['DEPTH_MD'].min()) / (df['DEPTH_MD'].max() - df['DEPTH_MD'].min() + 1e-8)
        
        # Populate missing columns with training dataset medians to prevent row drops
        for col in ORIGINAL_COLS:
            if col not in df.columns:
                fallback_val = float(df_processed[col].median()) if df_processed is not None else 8.5
                df[col] = fallback_val
                
        df['WELL_ID'] = 'uploaded_well'
        df['WELL'] = 'uploaded_well'
        
        # Preprocess features
        from src.features import handle_missing_values, compute_cwt_features, engineer_advanced_geological_features
        df_imputed = handle_missing_values(df, ORIGINAL_COLS)
        
        # In case entire rows were dropped, fallback to simple fillna
        if len(df_imputed) == 0:
            df_imputed = df.ffill().bfill().fillna(0.0)
            
        # Compute all advanced petrophysical and rolling stats features
        df_features = engineer_advanced_geological_features(df_imputed)
        
        # Compute CWT coefficients dynamically
        df_features = compute_cwt_features(df_features, WAVELET_LOGS)
        cached_uploaded_well_df = df_features
        
        
        # Check if ground truth lithology is present in LAS
        has_gt = False
        y_true = None
        litho_col = 'FORCE_2020_LITHOFACIES_LITHOLOGY'
        
        if litho_col in df_features.columns:
            from src.data_loader import LITHOLOGY_MAP
            df_features['LITHOLOGY'] = df_features[litho_col].map(LITHOLOGY_MAP)
            has_gt = True
        elif 'LITHOLOGY' in df_features.columns:
            has_gt = True
            
        if has_gt:
            y_true = df_features['LITHOLOGY'].fillna(-1).values.astype(int)
            
        # Select active parameters for inference
        model_name = request.form.get('model_name', 'Random Forest')
        feature_set = request.form.get('feature_set', 'wavelet')
        use_viterbi = request.form.get('use_viterbi', 'false').lower() == 'true'
        
        models = models_orig if feature_set == 'original' else models_wav
        cols = ORIGINAL_COLS if feature_set == 'original' else WAVELET_COLS
        scaler = scaler_orig if feature_set == 'original' else scaler_wav
        
        model = models.get(model_name, models['Random Forest'])
        
        # Prepare feature matrix
        X_raw = df_features[cols]
        if model_name == 'KNN':
            X_input = scaler.transform(X_raw)
        else:
            X_input = X_raw.values
            
        # Run prediction
        predictions = model.predict(X_input).astype(int)
        
        # Compute probabilities and advanced geological uncertainty
        try:
            probabilities = model.predict_proba(X_input)
            max_probs = [float(p[pred]) for p, pred in zip(probabilities, predictions)]
            entropies, top3_candidates, low_confidence_flags = calculate_uncertainty_metrics(probabilities)
        except Exception as p_err:
            logger.warning(f"Probability estimation failed, using flat fallbacks: {str(p_err)}")
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
            
        # Run Viterbi sequence smoothing
        try:
            if transition_matrix is not None:
                viterbi_predictions = viterbi_decode(probabilities, transition_matrix).astype(int).tolist()
            else:
                fallback_trans = np.eye(12) * 0.95 + (1.0 - np.eye(12)) * (0.05 / 11.0)
                viterbi_predictions = viterbi_decode(probabilities, fallback_trans).astype(int).tolist()
        except Exception as vex:
            logger.error(f"Viterbi sequence decoding failed: {str(vex)}")
            viterbi_predictions = predictions.astype(int).tolist()
            
        # Active y_pred based on use_viterbi toggle
        y_pred = np.array(viterbi_predictions) if use_viterbi else predictions
        
        # Clean up temp file
        if os.path.exists(temp_path):
            os.remove(temp_path)
            
        metrics = None
        mismatches = None
        comparison_metrics = None
        if has_gt and y_true is not None:
            from src.metrics import get_classification_metrics
            # Filter out unmapped indicators (-1)
            valid_mask = y_true >= 0
            if valid_mask.any():
                y_true_valid = y_true[valid_mask]
                metrics_res = get_classification_metrics(y_true_valid, y_pred[valid_mask], penalty_matrix)
                metrics = {
                    "accuracy": float(metrics_res["Accuracy"]),
                    "hamming_loss": float(metrics_res["HammingLoss"]),
                    "penalty_score": float(metrics_res["PenaltyScore"])
                }
                mismatches = (y_true != y_pred).astype(int).tolist()
                
                # Compute comparison_metrics for all 5 models (using downsampled dataset for cloud resource protection)
                comparison_metrics = []
                model_names = ['KNN', 'Random Forest', 'Decision Tree', 'XGBoost', 'LightGBM']
                
                if len(df_features) > 1000:
                    df_features_metrics = df_features.iloc[::10].copy()
                    y_true_metrics = y_true[::10]
                    valid_mask_metrics = y_true_metrics >= 0
                    y_true_valid_metrics = y_true_metrics[valid_mask_metrics]
                else:
                    df_features_metrics = df_features.copy()
                    valid_mask_metrics = valid_mask
                    y_true_valid_metrics = y_true_valid
                
                for name in model_names:
                    acc12, pen12, ham12 = 0.0, 0.0, 0.0
                    try:
                        if name in models_orig:
                            model12 = models_orig[name]
                            X_raw12 = df_features_metrics[ORIGINAL_COLS]
                            X_in12 = scaler_orig.transform(X_raw12) if name == 'KNN' else X_raw12.values
                            y_pred12 = model12.predict(X_in12).astype(int)[valid_mask_metrics]
                            m12 = get_classification_metrics(y_true_valid_metrics, y_pred12, penalty_matrix)
                            acc12 = float(m12["Accuracy"])
                            pen12 = float(m12["PenaltyScore"])
                            ham12 = float(m12["HammingLoss"])
                    except Exception as e12:
                        logger.error(f"Error benchmarking original model {name} on upload: {str(e12)}")
                        
                    acc19, pen19, ham19 = 0.0, 0.0, 0.0
                    try:
                        if name in models_wav:
                            model19 = models_wav[name]
                            X_raw19 = df_features_metrics[WAVELET_COLS]
                            X_in19 = scaler_wav.transform(X_raw19) if name == 'KNN' else X_raw19.values
                            y_pred19 = model19.predict(X_in19).astype(int)[valid_mask_metrics]
                            m19 = get_classification_metrics(y_true_valid_metrics, y_pred19, penalty_matrix)
                            acc19 = float(m19["Accuracy"])
                            pen19 = float(m19["PenaltyScore"])
                            ham19 = float(m19["HammingLoss"])
                    except Exception as e19:
                        logger.error(f"Error benchmarking wavelet model {name} on upload: {str(e19)}")
                        
                    comparison_metrics.append({
                        "name": name + " ★" if name == "Random Forest" else name,
                        "acc12": round(acc12, 3),
                        "pen12": round(pen12, 4),
                        "ham12": round(ham12, 4),
                        "acc19": round(acc19, 3),
                        "pen19": round(pen19, 4),
                        "ham19": round(ham19, 4)
                    })
                    
        # Downsample the visualization payload if the uploaded well is extremely large (> 5000 samples)
        # to ensure fast network transfer and ultra-smooth browser chart rendering!
        if len(df_features) > 5000:
            step = len(df_features) // 5000 + 1
            df_features_vis = df_features.iloc[::step].copy()
            predictions_vis = predictions[::step].tolist()
            viterbi_predictions_vis = np.array(viterbi_predictions)[::step].tolist()
            max_probs_vis = np.array(max_probs)[::step].tolist()
            entropies_vis = np.array(entropies)[::step].tolist()
            top3_candidates_vis = [top3_candidates[i] for i in range(0, len(top3_candidates), step)]
            low_confidence_flags_vis = np.array(low_confidence_flags)[::step].tolist()
            if mismatches is not None:
                mismatches_vis = np.array(mismatches)[::step].tolist()
            else:
                mismatches_vis = None
        else:
            df_features_vis = df_features
            predictions_vis = predictions.tolist()
            viterbi_predictions_vis = viterbi_predictions
            max_probs_vis = max_probs
            entropies_vis = entropies
            top3_candidates_vis = top3_candidates
            low_confidence_flags_vis = low_confidence_flags
            mismatches_vis = mismatches
            
        # Map values to payload lists (replacing NaNs with None for JSON compliance)
        payload = {}
        for col in df_features_vis.columns:
            if pd.api.types.is_numeric_dtype(df_features_vis[col]):
                payload[col] = [None if pd.isna(x) else float(x) for x in df_features_vis[col].values]
            else:
                payload[col] = df_features_vis[col].values.tolist()
                
        return jsonify({
            "well_id": file.filename,
            "model_name": model_name,
            "feature_set": feature_set,
            "well_data": payload,
            "predictions": predictions_vis,
            "viterbi_predictions": viterbi_predictions_vis,
            "probabilities": max_probs_vis,
            "entropies": entropies_vis,
            "top3_candidates": top3_candidates_vis,
            "low_confidence_flags": low_confidence_flags_vis,
            "has_ground_truth": has_gt,
            "mismatches": mismatches_vis,
            "metrics": metrics,
            "comparison_metrics": comparison_metrics
        })

        
    except Exception as e:
        logger.error(f"LAS Upload API Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

# ---------------------------------------------------------
# MAIN BOOTSTRAPPER
# ---------------------------------------------------------
if __name__ == '__main__':
    # Ensure plots folder mapping matches
    os.makedirs('plots', exist_ok=True)
    os.makedirs('templates', exist_ok=True)
    os.makedirs('static/css', exist_ok=True)
    os.makedirs('static/js', exist_ok=True)
    
    logger.info("Starting Lithofacies ML V1 Web Application...")
    port = int(os.environ.get("PORT", 5000))
    is_prod = "PORT" in os.environ
    app.run(host='0.0.0.0', port=port, debug=not is_prod)
