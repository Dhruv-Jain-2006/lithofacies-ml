import os
import logging
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Make sure matplotlib runs headless for notebook/script execution
plt.switch_backend('Agg')

def compute_shap_explanations(model, X_test, y_test=None, num_samples=200):
    """
    Computes SHAP values using TreeExplainer for the given tree-based model (e.g. Random Forest).
    Uses a representative stratified subset of the test set to ensure rare classes are represented,
    optimizing calculation speed.
    """
    logger.info(f"Downsampling test set to {num_samples} samples for SHAP explanations...")
    
    # If the model is a CalibratedClassifierCV, we extract the underlying base estimators
    # and explain the first base estimator to make it compatible with TreeExplainer
    from sklearn.calibration import CalibratedClassifierCV
    if isinstance(model, CalibratedClassifierCV):
        logger.info("Calibrated model detected. Extracting base estimator for TreeExplainer...")
        base_model = model.calibrated_classifiers_[0].estimator
    else:
        base_model = model

    if y_test is not None:
        y_test = np.array(y_test)
        unique_classes = np.unique(y_test)
        
        min_samples = max(10, num_samples // (len(unique_classes) * 2))
        
        selected_indices = []
        for c in unique_classes:
            class_indices = np.where(y_test == c)[0]
            if len(class_indices) > 0:
                sampled = np.random.RandomState(42).choice(
                    class_indices, 
                    size=min(min_samples, len(class_indices)), 
                    replace=False
                )
                selected_indices.extend(sampled)
                
        remaining = list(set(range(len(X_test))) - set(selected_indices))
        if len(selected_indices) < num_samples and len(remaining) > 0:
            fill_size = min(num_samples - len(selected_indices), len(remaining))
            filled = np.random.RandomState(42).choice(remaining, size=fill_size, replace=False)
            selected_indices.extend(filled)
            
        selected_indices = sorted(list(set(selected_indices)))
        X_sample = X_test.iloc[selected_indices].copy()
        logger.info(f"Stratified sampling completed: kept {len(X_sample)} samples.")
    else:
        X_sample = X_test.sample(n=min(num_samples, len(X_test)), random_state=42)
        
    logger.info("Initializing SHAP TreeExplainer...")
    explainer = shap.TreeExplainer(base_model)
    
    logger.info("Computing SHAP values...")
    shap_values = explainer.shap_values(X_sample)
    
    logger.info("SHAP value calculation completed.")
    return explainer, shap_values, X_sample

def plot_global_importance(shap_values, X_sample, class_names, output_path='plots/shap_global_importance.png'):
    """
    Generates and saves the multi-class global feature importance bar chart.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.figure(figsize=(10, 8))
    
    shap.summary_plot(
        shap_values, 
        X_sample, 
        class_names=class_names, 
        show=False,
        plot_size=(10, 8)
    )
    
    plt.title("FORCE 2020 Multi-Class Global Feature Importance (SHAP)", fontsize=14, pad=15)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    logger.info(f"Global SHAP feature importance plot saved to {output_path}")

def plot_per_class_beeswarm(shap_values, X_sample, class_names, output_dir='plots'):
    """
    Generates and saves beeswarm summary plots for each of the 12 lithofacies classes.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    is_list = isinstance(shap_values, list)
    num_classes = len(shap_values) if is_list else shap_values.shape[-1]
    
    logger.info(f"Generating {num_classes} per-class SHAP beeswarm plots...")
    
    for class_idx in range(num_classes):
        class_name = class_names[class_idx]
        logger.info(f"Plotting beeswarm for class {class_idx}: {class_name}")
        
        plt.figure(figsize=(10, 6))
        
        class_shap = shap_values[class_idx] if is_list else shap_values[:, :, class_idx]
        
        shap.summary_plot(
            class_shap, 
            X_sample, 
            plot_type="dot", 
            show=False,
            plot_size=(10, 6)
        )
        
        plt.title(f"SHAP Beeswarm Plot — Class {class_idx}: {class_name}", fontsize=14, pad=15)
        plt.tight_layout()
        
        safe_class_name = class_name.replace("/", "_").replace(" ", "_").lower()
        filepath = os.path.join(output_dir, f"shap_beeswarm_class_{class_idx:02d}_{safe_class_name}.png")
        
        plt.savefig(filepath, dpi=150)
        plt.close()
        
    logger.info("All per-class beeswarm plots successfully saved.")

def plot_class_beeswarm(shap_values, X_sample, class_index, class_name, output_path='plots/shap_beeswarm.png'):
    """
    Plots the beeswarm summary diagram for a specific target class.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.figure(figsize=(10, 6))
    
    is_list = isinstance(shap_values, list)
    class_shap = shap_values[class_index] if is_list else shap_values[:, :, class_index]
    
    shap.summary_plot(
        class_shap, 
        X_sample, 
        plot_type="dot", 
        show=False,
        plot_size=(10, 6)
    )
    
    plt.title(f"SHAP Beeswarm Summary — Class {class_index}: {class_name}", fontsize=14, pad=15)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    logger.info(f"Class SHAP beeswarm plot for {class_name} saved to {output_path}")

def generate_shap_drift_report(trained_models_dict, y_test, output_path='reports/shap_drift_report.csv'):
    """
    Compares global SHAP feature importance percentages between different feature configurations
    to produce a semantic feature importance drift report, showing that raw physical log curves 
    are not drowned out by wavelets and engineered features.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    logger.info("Generating SHAP feature drift and control report...")
    
    drift_data = []
    
    for config_name, (models, feature_list, _, df_test_imp, _, _) in trained_models_dict.items():
        if "KNN" in config_name or "Ensemble" in config_name:
            continue
            
        logger.info(f"Analyzing SHAP drift for configuration: {config_name}")
        
        # Use Random Forest (calibrated or base) for robust multi-class tree explanations
        rf_model = models.get("Random Forest")
        if rf_model is None:
            continue
            
        try:
            # Fit tree explainer
            _, shap_values, X_sample = compute_shap_explanations(
                rf_model, df_test_imp[feature_list], y_test, num_samples=100
            )
            
            is_list = isinstance(shap_values, list)
            n_classes = len(shap_values) if is_list else shap_values.shape[-1]
            
            # Compute mean absolute SHAP value for each feature
            if is_list:
                mean_abs_shap = np.mean([np.mean(np.abs(shap_values[c]), axis=0) for c in range(n_classes)], axis=0)
            else:
                mean_abs_shap = np.mean(np.mean(np.abs(shap_values), axis=0), axis=-1)
                
            total_shap = np.sum(mean_abs_shap) + 1e-15
            
            for i, feat in enumerate(feature_list):
                abs_val = mean_abs_shap[i]
                pct_contrib = (abs_val / total_shap) * 100.0
                
                # Classify feature family
                if feat in ['GR', 'RHOB', 'NPHI', 'DTC', 'SP', 'PEF', 'RDEP', 'RMED', 'RSHA', 'CALI', 'BS', 'RELATIVE_DEPTH']:
                    family = "Raw Physical Log"
                elif any(x in feat for x in ['roll', 'mean', 'std', 'var', 'gradient']):
                    family = "Contextual Rolling Stats"
                elif "CWT" in feat:
                    family = "Wavelet Coefficients"
                else:
                    family = "Petrophysical Engineered"
                    
                drift_data.append({
                    "FeatureSpace": config_name,
                    "FeatureName": feat,
                    "FeatureFamily": family,
                    "MeanAbsSHAP": abs_val,
                    "PercentageContribution": pct_contrib
                })
        except Exception as e:
            logger.error(f"Failed to generate SHAP values for {config_name}: {str(e)}")
            
    if len(drift_data) == 0:
        logger.warning("No SHAP drift data generated!")
        return
        
    df_drift = pd.DataFrame(drift_data)
    df_drift.to_csv(output_path, index=False)
    logger.info(f"SHAP feature drift report successfully saved to {output_path}!")
    
    # Generate publication-ready summary of raw log preservation
    raw_logs = df_drift[df_drift['FeatureFamily'] == 'Raw Physical Log']
    logger.info("Preservation summary for raw physical log curves in Combined feature sets:")
    for space in raw_logs['FeatureSpace'].unique():
        space_df = raw_logs[raw_logs['FeatureSpace'] == space]
        tot_pct = space_df['PercentageContribution'].sum()
        logger.info(f" -> Feature Space '{space}': Raw physical curves hold {tot_pct:.2f}% of total SHAP attribution.")
