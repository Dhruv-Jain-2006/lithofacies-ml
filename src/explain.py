import os
import logging
import matplotlib.pyplot as plt
import numpy as np
import shap

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Make sure matplotlib runs headless for notebook/script execution
plt.switch_backend('Agg')

def compute_shap_explanations(model, X_test, num_samples=200):
    """
    Computes SHAP values using TreeExplainer for the given tree-based model (e.g. Random Forest).
    Uses a representative downsampled subset of the test set to optimize calculation speed.
    """
    logger.info(f"Downsampling test set to {num_samples} samples for SHAP explanations...")
    
    # Reset index and take a random sample to preserve structure
    X_sample = X_test.sample(n=min(num_samples, len(X_test)), random_state=42)
    
    logger.info("Initializing SHAP TreeExplainer...")
    explainer = shap.TreeExplainer(model)
    
    logger.info("Computing SHAP values...")
    shap_values = explainer.shap_values(X_sample)
    
    logger.info("SHAP value calculation completed.")
    return explainer, shap_values, X_sample

def plot_global_importance(shap_values, X_sample, class_names, output_path='plots/shap_global_importance.png'):
    """
    Generates and saves the multi-class global feature importance bar chart
    (replicating Figure 6 in the paper).
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.figure(figsize=(10, 8))
    
    # In SHAP multi-class, shap_values is a list of length C (num classes).
    # shap.summary_plot with a list of shap_values plots a stacked bar chart of mean absolute shap values.
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
    Generates and saves beeswarm summary plots for each of the 12 lithofacies classes
    (replicating panels a-l in Figure 7 of the paper).
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # shap_values is a list. One element per class.
    # If shap_values is a single array (e.g., in newer SHAP versions), handle appropriately
    is_list = isinstance(shap_values, list)
    num_classes = len(shap_values) if is_list else shap_values.shape[-1]
    
    logger.info(f"Generating {num_classes} per-class SHAP beeswarm plots...")
    
    for class_idx in range(num_classes):
        class_name = class_names[class_idx]
        logger.info(f"Plotting beeswarm for class {class_idx}: {class_name}")
        
        plt.figure(figsize=(10, 6))
        
        # Get the SHAP values corresponding to this specific class
        class_shap = shap_values[class_idx] if is_list else shap_values[:, :, class_idx]
        
        # Generate beeswarm summary plot
        shap.summary_plot(
            class_shap, 
            X_sample, 
            plot_type="dot", 
            show=False,
            plot_size=(10, 6)
        )
        
        plt.title(f"SHAP Beeswarm Plot — Class {class_idx}: {class_name}", fontsize=14, pad=15)
        plt.tight_layout()
        
        # Clean up filename
        safe_class_name = class_name.replace("/", "_").replace(" ", "_").lower()
        filepath = os.path.join(output_dir, f"shap_beeswarm_class_{class_idx:02d}_{safe_class_name}.png")
        
        plt.savefig(filepath, dpi=150)
        plt.close()
        
    logger.info("All per-class beeswarm plots successfully saved.")
