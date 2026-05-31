import os
import logging
import urllib.request
import numpy as np
from sklearn.metrics import accuracy_score, hamming_loss

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Official raw URL of the FORCE 2020 penalty matrix
PENALTY_MATRIX_URL = "https://raw.githubusercontent.com/bolgebrygg/Force-2020-Machine-Learning-competition/master/lithology_competition/data/penalty_matrix.npy"
LOCAL_PENALTY_PATH = "data/penalty_matrix.npy"

# Reconstructed geologically-weighted 12x12 fallback penalty matrix
# Values reflect geoscience similarity. Confusion of dissimilar rock types (e.g. Sandstone vs Halite)
# incurs a higher cost than similar types (e.g. Sandstone vs Sandstone/Shale).
# Columns/Rows: 0: Sandstone, 1: Sandstone/Shale, 2: Shale, 3: Marl, 4: Dolomite, 5: Limestone,
#               6: Chalk, 7: Halite, 8: Anhydrite, 9: Tuff, 10: Coal, 11: Basement
FALLBACK_PENALTY_MATRIX = np.array([
    [0.0, 0.5, 1.0, 1.2, 1.2, 1.0, 1.0, 2.0, 2.0, 1.0, 1.5, 2.0],  # 0: Sandstone
    [0.5, 0.0, 0.5, 0.8, 1.2, 1.0, 1.0, 2.0, 2.0, 0.8, 1.2, 2.0],  # 1: Sandstone/Shale
    [1.0, 0.5, 0.0, 0.5, 1.2, 1.0, 1.0, 2.0, 2.0, 0.8, 1.2, 2.0],  # 2: Shale
    [1.2, 0.8, 0.5, 0.0, 1.0, 0.8, 0.6, 2.0, 1.8, 1.0, 1.5, 2.0],  # 3: Marl
    [1.2, 1.2, 1.2, 1.0, 0.0, 0.4, 0.6, 1.8, 1.5, 1.2, 2.0, 2.0],  # 4: Dolomite
    [1.0, 1.0, 1.0, 0.8, 0.4, 0.0, 0.4, 1.8, 1.5, 1.2, 2.0, 2.0],  # 5: Limestone
    [1.0, 1.0, 1.0, 0.6, 0.6, 0.4, 0.0, 2.0, 1.8, 1.2, 2.0, 2.0],  # 6: Chalk
    [2.0, 2.0, 2.0, 2.0, 1.8, 1.8, 2.0, 0.0, 0.8, 2.0, 2.0, 2.5],  # 7: Halite
    [2.0, 2.0, 2.0, 1.8, 1.5, 1.5, 1.8, 0.8, 0.0, 2.0, 2.0, 2.5],  # 8: Anhydrite
    [1.0, 0.8, 0.8, 1.0, 1.2, 1.2, 1.2, 2.0, 2.0, 0.0, 1.5, 1.8],  # 9: Tuff
    [1.5, 1.2, 1.2, 1.5, 2.0, 2.0, 2.0, 2.0, 2.0, 1.5, 0.0, 2.5],  # 10: Coal
    [2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.5, 2.5, 1.8, 2.5, 0.0]   # 11: Basement
])

def load_penalty_matrix():
    """
    Attempts to download and load the official FORCE 2020 penalty matrix.
    If offline or if downloading fails, automatically returns the geological fallback matrix.
    """
    os.makedirs(os.path.dirname(LOCAL_PENALTY_PATH), exist_ok=True)
    
    # Try local load first
    if os.path.exists(LOCAL_PENALTY_PATH):
        try:
            matrix = np.load(LOCAL_PENALTY_PATH)
            logger.info("Loaded penalty matrix from local cache.")
            return matrix
        except Exception as e:
            logger.warning(f"Error reading cached penalty matrix: {str(e)}. Re-downloading.")
            
    # Try downloading
    try:
        logger.info(f"Downloading official FORCE 2020 penalty matrix from {PENALTY_MATRIX_URL}...")
        
        # Configure download request with a User-Agent to prevent standard bot blocks
        req = urllib.request.Request(
            PENALTY_MATRIX_URL, 
            headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
        )
        
        with urllib.request.urlopen(req, timeout=10) as response:
            content = response.read()
            with open(LOCAL_PENALTY_PATH, 'wb') as f:
                f.write(content)
                
        matrix = np.load(LOCAL_PENALTY_PATH)
        logger.info("Successfully downloaded and cached official penalty matrix.")
        return matrix
    except Exception as e:
        logger.error(f"Failed to download official penalty matrix: {str(e)}.")
        logger.warning("Using geologically coherent 12x12 fallback penalty matrix instead.")
        return FALLBACK_PENALTY_MATRIX

def calculate_penalty_score(y_true, y_pred, penalty_matrix=None):
    """
    Computes the domain-specific penalty matrix score as defined in the paper:
    Score = - (1 / N) * Sum( A[y_true_i, y_pred_i] )
    Range is typically negative (higher is better, 0.0 is perfect prediction).
    """
    if penalty_matrix is None:
        penalty_matrix = load_penalty_matrix()
        
    y_true = np.array(y_true, dtype=int)
    y_pred = np.array(y_pred, dtype=int)
    
    # Ensure labels are bounded by matrix shape
    max_label = penalty_matrix.shape[0] - 1
    y_true_clipped = np.clip(y_true, 0, max_label)
    y_pred_clipped = np.clip(y_pred, 0, max_label)
    
    # Calculate sum of penalty costs using fast NumPy vectorization
    penalties = penalty_matrix[y_true_clipped, y_pred_clipped]
    total_penalty = float(np.sum(penalties))
        
    # Return average negative penalty
    score = -total_penalty / len(y_true)
    return score

def get_classification_metrics(y_true, y_pred, penalty_matrix=None):
    """
    Returns all three metrics specified by the paper: Jaccard Accuracy,
    Hamming Loss, and Penalty Matrix Score.
    """
    accuracy = accuracy_score(y_true, y_pred)
    h_loss = hamming_loss(y_true, y_pred)
    penalty_score = calculate_penalty_score(y_true, y_pred, penalty_matrix)
    
    return {
        "Accuracy": accuracy,
        "HammingLoss": h_loss,
        "PenaltyScore": penalty_score
    }

def calculate_ece(y_true, y_probs, n_bins=10):
    """
    Computes Expected Calibration Error (ECE) for multi-class classification.
    y_true: 1D array of true labels of shape (N,)
    y_probs: 2D array of class probabilities of shape (N, C)
    """
    y_true = np.array(y_true, dtype=int)
    y_probs = np.array(y_probs)
    
    N = len(y_true)
    confidences = np.max(y_probs, axis=1)
    predictions = np.argmax(y_probs, axis=1)
    accuracies = (predictions == y_true)
    
    ece = 0.0
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    
    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]
        
        # Find indices of samples in this confidence bin
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop_in_bin = np.mean(in_bin)
        
        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(accuracies[in_bin])
            avg_confidence_in_bin = np.mean(confidences[in_bin])
            ece += prop_in_bin * np.abs(accuracy_in_bin - avg_confidence_in_bin)
            
    return ece

def generate_geological_confusion_report(y_true, y_pred, penalty_matrix=None, output_path='plots/geological_confusion_report.txt'):
    """
    Generates a detailed geological confusion analysis classifying errors by their geological plausibility
    (High, Medium, and Low Plausibility/Severe geological penalties).
    """
    if penalty_matrix is None:
        penalty_matrix = load_penalty_matrix()
        
    y_true = np.array(y_true, dtype=int)
    y_pred = np.array(y_pred, dtype=int)
    
    LITHOLOGY_LABELS = [
        'Sandstone', 'Sandstone/Shale', 'Shale', 'Marl', 'Dolomite', 'Limestone',
        'Chalk', 'Halite', 'Anhydrite', 'Tuff', 'Coal', 'Basement'
    ]
    
    total_samples = len(y_true)
    total_errors = np.sum(y_true != y_pred)
    
    if total_errors == 0:
        return "No errors found to report."
        
    high_plausible_count = 0
    med_plausible_count = 0
    severe_penalty_count = 0
    
    rare_classes = [9, 10, 11] # Tuff, Coal, Basement
    rare_confusion_log = []
    
    for i in range(len(y_true)):
        t = y_true[i]
        p = y_pred[i]
        if t != p:
            penalty = penalty_matrix[t, p]
            if penalty <= 0.5:
                high_plausible_count += 1
            elif penalty <= 1.2:
                med_plausible_count += 1
            else:
                severe_penalty_count += 1
                
            if t in rare_classes or p in rare_classes:
                rare_confusion_log.append(
                    f"Sample {i}: True '{LITHOLOGY_LABELS[t]}' confused with Predicted '{LITHOLOGY_LABELS[p]}' (Penalty Cost: {penalty})"
                )
                
    # Save text report
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        f.write("=== GEOLOGICAL CONFUSION PLAUSIBILITY REPORT ===\n")
        f.write(f"Total Test Samples: {total_samples}\n")
        f.write(f"Total Misclassifications: {total_errors} ({total_errors/total_samples*100:.2f}% error rate)\n\n")
        f.write(f"High Plausibility Confusions (Cost <= 0.5): {high_plausible_count} ({high_plausible_count/total_errors*100:.2f}% of errors)\n")
        f.write(f"Medium Plausibility Confusions (0.5 < Cost <= 1.2): {med_plausible_count} ({med_plausible_count/total_errors*100:.2f}% of errors)\n")
        f.write(f"Severe geological penalty / Implausible Confusions (Cost > 1.2): {severe_penalty_count} ({severe_penalty_count/total_errors*100:.2f}% of errors)\n\n")
        f.write("--- RARE CLASS LITHOLOGY CONFUSION SAMPLES ---\n")
        if len(rare_confusion_log) == 0:
            f.write("No rare class confusions found.\n")
        else:
            for log_line in rare_confusion_log[:50]: # Limit to top 50 samples
                f.write("- " + log_line + "\n")
            if len(rare_confusion_log) > 50:
                f.write(f"... and {len(rare_confusion_log) - 50} more rare class confusion incidents.\n")
                
    logger.info(f"Geological confusion plausibility report saved to {output_path}")
    return output_path

