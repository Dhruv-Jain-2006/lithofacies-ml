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
    
    # Calculate sum of penalty costs
    total_penalty = 0.0
    for i in range(len(y_true)):
        total_penalty += penalty_matrix[y_true_clipped[i], y_pred_clipped[i]]
        
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
