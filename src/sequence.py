import logging
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# List of geologically impossible stratigraphic transitions (symmetric)
IMPOSSIBLE_TRANSITIONS = [
    (11, 10), (10, 11),  # Basement <-> Coal
    (11, 6),  (6, 11),   # Basement <-> Chalk
    (11, 7),  (7, 11),   # Basement <-> Halite
    (11, 8),  (8, 11),   # Basement <-> Anhydrite
    (10, 6),  (6, 10),   # Coal <-> Chalk
    (10, 7),  (7, 10),   # Coal <-> Halite
    (10, 8),  (8, 10),   # Coal <-> Anhydrite
]

def estimate_transition_matrix(y_train, groups=None, epsilon=0.01):
    """
    Estimates a 12x12 state transition matrix empirically from contiguous label sequences in the training set.
    If groups (well indices) is provided, transition counts are calculated strictly within each contiguous well
    to avoid counting jumps between wells.
    
    Applies Laplacian stratigraphic smoothing to possible transitions while setting impossible transitions to zero.
    """
    logger.info("Estimating empirical 12x12 regularized stratigraphic transition matrix...")
    
    n_classes = 12
    transition_counts = np.zeros((n_classes, n_classes))
    
    if groups is not None:
        unique_groups = np.unique(groups)
        for g in unique_groups:
            mask = groups == g
            y_group = y_train[mask]
            
            # Count transitions
            for t in range(len(y_group) - 1):
                from_state = int(y_group[t])
                to_state = int(y_group[t+1])
                if 0 <= from_state < n_classes and 0 <= to_state < n_classes:
                    transition_counts[from_state, to_state] += 1
    else:
        for t in range(len(y_train) - 1):
            from_state = int(y_train[t])
            to_state = int(y_train[t+1])
            if 0 <= from_state < n_classes and 0 <= to_state < n_classes:
                transition_counts[from_state, to_state] += 1
                
    # 1. Enforce same-facies self-transition vertical continuity
    for i in range(n_classes):
        transition_counts[i, i] += 100.0
        
    # 2. Add Laplace smoothing epsilon to possible transitions only
    for i in range(n_classes):
        for j in range(n_classes):
            if (i, j) not in IMPOSSIBLE_TRANSITIONS:
                transition_counts[i, j] += epsilon
            else:
                transition_counts[i, j] = 0.0  # Strictly zero counts for geologically impossible jumps
                
    # Normalize rows to form transition probability distribution
    row_sums = transition_counts.sum(axis=1, keepdims=True)
    transition_matrix = np.where(row_sums > 0, transition_counts / row_sums, 0.0)
    
    logger.info("Empirical stratigraphic transition matrix estimated and regularized successfully.")
    return transition_matrix

def viterbi_decode(prob_sequence, transition_matrix):
    """
    Viterbi sequence decoder to find the geologically most probable lithofacies sequence.
    Input:
      - prob_sequence: array-like of shape (T, 12) containing calibrated class probabilities.
      - transition_matrix: 12x12 regularized transition probability matrix.
    Output:
      - smoothed_sequence: list/array of shape (T,) containing smoothed class indices.
    """
    T, n_classes = prob_sequence.shape
    
    # Work in log space to prevent numerical underflow
    log_trans = np.log(transition_matrix + 1e-20)
    
    # Explicitly enforce infinite penalty (-1e10) for geologically impossible transitions
    for f, t in IMPOSSIBLE_TRANSITIONS:
        log_trans[f, t] = -1e10
        
    # Clip and normalize probabilities to prevent log(0)
    eps_probs = np.clip(prob_sequence, 1e-20, 1.0)
    log_emissions = np.log(eps_probs)
    
    # Viterbi DP table: viterbi[t, j] stores maximum log probability up to depth t ending in class j
    viterbi = np.zeros((T, n_classes))
    # backpointer[t, j] stores the best previous state
    backpointer = np.zeros((T, n_classes), dtype=int)
    
    # Initial state (based on the first depth emissions)
    viterbi[0] = log_emissions[0]
    
    # DP forward pass
    for t in range(1, T):
        for j in range(n_classes):
            # For each target class j at depth t, find the maximum transition from previous states
            trans_probs = viterbi[t-1] + log_trans[:, j]
            best_prev_state = np.argmax(trans_probs)
            viterbi[t, j] = trans_probs[best_prev_state] + log_emissions[t, j]
            backpointer[t, j] = best_prev_state
            
    # Traceback pass
    smoothed_sequence = np.zeros(T, dtype=int)
    best_last_state = np.argmax(viterbi[T-1])
    smoothed_sequence[T-1] = best_last_state
    
    for t in range(T - 2, -1, -1):
        smoothed_sequence[t] = backpointer[t + 1, smoothed_sequence[t + 1]]
        
    return smoothed_sequence
