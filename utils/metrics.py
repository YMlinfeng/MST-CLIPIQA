import numpy as np
from scipy.stats import spearmanr, pearsonr

def compute_metrics(preds, targets):
    """
    Computes Spearman Rank Correlation Coefficient (SRCC) and 
    Pearson Linear Correlation Coefficient (PLCC).
    
    Args:
        preds (list or np.ndarray): Predicted scores.
        targets (list or np.ndarray): Ground truth MOS scores.
        
    Returns:
        dict: Dictionary containing 'srcc' and 'plcc' values.
    """
    preds = np.array(preds).flatten()
    targets = np.array(targets).flatten()
    
    if len(preds) < 2:
        return {'srcc': 0.0, 'plcc': 0.0}
        
    srcc, _ = spearmanr(preds, targets)
    plcc, _ = pearsonr(preds, targets)
    
    return {
        'srcc': float(srcc),
        'plcc': float(plcc)
    }
