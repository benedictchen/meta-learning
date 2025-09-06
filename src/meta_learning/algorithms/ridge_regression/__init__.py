"""
Ridge Regression Module for Few-Shot Learning
============================================

Modular ridge regression implementation with:
- Core algorithm (ridge_core.py)  
- Matrix solvers (solvers.py)
- Utilities and preprocessing (utils.py)
- Integration helpers (integration.py)
"""

from .ridge_core import RidgeRegression, ridge_regression_solve
from .solvers import woodbury_solver, safe_solve, condition_number_check
from .utils import preprocess_features, select_regularization, uncertainty_estimation
from .integration import create_ridge_classifier, ridge_episode_loss

# Backward compatibility wrapper
def ridge_regression(embeddings, targets, reg_lambda=0.01, num_classes=None, **kwargs):
    """
    Backward compatible wrapper for ridge regression function.
    
    Args:
        embeddings: Input embeddings
        targets: Target values  
        reg_lambda: Regularization parameter
        num_classes: Number of classes (for classification)
        **kwargs: Additional arguments
        
    Returns:
        Tuple of (weights, bias_term)
    """
    model = RidgeRegression(reg_lambda=reg_lambda, **kwargs)
    
    # Handle classification targets
    if targets.dtype == torch.long and num_classes is not None:
        import torch.nn.functional as F
        targets = F.one_hot(targets, num_classes=num_classes).float()
    
    model.fit(embeddings, targets)
    return model.weights, model.bias_term

__all__ = [
    'RidgeRegression',
    'ridge_regression', 
    'ridge_regression_solve',
    'woodbury_solver',
    'safe_solve',
    'condition_number_check',
    'preprocess_features',
    'select_regularization', 
    'uncertainty_estimation',
    'create_ridge_classifier',
    'ridge_episode_loss'
]