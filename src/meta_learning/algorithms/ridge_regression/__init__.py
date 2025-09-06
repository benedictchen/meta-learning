"""
Ridge Regression Module for Few-Shot Learning
============================================

Modular ridge regression implementation with:
- Core algorithm (ridge_core.py)  
- Matrix solvers (solvers.py)
- Utilities and preprocessing (utils.py)
- Integration helpers (integration.py)
"""

import torch
import torch.nn.functional as F

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
        targets = F.one_hot(targets, num_classes=num_classes).float()
    
    model.fit(embeddings, targets)
    return model.weights, model.bias_term

# Additional utility functions
def safe_ridge_regression(embeddings, targets, reg_lambda=0.01, **kwargs):
    """Safe wrapper for ridge regression with error handling."""
    try:
        return ridge_regression(embeddings, targets, reg_lambda, **kwargs)
    except Exception as e:
        import warnings
        warnings.warn(f"Ridge regression failed: {e}, returning zero weights")
        n_features = embeddings.shape[1]
        n_targets = targets.shape[1] if targets.dim() > 1 else 1
        weights = torch.zeros(n_features, n_targets, device=embeddings.device)
        bias = torch.zeros(1, n_targets, device=embeddings.device) if kwargs.get('bias', True) else None
        return weights, bias

def ridge_cross_validation(embeddings, targets, lambda_candidates=None, n_folds=5):
    """Cross-validation for optimal regularization parameter selection."""
    if lambda_candidates is None:
        lambda_candidates = [1e-6, 1e-4, 1e-2, 1e-1, 1.0, 10.0, 100.0]
    
    n_samples = embeddings.shape[0]
    fold_size = n_samples // n_folds
    best_lambda = lambda_candidates[0]
    best_score = float('inf')
    
    for reg_lambda in lambda_candidates:
        cv_scores = []
        
        for fold in range(n_folds):
            # Split data
            start_idx = fold * fold_size
            end_idx = start_idx + fold_size if fold < n_folds - 1 else n_samples
            
            # Validation set
            val_embed = embeddings[start_idx:end_idx]
            val_targets = targets[start_idx:end_idx]
            
            # Training set
            train_embed = torch.cat([embeddings[:start_idx], embeddings[end_idx:]], dim=0)
            train_targets = torch.cat([targets[:start_idx], targets[end_idx:]], dim=0)
            
            # Train and validate
            try:
                weights, bias = ridge_regression(train_embed, train_targets, reg_lambda)
                
                # Predict
                if bias is not None:
                    val_embed_bias = torch.cat([val_embed, torch.ones(val_embed.shape[0], 1)], dim=1)
                    weights_full = torch.cat([weights, bias], dim=0)
                    pred = val_embed_bias @ weights_full
                else:
                    pred = val_embed @ weights
                
                # Compute MSE
                mse = F.mse_loss(pred, val_targets.float())
                cv_scores.append(mse.item())
                
            except Exception:
                cv_scores.append(float('inf'))
        
        import numpy as np
        avg_score = np.mean(cv_scores)
        if avg_score < best_score:
            best_score = avg_score
            best_lambda = reg_lambda
    
    return best_lambda

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
    'ridge_episode_loss',
    'safe_ridge_regression',
    'ridge_cross_validation'
]