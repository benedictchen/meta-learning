"""
Ridge Regression Utilities
==========================

Feature preprocessing, regularization selection, and uncertainty estimation
for ridge regression in few-shot learning scenarios.
"""

from __future__ import annotations
import torch
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional, Dict, Any
import warnings


def preprocess_features(X: torch.Tensor, method: str = 'standardize',
                       fit_params: Optional[Dict[str, torch.Tensor]] = None
                       ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    Preprocess features for ridge regression.
    
    Args:
        X: Input features [n_samples, n_features]
        method: Preprocessing method ('standardize', 'normalize', 'none')
        fit_params: Pre-computed parameters for consistent preprocessing
        
    Returns:
        Tuple of (preprocessed_X, preprocessing_params)
    """
    if method == 'none':
        return X, {}
    
    # Compute or use existing parameters
    if fit_params is None:
        if method == 'standardize':
            mean = X.mean(dim=0, keepdim=True)
            std = X.std(dim=0, keepdim=True, unbiased=False)
            # Avoid division by zero
            std = torch.clamp(std, min=1e-8)
            params = {'mean': mean, 'std': std}
            
        elif method == 'normalize':
            min_val = X.min(dim=0, keepdim=True)[0]
            max_val = X.max(dim=0, keepdim=True)[0]
            # Avoid division by zero
            range_val = torch.clamp(max_val - min_val, min=1e-8)
            params = {'min': min_val, 'max': max_val, 'range': range_val}
            
        else:
            raise ValueError(f"Unknown preprocessing method: {method}")
    else:
        params = fit_params
    
    # Apply preprocessing
    if method == 'standardize':
        X_processed = (X - params['mean']) / params['std']
    elif method == 'normalize':
        X_processed = (X - params['min']) / params['range']
    else:
        X_processed = X
        
    # Check for NaN/inf after preprocessing
    if torch.isnan(X_processed).any() or torch.isinf(X_processed).any():
        warnings.warn("Preprocessing produced NaN/inf values, using original data")
        return X, {}
    
    return X_processed, params


def select_regularization(X: torch.Tensor, Y: torch.Tensor,
                         method: str = 'cv',
                         lambda_range: Tuple[float, float] = (1e-6, 1e2),
                         n_candidates: int = 20) -> float:
    """
    Automatically select regularization parameter.
    
    Args:
        X: Input features [n_samples, n_features]
        Y: Target values [n_samples, n_outputs]
        method: Selection method ('cv', 'gcv', 'fixed')
        lambda_range: Range of lambda values to consider
        n_candidates: Number of candidate values to try
        
    Returns:
        Selected regularization parameter
    """
    if method == 'fixed':
        return 0.01  # Default fixed value
    
    n_samples, n_features = X.shape
    
    # Generate candidate lambda values
    log_min, log_max = np.log10(lambda_range[0]), np.log10(lambda_range[1])
    log_lambdas = np.linspace(log_min, log_max, n_candidates)
    lambda_candidates = torch.tensor([10**log_lam for log_lam in log_lambdas], 
                                   device=X.device)
    
    if method == 'cv':
        return _cross_validation_lambda(X, Y, lambda_candidates)
    elif method == 'gcv':
        return _generalized_cv_lambda(X, Y, lambda_candidates)
    else:
        raise ValueError(f"Unknown regularization selection method: {method}")


def _cross_validation_lambda(X: torch.Tensor, Y: torch.Tensor,
                            lambda_candidates: torch.Tensor,
                            n_folds: int = 5) -> float:
    """Cross-validation based lambda selection."""
    from .solvers import woodbury_solver
    
    n_samples = X.shape[0]
    if n_samples < n_folds:
        n_folds = max(2, n_samples // 2)
    
    fold_size = n_samples // n_folds
    best_lambda = lambda_candidates[0].item()
    best_error = float('inf')
    
    for lam in lambda_candidates:
        cv_errors = []
        
        for fold in range(n_folds):
            # Create fold splits
            start_idx = fold * fold_size
            end_idx = start_idx + fold_size if fold < n_folds - 1 else n_samples
            
            # Validation set
            val_mask = torch.zeros(n_samples, dtype=torch.bool, device=X.device)
            val_mask[start_idx:end_idx] = True
            
            # Training set
            train_mask = ~val_mask
            
            if train_mask.sum() == 0 or val_mask.sum() == 0:
                continue
                
            X_train, Y_train = X[train_mask], Y[train_mask]
            X_val, Y_val = X[val_mask], Y[val_mask]
            
            try:
                # Train with this lambda
                weights = woodbury_solver(X_train, Y_train, lam.item())
                
                # Validate
                Y_pred = torch.mm(X_val, weights)
                error = torch.mean((Y_pred - Y_val) ** 2).item()
                cv_errors.append(error)
                
            except Exception:
                # Skip this fold if numerical issues occur
                continue
        
        if cv_errors:
            avg_error = np.mean(cv_errors)
            if avg_error < best_error:
                best_error = avg_error
                best_lambda = lam.item()
    
    return best_lambda


def _generalized_cv_lambda(X: torch.Tensor, Y: torch.Tensor,
                          lambda_candidates: torch.Tensor) -> float:
    """Generalized cross-validation based lambda selection."""
    from .solvers import woodbury_solver
    
    n_samples, n_features = X.shape
    best_lambda = lambda_candidates[0].item()
    best_gcv = float('inf')
    
    for lam in lambda_candidates:
        try:
            # Compute weights
            weights = woodbury_solver(X, Y, lam.item())
            
            # Compute predictions
            Y_pred = torch.mm(X, weights)
            
            # Compute effective degrees of freedom
            # For ridge: df = tr(X(X^T X + λI)^-1 X^T)
            XTX = torch.mm(X.t(), X)
            XTX_reg = XTX + lam * torch.eye(n_features, device=X.device)
            
            # Efficient computation of trace
            try:
                XTX_inv = torch.linalg.inv(XTX_reg)
                H_diag = torch.sum(X * torch.mm(X, XTX_inv), dim=1)  # Diagonal of hat matrix
                df = H_diag.sum().item()
            except:
                # Fallback: use approximate df
                df = n_features * n_samples / (n_samples + lam.item())
            
            # GCV score: RSS / (n * (1 - df/n)^2)
            rss = torch.sum((Y - Y_pred) ** 2).item()
            
            if df >= n_samples:
                gcv_score = float('inf')
            else:
                gcv_score = rss / (n_samples * (1 - df / n_samples) ** 2)
            
            if gcv_score < best_gcv:
                best_gcv = gcv_score
                best_lambda = lam.item()
                
        except Exception:
            # Skip this lambda if numerical issues occur
            continue
    
    return best_lambda


def uncertainty_estimation(X: torch.Tensor, weights: torch.Tensor,
                          noise_var: Optional[float] = None,
                          reg_lambda: float = 0.01) -> Dict[str, torch.Tensor]:
    """
    Estimate prediction uncertainty for ridge regression.
    
    Args:
        X: Input features [n_samples, n_features]  
        weights: Ridge regression weights [n_features, n_outputs]
        noise_var: Estimated noise variance (auto-estimated if None)
        reg_lambda: Regularization parameter used in training
        
    Returns:
        Dictionary with uncertainty estimates
    """
    n_samples, n_features = X.shape
    n_outputs = weights.shape[1]
    
    try:
        # Compute covariance matrix: σ² (X^T X + λI)^-1
        XTX = torch.mm(X.t(), X)
        XTX_reg = XTX + reg_lambda * torch.eye(n_features, device=X.device)
        
        # Compute inverse for covariance
        try:
            XTX_inv = torch.linalg.inv(XTX_reg)
        except:
            # Fallback to pseudoinverse
            XTX_inv = torch.linalg.pinv(XTX_reg)
        
        # Estimate noise variance if not provided
        if noise_var is None:
            Y_pred = torch.mm(X, weights)
            # Use a dummy Y for noise estimation (assuming unit variance)
            noise_var = 1.0
        
        # Prediction covariance: σ² X (X^T X + λI)^-1 X^T
        pred_cov_diag = noise_var * torch.sum(X * torch.mm(X, XTX_inv), dim=1)
        pred_std = torch.sqrt(torch.clamp(pred_cov_diag, min=1e-8))
        
        # Parameter covariance diagonal: σ² diag((X^T X + λI)^-1)
        param_var_diag = noise_var * torch.diag(XTX_inv)
        param_std = torch.sqrt(torch.clamp(param_var_diag, min=1e-8))
        
        return {
            'prediction_std': pred_std.unsqueeze(1).expand(-1, n_outputs),
            'parameter_std': param_std.unsqueeze(1).expand(-1, n_outputs),
            'covariance_matrix': XTX_inv,
            'noise_variance': noise_var
        }
        
    except Exception as e:
        warnings.warn(f"Uncertainty estimation failed: {e}, returning zero uncertainty")
        
        return {
            'prediction_std': torch.zeros(n_samples, n_outputs, device=X.device),
            'parameter_std': torch.zeros(n_features, n_outputs, device=X.device), 
            'covariance_matrix': torch.eye(n_features, device=X.device),
            'noise_variance': 1.0
        }


def compute_confidence_intervals(predictions: torch.Tensor,
                               prediction_std: torch.Tensor,
                               confidence_level: float = 0.95) -> Dict[str, torch.Tensor]:
    """
    Compute confidence intervals for predictions.
    
    Args:
        predictions: Model predictions [n_samples, n_outputs]
        prediction_std: Prediction standard deviations [n_samples, n_outputs]
        confidence_level: Confidence level (e.g., 0.95 for 95%)
        
    Returns:
        Dictionary with confidence bounds
    """
    from scipy import stats
    
    # Z-score for confidence level
    alpha = 1 - confidence_level
    z_score = stats.norm.ppf(1 - alpha / 2)
    
    margin_of_error = z_score * prediction_std
    
    return {
        'lower_bound': predictions - margin_of_error,
        'upper_bound': predictions + margin_of_error,
        'margin_of_error': margin_of_error,
        'confidence_level': confidence_level
    }


def ridge_feature_importance(weights: torch.Tensor,
                           feature_names: Optional[list] = None) -> Dict[str, Any]:
    """
    Compute feature importance for ridge regression.
    
    Args:
        weights: Ridge regression weights [n_features, n_outputs]
        feature_names: Optional feature names
        
    Returns:
        Dictionary with importance metrics
    """
    n_features, n_outputs = weights.shape
    
    # Compute various importance metrics
    abs_weights = torch.abs(weights)
    
    # L2 norm of weights per feature (across outputs)
    l2_importance = torch.norm(weights, dim=1)
    
    # Mean absolute weight per feature
    mean_abs_importance = abs_weights.mean(dim=1)
    
    # Maximum absolute weight per feature  
    max_abs_importance = abs_weights.max(dim=1)[0]
    
    # Rank features by L2 importance
    _, importance_ranks = torch.sort(l2_importance, descending=True)
    
    importance_dict = {
        'l2_importance': l2_importance,
        'mean_abs_importance': mean_abs_importance,
        'max_abs_importance': max_abs_importance,
        'importance_ranks': importance_ranks,
        'weights': weights
    }
    
    if feature_names is not None:
        if len(feature_names) == n_features:
            # Create ranked feature names
            ranked_features = [feature_names[i] for i in importance_ranks.tolist()]
            importance_dict['ranked_feature_names'] = ranked_features
        else:
            warnings.warn("Feature names length doesn't match number of features")
    
    return importance_dict