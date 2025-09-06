"""
Ridge Regression Algorithm for Few-Shot Learning
===============================================

Closed-form ridge regression implementation with Woodbury formula optimization.
Provides fast, analytically exact solutions for linear few-shot learning tasks.

This module has been fully implemented with modular architecture in the
ridge_regression/ subdirectory. All TODOs have been completed.
"""

from __future__ import annotations
from typing import Optional, Union, Tuple, Dict, Any
import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings
import numpy as np
from ..core.episode import Episode

# Import from modular implementation in ridge_regression/ subdirectory
from .ridge_regression.ridge_core import RidgeRegression as _RidgeRegression
from .ridge_regression.ridge_core import ridge_regression_solve, create_ridge_classifier as _create_ridge_classifier
from .ridge_regression.solvers import woodbury_solver, safe_solve, condition_number_check
from .ridge_regression.utils import preprocess_features, select_regularization, uncertainty_estimation
from .ridge_regression.integration import create_ridge_classifier, ridge_episode_loss


class RidgeRegression(_RidgeRegression):
    """
    Ridge regression implementation for few-shot learning.
    
    Provides fast, analytically exact solutions using closed-form ridge regression
    with automatic Woodbury formula optimization for efficiency.
    
    This class extends the modular implementation with Episode compatibility.
    """
    
    def forward(self, episode: Episode) -> torch.Tensor:
        """
        Forward pass using closed-form ridge regression solution.
        
        Args:
            episode: Episode containing support and query data
            
        Returns:
            Query predictions (logits for classification)
        """
        # Extract support and query embeddings
        support_embeddings = episode.support_x
        support_labels = episode.support_y
        query_embeddings = episode.query_x
        
        # Determine number of classes
        num_classes = len(torch.unique(support_labels))
        
        # Convert labels to one-hot for regression
        targets = F.one_hot(support_labels, num_classes=num_classes).float()
        
        # Fit ridge regression and predict
        self.fit(support_embeddings, targets)
        query_logits = self.predict(query_embeddings)
        
        return query_logits


def ridge_regression(
    embeddings: torch.Tensor,
    targets: torch.Tensor, 
    reg_lambda: Union[float, torch.Tensor],
    num_classes: Optional[int] = None,
    use_woodbury: Optional[bool] = None,
    preprocessing: str = 'standardize',
    bias: bool = True
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    Closed-form ridge regression solution.
    
    Args:
        embeddings: Input embeddings [n_samples, n_features]
        targets: Target values [n_samples] or [n_samples, n_targets]
        reg_lambda: Regularization parameter
        num_classes: Number of classes (for classification)
        use_woodbury: Whether to use Woodbury formula (auto if None)
        preprocessing: Feature preprocessing method
        bias: Whether to add bias term
        
    Returns:
        Tuple of (weights, bias_term)
    """
    model = RidgeRegression(
        reg_lambda=reg_lambda,
        use_woodbury=use_woodbury,
        preprocessing=preprocessing,
        bias=bias
    )
    
    # Handle classification targets
    if targets.dtype == torch.long and num_classes is not None:
        targets = F.one_hot(targets, num_classes=num_classes).float()
    
    model.fit(embeddings, targets)
    return model.weights, model.bias_term


def safe_ridge_regression(
    embeddings: torch.Tensor,
    targets: torch.Tensor,
    reg_lambda: float = 0.01,
    **kwargs
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    Safe wrapper for ridge regression with error handling.
    
    Args:
        embeddings: Input embeddings
        targets: Target values
        reg_lambda: Regularization parameter
        **kwargs: Additional arguments for ridge_regression
        
    Returns:
        Tuple of (weights, bias_term) or fallback to zeros
    """
    try:
        return ridge_regression(embeddings, targets, reg_lambda, **kwargs)
    except Exception as e:
        warnings.warn(f"Ridge regression failed: {e}, returning zero weights")
        n_features = embeddings.shape[1]
        n_targets = targets.shape[1] if targets.dim() > 1 else 1
        weights = torch.zeros(n_features, n_targets, device=embeddings.device)
        bias = torch.zeros(1, n_targets, device=embeddings.device) if kwargs.get('bias', True) else None
        return weights, bias


def ridge_cross_validation(
    embeddings: torch.Tensor,
    targets: torch.Tensor,
    lambda_candidates: list = None,
    n_folds: int = 5
) -> float:
    """
    Cross-validation for optimal regularization parameter selection.
    
    Args:
        embeddings: Input embeddings
        targets: Target values
        lambda_candidates: List of lambda values to test
        n_folds: Number of CV folds
        
    Returns:
        Optimal lambda value
    """
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
        
        avg_score = np.mean(cv_scores)
        if avg_score < best_score:
            best_score = avg_score
            best_lambda = reg_lambda
    
    return best_lambda