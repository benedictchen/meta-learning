"""
💰 SUPPORT THIS RESEARCH - PLEASE DONATE! 💰

🙏 If this library helps your research or project, please consider donating:
💳 PayPal: https://www.paypal.com/cgi-bin/webscr?cmd=_s-xclick&hosted_button_id=WXQKYYKPHWXHS
⭐ GitHub Sponsors: https://github.com/sponsors/benedictchen

👨‍💻 Author: Benedict Chen
💰 Donations: Help support this work! Buy me a coffee ☕, beer 🍺, lamborghini 🏎️, or private island 🏝️
💖 Please consider recurring donations to fully support continued research

Your support enables cutting-edge AI research for everyone! 🚀

Ridge Regression Algorithm for Few-Shot Learning
===============================================

Closed-form ridge regression implementation with Woodbury formula optimization.
Provides fast, analytically exact solutions for linear few-shot learning tasks.
"""

# TODO: PHASE 1.2 - RIDGE REGRESSION ALGORITHM IMPLEMENTATION
# TODO: Create RidgeRegression class extending nn.Module
# TODO: - Implement __init__ with regularization parameter (lambda)
# TODO: - Add use_woodbury flag for automatic efficiency selection
# TODO: - Support both classification and regression targets  
# TODO: - Include bias term option for improved fitting
# TODO: - Add numerical stability checks for ill-conditioned matrices

# TODO: Implement ridge_regression() function for closed-form solutions
# TODO: - Use formula: W* = (X^T X + λI)^-1 X^T Y for standard form
# TODO: - Use Woodbury formula when n_samples < n_features for efficiency
# TODO: - Support both LongTensor (classification) and FloatTensor (regression) targets
# TODO: - Add automatic regularization parameter selection based on data
# TODO: - Include confidence intervals for predictions when possible

# TODO: Implement efficient matrix operations
# TODO: - Add woodbury_solver() for efficient matrix inversion
# TODO: - Implement safe_solve() with singular value decomposition fallback
# TODO: - Add condition number monitoring for numerical stability warnings
# TODO: - Optimize memory usage for large embedding matrices
# TODO: - Include batch processing for multiple tasks simultaneously

# TODO: Add integration with existing meta-learning framework
# TODO: - Integrate with Episode data structure for few-shot tasks
# TODO: - Add to algorithm selector as fast closed-form option
# TODO: - Include in A/B testing framework for performance comparison
# TODO: - Connect with failure prediction (low failure risk for stable method)
# TODO: - Add to performance monitoring dashboard with timing metrics

# TODO: Implement advanced features
# TODO: - Add support for different regularization types (L1, elastic net)
# TODO: - Implement cross-validation for optimal lambda selection
# TODO: - Add feature preprocessing and normalization options
# TODO: - Support for multi-output regression tasks
# TODO: - Include uncertainty quantification for predictions

# TODO: Add comprehensive testing and validation
# TODO: - Test mathematical correctness against sklearn Ridge
# TODO: - Validate Woodbury formula efficiency gains on large problems
# TODO: - Test numerical stability with ill-conditioned data
# TODO: - Benchmark performance against MAML and prototypical networks
# TODO: - Add regression tests for various data distributions

from __future__ import annotations
from typing import Optional, Union, Tuple, Dict, Any
import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings
import numpy as np
from ..shared.types import Episode


class RidgeRegression(nn.Module):
    """
    Ridge regression implementation for few-shot learning.
    
    Provides fast, analytically exact solutions using closed-form ridge regression
    with automatic Woodbury formula optimization for efficiency.
    """
    
    def __init__(
        self, 
        reg_lambda: float = 0.01,
        use_woodbury: Optional[bool] = None,
        bias: bool = True,
        scale: bool = True
    ):
        """
        Initialize Ridge Regression classifier.
        
        Args:
            reg_lambda: Regularization parameter (lambda)
            use_woodbury: Whether to use Woodbury formula (auto-select if None)
            bias: Whether to add bias term
            scale: Whether to scale features
        """
        super().__init__()
        self.reg_lambda = reg_lambda
        self.use_woodbury = use_woodbury
        self.bias = bias
        self.scale = scale
        
        # Learnable parameters
        self.weights = None
        self.bias_term = None
        self.feature_mean = None
        self.feature_std = None
    
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
    
    def fit(self, embeddings: torch.Tensor, targets: torch.Tensor) -> Dict[str, Any]:
        """
        Fit ridge regression using closed-form solution.
        
        Args:
            embeddings: Input embeddings [n_samples, n_features]
            targets: Target values [n_samples, n_targets]
            
        Returns:
            Dict containing fitting statistics
        """
        X = embeddings
        Y = targets
        
        n_samples, n_features = X.shape
        n_targets = Y.shape[1] if Y.dim() > 1 else 1
        
        # Feature scaling
        if self.scale:
            self.feature_mean = X.mean(dim=0, keepdim=True)
            self.feature_std = X.std(dim=0, keepdim=True) + 1e-8
            X = (X - self.feature_mean) / self.feature_std
        
        # Add bias term
        if self.bias:
            X = torch.cat([X, torch.ones(n_samples, 1, device=X.device)], dim=1)
            n_features += 1
        
        # Automatic Woodbury selection
        use_woodbury = self.use_woodbury
        if use_woodbury is None:
            use_woodbury = n_samples < n_features
        
        # Solve ridge regression
        if use_woodbury:
            weights, condition_num = woodbury_solver(X, Y, self.reg_lambda)
        else:
            weights, condition_num = standard_solver(X, Y, self.reg_lambda)
        
        # Store weights
        if self.bias:
            self.weights = weights[:-1]
            self.bias_term = weights[-1:]
        else:
            self.weights = weights
            self.bias_term = None
        
        return {
            'condition_number': condition_num,
            'method': 'woodbury' if use_woodbury else 'standard',
            'n_samples': n_samples,
            'n_features': n_features,
            'reg_lambda': self.reg_lambda
        }
    
    def predict(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Predict using fitted ridge regression model.
        
        Args:
            embeddings: Query embeddings [n_query, n_features]
            
        Returns:
            Predictions [n_query, n_targets]
        """
        if self.weights is None:
            raise RuntimeError("Model must be fitted before prediction")
        
        X = embeddings
        
        # Apply feature scaling
        if self.scale and self.feature_mean is not None:
            X = (X - self.feature_mean) / self.feature_std
        
        # Compute predictions
        predictions = torch.matmul(X, self.weights)
        
        # Add bias term
        if self.bias and self.bias_term is not None:
            predictions = predictions + self.bias_term
        
        return predictions


def ridge_regression(
    embeddings: torch.Tensor,
    targets: torch.Tensor, 
    reg_lambda: Union[float, torch.Tensor],
    num_classes: Optional[int] = None,
    use_woodbury: Optional[bool] = None,
    scale: bool = True,
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
        scale: Whether to scale features
        bias: Whether to add bias term
        
    Returns:
        Tuple of (weights, bias_term)
    """
    X = embeddings
    Y = targets
    
    # Handle classification targets
    if Y.dtype == torch.long and num_classes is not None:
        Y = F.one_hot(Y, num_classes=num_classes).float()
    elif Y.dim() == 1:
        Y = Y.unsqueeze(1)
    
    n_samples, n_features = X.shape
    
    # Feature scaling
    if scale:
        X_mean = X.mean(dim=0, keepdim=True)
        X_std = X.std(dim=0, keepdim=True) + 1e-8
        X = (X - X_mean) / X_std
    
    # Add bias term
    if bias:
        X = torch.cat([X, torch.ones(n_samples, 1, device=X.device)], dim=1)
        n_features += 1
    
    # Choose method
    if use_woodbury is None:
        use_woodbury = n_samples < n_features
    
    # Solve
    if use_woodbury:
        weights, condition_num = woodbury_solver(X, Y, reg_lambda)
    else:
        weights, condition_num = standard_solver(X, Y, reg_lambda)
    
    # Split weights and bias
    if bias:
        return weights[:-1], weights[-1:]
    else:
        return weights, None


def standard_solver(X: torch.Tensor, Y: torch.Tensor, reg_lambda: float) -> Tuple[torch.Tensor, float]:
    """
    Standard ridge regression: W* = (X^T X + λI)^-1 X^T Y
    
    Args:
        X: Input matrix [n_samples, n_features]
        Y: Target matrix [n_samples, n_targets]
        reg_lambda: Regularization parameter
        
    Returns:
        Tuple of (weights, condition_number)
    """
    n_features = X.shape[1]
    
    # Compute X^T X
    XtX = torch.matmul(X.t(), X)
    
    # Add regularization: X^T X + λI
    regularized = XtX + reg_lambda * torch.eye(n_features, device=X.device)
    
    # Compute condition number for numerical stability monitoring
    try:
        condition_num = torch.linalg.cond(regularized).item()
        if condition_num > 1e12:
            warnings.warn(f"High condition number {condition_num:.2e}, solution may be unstable")
    except:
        condition_num = float('inf')
    
    # Compute X^T Y
    XtY = torch.matmul(X.t(), Y)
    
    # Solve: W* = (X^T X + λI)^-1 X^T Y
    try:
        weights = torch.linalg.solve(regularized, XtY)
    except RuntimeError:
        # Fallback to pseudo-inverse for singular matrices
        warnings.warn("Matrix is singular, using pseudo-inverse")
        weights = torch.linalg.pinv(regularized) @ XtY
        condition_num = float('inf')
    
    return weights, condition_num


def woodbury_solver(X: torch.Tensor, Y: torch.Tensor, reg_lambda: float) -> Tuple[torch.Tensor, float]:
    """
    Efficient ridge regression using Woodbury matrix identity.
    
    Woodbury formula: (X^T X + λI)^-1 = (1/λ)I - (1/λ)X^T (XX^T + λI)^-1 X
    More efficient when n_samples < n_features.
    
    Args:
        X: Input matrix [n_samples, n_features]
        Y: Target matrix [n_samples, n_targets]
        reg_lambda: Regularization parameter
        
    Returns:
        Tuple of (weights, condition_number)
    """
    n_samples, n_features = X.shape
    
    # Compute XX^T
    XXt = torch.matmul(X, X.t())
    
    # Add regularization: XX^T + λI
    regularized = XXt + reg_lambda * torch.eye(n_samples, device=X.device)
    
    # Compute condition number
    try:
        condition_num = torch.linalg.cond(regularized).item()
        if condition_num > 1e12:
            warnings.warn(f"High condition number {condition_num:.2e}, solution may be unstable")
    except:
        condition_num = float('inf')
    
    # Solve (XX^T + λI)^-1 Y
    try:
        inv_term = torch.linalg.solve(regularized, Y)
    except RuntimeError:
        warnings.warn("Matrix is singular, using pseudo-inverse")
        inv_term = torch.linalg.pinv(regularized) @ Y
        condition_num = float('inf')
    
    # Woodbury formula: W* = (1/λ) X^T (Y - X (1/λ) X^T (XX^T + λI)^-1 Y)
    # Simplified: W* = (1/λ) X^T Y - (1/λ²) X^T X (XX^T + λI)^-1 Y
    XtY = torch.matmul(X.t(), Y)
    XtX_inv_term = torch.matmul(X.t(), inv_term)
    
    weights = (1.0 / reg_lambda) * XtY - (1.0 / (reg_lambda * reg_lambda)) * XtX_inv_term
    
    return weights, condition_num


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