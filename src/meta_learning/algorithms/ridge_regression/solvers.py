"""
Matrix Solvers for Ridge Regression
===================================

Efficient matrix operations for ridge regression including:
- Woodbury formula implementation
- Safe matrix solving with fallbacks
- Numerical stability monitoring
"""

from __future__ import annotations
import torch
import warnings
import numpy as np
from typing import Tuple, Optional


def condition_number_check(matrix: torch.Tensor, threshold: float = 1e12) -> Tuple[float, bool]:
    """
    Check condition number of matrix for numerical stability.
    
    Args:
        matrix: Square matrix to check
        threshold: Condition number threshold for stability warning
        
    Returns:
        Tuple of (condition_number, is_stable)
    """
    try:
        # Use SVD for stable condition number computation
        U, S, V = torch.linalg.svd(matrix)
        if len(S) == 0 or S[-1] == 0:
            return float('inf'), False
            
        condition_num = (S[0] / S[-1]).item()
        is_stable = condition_num < threshold
        
        if not is_stable:
            warnings.warn(
                f"Matrix is ill-conditioned (condition number: {condition_num:.2e}). "
                f"Results may be numerically unstable."
            )
            
        return condition_num, is_stable
        
    except Exception as e:
        warnings.warn(f"Could not compute condition number: {e}")
        return float('inf'), False


def safe_solve(A: torch.Tensor, b: torch.Tensor, 
               reg_lambda: float = 1e-6) -> torch.Tensor:
    """
    Safely solve Ax = b with SVD fallback for numerical stability.
    
    Args:
        A: Coefficient matrix [n, n]
        b: Right-hand side [n, m]
        reg_lambda: Regularization for numerical stability
        
    Returns:
        Solution x [n, m]
    """
    try:
        # First try standard solve
        solution = torch.linalg.solve(A, b)
        
        # Check for NaN or inf
        if torch.isnan(solution).any() or torch.isinf(solution).any():
            raise RuntimeError("Standard solve produced NaN/inf values")
            
        return solution
        
    except RuntimeError:
        # Fallback to SVD-based pseudoinverse
        try:
            warnings.warn("Standard solve failed, using SVD pseudoinverse fallback")
            
            # Add regularization for stability
            if A.shape[0] == A.shape[1]:  # Square matrix
                A_reg = A + reg_lambda * torch.eye(A.shape[0], device=A.device)
            else:
                A_reg = A
                
            # Use pseudoinverse
            A_pinv = torch.linalg.pinv(A_reg)
            solution = A_pinv @ b
            
            if torch.isnan(solution).any() or torch.isinf(solution).any():
                # Last resort: return zero solution
                warnings.warn("SVD solve also failed, returning zero solution")
                return torch.zeros_like(b)
                
            return solution
            
        except Exception as e:
            warnings.warn(f"All solve methods failed: {e}, returning zero solution")
            return torch.zeros_like(b)


def woodbury_solver(X: torch.Tensor, Y: torch.Tensor, 
                   reg_lambda: float = 0.01) -> torch.Tensor:
    """
    Solve ridge regression using Woodbury matrix identity for efficiency.
    
    Uses Woodbury formula when n_samples < n_features:
    (X^T X + λI)^-1 X^T Y = X^T (XX^T + λI)^-1 Y / λ
    
    Args:
        X: Input features [n_samples, n_features]
        Y: Target values [n_samples, n_outputs] 
        reg_lambda: Regularization parameter
        
    Returns:
        Weights [n_features, n_outputs]
    """
    n_samples, n_features = X.shape
    
    if reg_lambda <= 0:
        reg_lambda = 1e-8  # Minimum regularization for stability
    
    if n_samples < n_features:
        # Use Woodbury formula: more efficient when n_samples < n_features
        # (X^T X + λI)^-1 X^T = X^T (XX^T + λI)^-1 / λ
        
        # Compute XX^T + λI
        XXT = torch.mm(X, X.t())  # [n_samples, n_samples]
        XXT_reg = XXT + reg_lambda * torch.eye(n_samples, device=X.device)
        
        # Solve (XX^T + λI)^-1 Y
        try:
            XXT_inv_Y = safe_solve(XXT_reg, Y)  # [n_samples, n_outputs]
            
            # Compute final weights: X^T (XX^T + λI)^-1 Y
            weights = torch.mm(X.t(), XXT_inv_Y)  # [n_features, n_outputs]
            
        except Exception as e:
            warnings.warn(f"Woodbury solver failed: {e}, using standard formulation")
            # Fallback to standard formulation
            weights = standard_ridge_solve(X, Y, reg_lambda)
            
    else:
        # Use standard formulation when n_samples >= n_features
        weights = standard_ridge_solve(X, Y, reg_lambda)
    
    return weights


def standard_ridge_solve(X: torch.Tensor, Y: torch.Tensor, 
                        reg_lambda: float = 0.01) -> torch.Tensor:
    """
    Solve ridge regression using standard formulation.
    
    Solves: W* = (X^T X + λI)^-1 X^T Y
    
    Args:
        X: Input features [n_samples, n_features]
        Y: Target values [n_samples, n_outputs]
        reg_lambda: Regularization parameter
        
    Returns:
        Weights [n_features, n_outputs]
    """
    n_features = X.shape[1]
    
    # Compute X^T X + λI
    XTX = torch.mm(X.t(), X)  # [n_features, n_features]
    XTX_reg = XTX + reg_lambda * torch.eye(n_features, device=X.device)
    
    # Check condition number
    cond_num, is_stable = condition_number_check(XTX_reg)
    
    # Compute X^T Y
    XTY = torch.mm(X.t(), Y)  # [n_features, n_outputs]
    
    # Solve for weights
    weights = safe_solve(XTX_reg, XTY, reg_lambda)
    
    return weights


def batch_ridge_solve(X_batch: torch.Tensor, Y_batch: torch.Tensor,
                     reg_lambda: float = 0.01) -> torch.Tensor:
    """
    Solve multiple ridge regression problems in batch.
    
    Args:
        X_batch: Batch of input features [batch_size, n_samples, n_features]
        Y_batch: Batch of targets [batch_size, n_samples, n_outputs]
        reg_lambda: Regularization parameter
        
    Returns:
        Batch of weights [batch_size, n_features, n_outputs]
    """
    batch_size = X_batch.shape[0]
    weights_batch = []
    
    for i in range(batch_size):
        X_i = X_batch[i]  # [n_samples, n_features]
        Y_i = Y_batch[i]  # [n_samples, n_outputs]
        
        # Choose solver based on problem size
        weights_i = woodbury_solver(X_i, Y_i, reg_lambda)
        weights_batch.append(weights_i)
    
    return torch.stack(weights_batch)  # [batch_size, n_features, n_outputs]


def adaptive_regularization(X: torch.Tensor, Y: torch.Tensor,
                           lambda_candidates: Optional[torch.Tensor] = None) -> float:
    """
    Select optimal regularization parameter using cross-validation.
    
    Args:
        X: Input features [n_samples, n_features]
        Y: Target values [n_samples, n_outputs]
        lambda_candidates: Candidate lambda values to try
        
    Returns:
        Optimal regularization parameter
    """
    if lambda_candidates is None:
        # Default candidates: logarithmic scale
        lambda_candidates = torch.logspace(-6, 2, 20, device=X.device)
    
    n_samples = X.shape[0]
    if n_samples < 5:
        # Too few samples for cross-validation, use default
        return 0.01
    
    # Simple validation split (80/20)
    perm = torch.randperm(n_samples, device=X.device)
    n_train = int(0.8 * n_samples)
    
    train_indices = perm[:n_train]
    val_indices = perm[n_train:]
    
    X_train, X_val = X[train_indices], X[val_indices]
    Y_train, Y_val = Y[train_indices], Y[val_indices]
    
    best_lambda = lambda_candidates[0].item()
    best_error = float('inf')
    
    for lam in lambda_candidates:
        try:
            # Train with this lambda
            weights = woodbury_solver(X_train, Y_train, lam.item())
            
            # Validate
            Y_pred = torch.mm(X_val, weights)
            error = torch.mean((Y_pred - Y_val) ** 2).item()
            
            if error < best_error:
                best_error = error
                best_lambda = lam.item()
                
        except Exception:
            # Skip this lambda if it causes numerical issues
            continue
    
    return best_lambda