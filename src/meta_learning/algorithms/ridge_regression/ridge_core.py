"""
Ridge Regression Core Implementation
===================================

Main RidgeRegression class for few-shot learning scenarios.
Provides fast closed-form solutions with automatic Woodbury optimization.
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings
from typing import Optional, Union, Dict, Any, Tuple
from .solvers import woodbury_solver, standard_ridge_solve
from .utils import preprocess_features, select_regularization, uncertainty_estimation


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
        preprocessing: str = 'standardize',
        lambda_selection: str = 'fixed'
    ):
        """
        Initialize Ridge Regression classifier.
        
        Args:
            reg_lambda: Regularization parameter (lambda)
            use_woodbury: Whether to use Woodbury formula (auto-select if None)
            bias: Whether to add bias term
            preprocessing: Feature preprocessing ('standardize', 'normalize', 'none')
            lambda_selection: Lambda selection method ('fixed', 'cv', 'gcv')
        """
        super().__init__()
        self.reg_lambda = reg_lambda
        self.use_woodbury = use_woodbury
        self.bias = bias
        self.preprocessing = preprocessing
        self.lambda_selection = lambda_selection
        
        # Fitted parameters
        self.weights = None
        self.bias_term = None
        self.preprocessing_params = {}
        self.fitted_lambda = None
        
    def fit(self, X: torch.Tensor, Y: torch.Tensor) -> Dict[str, Any]:
        """
        Fit ridge regression using closed-form solution.
        
        Args:
            X: Input features [n_samples, n_features]
            Y: Target values [n_samples, n_targets]
            
        Returns:
            Dict containing fitting statistics
        """
        n_samples, n_features = X.shape
        
        # Preprocess features
        X_processed, self.preprocessing_params = preprocess_features(
            X, method=self.preprocessing
        )
        
        # Handle target dimensions and types
        if Y.dtype == torch.long:
            # Convert classification labels to one-hot
            num_classes = len(torch.unique(Y))
            Y = F.one_hot(Y, num_classes=num_classes).float()
        elif Y.dim() == 1:
            Y = Y.unsqueeze(1).float()
        else:
            Y = Y.float()  # Ensure float type
        
        n_targets = Y.shape[1]
        
        # Select regularization parameter
        if self.lambda_selection != 'fixed':
            self.fitted_lambda = select_regularization(
                X_processed, Y, method=self.lambda_selection
            )
        else:
            self.fitted_lambda = self.reg_lambda
        
        # Determine solver method
        use_woodbury = self.use_woodbury
        if use_woodbury is None:
            use_woodbury = n_samples < n_features
        
        # Solve ridge regression
        if use_woodbury:
            weights = woodbury_solver(X_processed, Y, self.fitted_lambda)
            method = 'woodbury'
        else:
            weights = standard_ridge_solve(X_processed, Y, self.fitted_lambda)
            method = 'standard'
        
        # Handle bias term
        if self.bias:
            # Add bias column to features for unified weight matrix
            X_with_bias = torch.cat([
                X_processed, 
                torch.ones(n_samples, 1, device=X.device)
            ], dim=1)
            
            # Re-solve with bias
            if use_woodbury:
                weights_with_bias = woodbury_solver(X_with_bias, Y, self.fitted_lambda)
            else:
                weights_with_bias = standard_ridge_solve(X_with_bias, Y, self.fitted_lambda)
            
            self.weights = weights_with_bias[:-1]
            self.bias_term = weights_with_bias[-1:]
        else:
            self.weights = weights
            self.bias_term = None
        
        return {
            'method': method,
            'n_samples': n_samples,
            'n_features': n_features,
            'n_targets': n_targets,
            'reg_lambda': self.fitted_lambda,
            'preprocessing': self.preprocessing,
            'use_bias': self.bias
        }
    
    def predict(self, X: torch.Tensor, return_uncertainty: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        """
        Predict using fitted ridge regression model.
        
        Args:
            X: Query features [n_query, n_features]
            return_uncertainty: Whether to return uncertainty estimates
            
        Returns:
            Predictions [n_query, n_targets] or (predictions, uncertainties)
        """
        if self.weights is None:
            raise RuntimeError("Model must be fitted before prediction")
        
        # Apply same preprocessing as training
        X_processed, _ = preprocess_features(
            X, method=self.preprocessing, fit_params=self.preprocessing_params
        )
        
        # Compute predictions
        predictions = torch.mm(X_processed, self.weights)
        
        # Add bias term
        if self.bias and self.bias_term is not None:
            predictions = predictions + self.bias_term
        
        if return_uncertainty:
            uncertainty_dict = uncertainty_estimation(
                X_processed, self.weights, reg_lambda=self.fitted_lambda
            )
            return predictions, uncertainty_dict
        
        return predictions
    
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for nn.Module compatibility.
        
        Args:
            X: Input features [n_samples, n_features]
            
        Returns:
            Predictions [n_samples, n_targets]
        """
        return self.predict(X)


def ridge_regression_solve(
    X: torch.Tensor,
    Y: torch.Tensor, 
    reg_lambda: float = 0.01,
    preprocessing: str = 'standardize',
    use_woodbury: Optional[bool] = None,
    bias: bool = True
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """
    Standalone ridge regression solver function.
    
    Args:
        X: Input features [n_samples, n_features]
        Y: Target values [n_samples] or [n_samples, n_targets]
        reg_lambda: Regularization parameter
        preprocessing: Feature preprocessing method
        use_woodbury: Whether to use Woodbury formula (auto if None)
        bias: Whether to add bias term
        
    Returns:
        Tuple of (predictions, metadata)
    """
    # Create and fit model
    model = RidgeRegression(
        reg_lambda=reg_lambda,
        use_woodbury=use_woodbury,
        bias=bias,
        preprocessing=preprocessing
    )
    
    fit_info = model.fit(X, Y)
    
    # Make predictions
    predictions = model.predict(X)
    
    metadata = {
        **fit_info,
        'weights': model.weights,
        'bias_term': model.bias_term,
        'preprocessing_params': model.preprocessing_params
    }
    
    return predictions, metadata


def create_ridge_classifier(
    reg_lambda: float = 0.01,
    preprocessing: str = 'standardize',
    **kwargs
) -> RidgeRegression:
    """
    Create a ridge regression classifier with sensible defaults.
    
    Args:
        reg_lambda: Regularization parameter
        preprocessing: Feature preprocessing method
        **kwargs: Additional arguments for RidgeRegression
        
    Returns:
        Configured RidgeRegression instance
    """
    return RidgeRegression(
        reg_lambda=reg_lambda,
        preprocessing=preprocessing,
        bias=True,
        lambda_selection='cv',
        **kwargs
    )


def batch_ridge_regression(
    X_batch: torch.Tensor,
    Y_batch: torch.Tensor,
    reg_lambda: float = 0.01,
    **kwargs
) -> torch.Tensor:
    """
    Apply ridge regression to a batch of problems.
    
    Args:
        X_batch: Batch of features [batch_size, n_samples, n_features]
        Y_batch: Batch of targets [batch_size, n_samples, n_targets]
        reg_lambda: Regularization parameter
        **kwargs: Additional arguments for RidgeRegression
        
    Returns:
        Batch of weight matrices [batch_size, n_features, n_targets]
    """
    batch_size = X_batch.shape[0]
    weights_list = []
    
    for i in range(batch_size):
        model = RidgeRegression(reg_lambda=reg_lambda, **kwargs)
        model.fit(X_batch[i], Y_batch[i])
        weights_list.append(model.weights)
    
    return torch.stack(weights_list)