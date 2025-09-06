"""
Transform Optimizer
==================

Optimizer for managing and training gradient transform parameters.
Extracted from large transforms.py for better maintainability.
"""

from __future__ import annotations
from typing import List, Dict, Any
import torch
import torch.nn as nn
from torch.optim import Adam

from .base_transform import GradientTransform


class TransformOptimizer:
    """
    Optimizer for training gradient transform parameters.
    
    Manages the learning of transform parameters across meta-learning tasks.
    """
    
    def __init__(
        self,
        transforms: List[GradientTransform],
        learning_rate: float = 1e-3,
        regularization_weight: float = 0.01
    ):
        """Initialize transform optimizer.
        
        Args:
            transforms: List of gradient transforms to optimize
            learning_rate: Learning rate for transform parameters
            regularization_weight: Overall regularization weight
        """
        self.transforms = transforms
        self.learning_rate = learning_rate
        self.regularization_weight = regularization_weight
        
        # Collect all transform parameters
        all_params = []
        for transform in transforms:
            all_params.extend(list(transform.parameters()))
        
        # Create optimizer for transform parameters
        self.optimizer = Adam(all_params, lr=learning_rate)
        
        # Track optimization metrics
        self.step_count = 0
        self.loss_history = []
    
    def step(self, loss: torch.Tensor):
        """Perform optimization step for transform parameters."""
        # Add regularization from all transforms
        total_loss = loss
        for transform in self.transforms:
            total_loss += transform.compute_regularization_loss()
        
        # Backward pass and optimization step
        self.optimizer.zero_grad()
        total_loss.backward(retain_graph=True)
        self.optimizer.step()
        
        # Track metrics
        self.step_count += 1
        self.loss_history.append(total_loss.item())
        
        # Keep history bounded
        if len(self.loss_history) > 1000:
            self.loss_history = self.loss_history[-500:]
    
    def get_transform_parameters(self) -> Dict[str, torch.Tensor]:
        """Get current parameters from all transforms."""
        params = {}
        for i, transform in enumerate(self.transforms):
            transform_name = f"{transform.__class__.__name__}_{i}"
            params[transform_name] = {
                name: param.clone() for name, param in transform.named_parameters()
            }
        return params
    
    def get_optimization_statistics(self) -> Dict[str, Any]:
        """Get optimization statistics."""
        if not self.loss_history:
            return {'steps': self.step_count, 'avg_loss': 0.0}
        
        import numpy as np
        return {
            'steps': self.step_count,
            'avg_loss': np.mean(self.loss_history),
            'recent_loss': self.loss_history[-1] if self.loss_history else 0.0,
            'loss_std': np.std(self.loss_history)
        }
    
    def reset_optimizer(self):
        """Reset optimizer state."""
        self.optimizer.zero_grad()
        for group in self.optimizer.param_groups:
            group['step'] = 0