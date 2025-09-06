"""
Base Gradient Transform
======================

Abstract base class for all gradient transforms in learnable optimization.
Extracted from large transforms.py for better maintainability.
"""

from __future__ import annotations
from typing import Dict
import torch
import torch.nn as nn
from abc import ABC, abstractmethod


class GradientTransform(nn.Module, ABC):
    """Abstract base class for gradient transforms."""
    
    def __init__(self, regularization_weight: float = 0.0):
        """Initialize base gradient transform.
        
        Args:
            regularization_weight: L2 regularization weight for transform parameters
        """
        super().__init__()
        self.regularization_weight = regularization_weight
        
        # Common parameters for all transforms
        self.num_applications = 0
        self.gradient_norm_history = []
        self.register_buffer('total_applications', torch.tensor(0, dtype=torch.long))
    
    @abstractmethod
    def forward(self, gradient: torch.Tensor, parameter: torch.Tensor = None) -> torch.Tensor:
        """
        Apply gradient transform.
        
        Args:
            gradient: Input gradient tensor
            parameter: Associated parameter tensor (for context)
            
        Returns:
            Transformed gradient tensor
        """
        pass
    
    def compute_regularization_loss(self) -> torch.Tensor:
        """
        Compute L2 regularization loss for transform parameters.
        
        Returns:
            Regularization loss scalar tensor
        """
        if self.regularization_weight == 0.0:
            return torch.tensor(0.0)
        
        # Get device from first parameter if available
        device = None
        param_list = list(self.parameters())
        if param_list:
            device = param_list[0].device
        
        # L2 regularization on all learnable parameters
        reg_loss = torch.tensor(0.0, device=device)
        
        for param in param_list:
            if param.requires_grad:
                reg_loss += torch.sum(param ** 2)
        
        return self.regularization_weight * reg_loss
    
    def _update_statistics(self, gradient: torch.Tensor) -> None:
        """Update internal statistics for monitoring."""
        with torch.no_grad():
            self.num_applications += 1
            self.total_applications += 1
            
            # Track gradient norm for analysis
            grad_norm = gradient.norm().item()
            self.gradient_norm_history.append(grad_norm)
            
            # Keep history bounded
            if len(self.gradient_norm_history) > 1000:
                self.gradient_norm_history = self.gradient_norm_history[-500:]
    
    def get_statistics(self) -> Dict[str, float]:
        """Get transform usage and performance statistics."""
        if not self.gradient_norm_history:
            return {'applications': self.num_applications, 'avg_gradient_norm': 0.0}
        
        import numpy as np
        return {
            'applications': self.num_applications,
            'avg_gradient_norm': np.mean(self.gradient_norm_history),
            'gradient_norm_std': np.std(self.gradient_norm_history),
            'recent_gradient_norm': self.gradient_norm_history[-1] if self.gradient_norm_history else 0.0
        }
    
    def reset_statistics(self) -> None:
        """Reset all tracked statistics."""
        self.num_applications = 0
        self.gradient_norm_history.clear()
        self.total_applications.zero_()
    
    def get_parameter_info(self) -> Dict[str, int]:
        """Get information about transform parameters."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'frozen_parameters': total_params - trainable_params
        }