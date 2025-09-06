"""
Adaptive Transform
==================

Adaptive parameter transforms for meta-learning optimization.
Extracted from large transforms.py for better maintainability.
"""

from .base_transform import GradientTransform
import torch


class AdaptiveTransform(GradientTransform):
    """Adaptive parameter transforms that learn from gradient history."""
    
    def __init__(self, adaptation_rate: float = 0.01, regularization_weight: float = 0.01):
        super().__init__(regularization_weight)
        self.adaptation_rate = torch.nn.Parameter(torch.tensor(adaptation_rate))
        self.running_mean = None
        self.running_var = None
    
    def forward(self, gradient: torch.Tensor, parameter: torch.Tensor = None) -> torch.Tensor:
        """Apply adaptive transform based on gradient statistics."""
        self._update_statistics(gradient)
        
        if self.running_mean is None:
            self.running_mean = torch.zeros_like(gradient)
            self.running_var = torch.ones_like(gradient)
        
        # Update running statistics
        with torch.no_grad():
            self.running_mean.lerp_(gradient, self.adaptation_rate)
            squared_diff = (gradient - self.running_mean).pow(2)
            self.running_var.lerp_(squared_diff, self.adaptation_rate)
        
        # Normalize gradient
        normalized_grad = (gradient - self.running_mean) / (torch.sqrt(self.running_var) + 1e-8)
        
        return normalized_grad