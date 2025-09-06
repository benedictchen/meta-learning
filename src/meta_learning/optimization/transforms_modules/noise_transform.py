"""
Noise Transform
===============

Gradient noise injection transform for regularization.
Extracted from large transforms.py for better maintainability.
"""

from .base_transform import GradientTransform
import torch


class NoiseTransform(GradientTransform):
    """Gradient noise injection for regularization and exploration."""
    
    def __init__(self, noise_scale: float = 0.01, regularization_weight: float = 0.0):
        super().__init__(regularization_weight)
        self.noise_scale = torch.nn.Parameter(torch.tensor(noise_scale))
    
    def forward(self, gradient: torch.Tensor, parameter: torch.Tensor = None) -> torch.Tensor:
        """Add learnable noise to gradient."""
        self._update_statistics(gradient)
        
        if self.training:
            noise = torch.randn_like(gradient) * self.noise_scale
            return gradient + noise
        else:
            return gradient