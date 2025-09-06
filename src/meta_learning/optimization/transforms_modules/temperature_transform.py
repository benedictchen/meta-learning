"""
Temperature Transform
====================

Temperature-based gradient scaling transform.
Extracted from large transforms.py for better maintainability.
"""

from .base_transform import GradientTransform
import torch


class TemperatureTransform(GradientTransform):
    """Temperature-based gradient scaling for controlled optimization."""
    
    def __init__(self, initial_temperature: float = 1.0, regularization_weight: float = 0.01):
        super().__init__(regularization_weight)
        self.temperature = torch.nn.Parameter(torch.tensor(initial_temperature))
    
    def forward(self, gradient: torch.Tensor, parameter: torch.Tensor = None) -> torch.Tensor:
        """Scale gradient by learnable temperature."""
        self._update_statistics(gradient)
        
        # Prevent division by zero
        temp = torch.clamp(self.temperature, min=1e-6)
        return gradient / temp