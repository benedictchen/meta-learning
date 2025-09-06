"""
Momentum Transform
=================

Learnable momentum-based transform for gradients.
Extracted from large transforms.py for better maintainability.
"""

from __future__ import annotations
from typing import Dict
import torch
import torch.nn as nn

from .base_transform import GradientTransform


class MomentumTransform(GradientTransform):
    """Learnable momentum-based gradient transform."""
    
    def __init__(self, momentum: float = 0.9, regularization_weight: float = 0.01):
        super().__init__(regularization_weight)
        self.momentum = nn.Parameter(torch.tensor(momentum))
        self.velocity_buffers = {}
    
    def forward(self, gradient: torch.Tensor, parameter: torch.Tensor = None) -> torch.Tensor:
        """Apply momentum transform to gradient."""
        self._update_statistics(gradient)
        
        param_id = id(parameter) if parameter is not None else 0
        
        if param_id not in self.velocity_buffers:
            self.velocity_buffers[param_id] = torch.zeros_like(gradient)
        
        velocity = self.velocity_buffers[param_id]
        
        with torch.no_grad():
            velocity.mul_(self.momentum).add_(gradient)
            self.velocity_buffers[param_id] = velocity
        
        return velocity
    
    def reset_buffers(self):
        """Reset all velocity buffers."""
        self.velocity_buffers.clear()