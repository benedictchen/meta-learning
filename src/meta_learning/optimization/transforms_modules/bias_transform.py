"""
Bias Transform
=============

Learnable bias addition transform for gradients in meta-learning optimization.
Extracted from large transforms.py for better maintainability.
"""

from __future__ import annotations
from typing import Dict
import torch
import torch.nn as nn

from .base_transform import GradientTransform


class BiasTransform(GradientTransform):
    """
    Learnable bias addition transform for gradients.
    
    Adds learnable bias terms to gradients, potentially with momentum.
    """
    
    def __init__(
        self,
        parameter_shapes: Dict[str, torch.Size],
        init_bias: float = 0.0,
        momentum: float = 0.9,
        regularization_weight: float = 0.01
    ):
        """Initialize learnable bias transform.
        
        Args:
            parameter_shapes: Dictionary mapping parameter names to shapes
            init_bias: Initial bias value
            momentum: Momentum factor for bias accumulation
            regularization_weight: L2 regularization for bias parameters
        """
        super().__init__(regularization_weight)
        
        self.parameter_shapes = parameter_shapes
        self.init_bias = init_bias
        self.momentum = momentum
        
        # Initialize learnable bias parameters
        self.biases = nn.ParameterDict()
        for name, shape in parameter_shapes.items():
            bias_param = torch.full(shape, init_bias, dtype=torch.float32)
            self.biases[name] = nn.Parameter(bias_param)
        
        # Initialize momentum buffers (non-learnable)
        self.momentum_buffers = {}
        for name, shape in parameter_shapes.items():
            self.register_buffer(f'momentum_{name}', torch.zeros(shape))
            self.momentum_buffers[name] = getattr(self, f'momentum_{name}')
    
    def forward(self, gradient: torch.Tensor, parameter: torch.Tensor = None) -> torch.Tensor:
        """
        Apply learnable bias addition to gradient.
        
        Args:
            gradient: Input gradient tensor
            parameter: Associated parameter tensor
            
        Returns:
            Gradient with bias added
        """
        self._update_statistics(gradient)
        
        if parameter is None:
            # No parameter context, apply zero bias
            return gradient
        
        # Find matching bias parameter by shape
        param_shape = tuple(parameter.shape)
        matching_bias = None
        matching_momentum = None
        
        for name, bias in self.biases.items():
            if tuple(bias.shape) == param_shape:
                matching_bias = bias
                matching_momentum = self.momentum_buffers[name]
                break
        
        if matching_bias is not None and matching_momentum is not None:
            # Apply momentum to bias
            with torch.no_grad():
                matching_momentum.mul_(self.momentum).add_(matching_bias, alpha=1-self.momentum)
            
            # Add bias to gradient
            biased_gradient = gradient + matching_momentum
        else:
            # Fallback: no bias addition if no matching shape
            biased_gradient = gradient
        
        return biased_gradient
    
    def get_current_biases(self) -> Dict[str, torch.Tensor]:
        """Get current bias values for analysis."""
        return {name: bias.clone() for name, bias in self.biases.items()}
    
    def get_momentum_buffers(self) -> Dict[str, torch.Tensor]:
        """Get current momentum buffer values."""
        return {name: buffer.clone() for name, buffer in self.momentum_buffers.items()}
    
    def reset_momentum_buffers(self):
        """Reset all momentum buffers to zero."""
        with torch.no_grad():
            for buffer in self.momentum_buffers.values():
                buffer.zero_()
    
    def set_momentum(self, momentum: float):
        """Update momentum factor."""
        self.momentum = max(0.0, min(1.0, momentum))  # Clamp to [0, 1]