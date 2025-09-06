"""
Scale Transform
==============

Learnable scaling transform for gradients in meta-learning optimization.
Extracted from large transforms.py for better maintainability.
"""

from __future__ import annotations
from typing import Dict
import torch
import torch.nn as nn

from .base_transform import GradientTransform


class ScaleTransform(GradientTransform):
    """
    Learnable scaling transform for gradients.
    
    Implements per-parameter or global learnable scaling factors that can adapt
    during meta-learning to improve optimization dynamics.
    """
    
    def __init__(
        self,
        parameter_shapes: Dict[str, torch.Size],
        init_scale: float = 1.0,
        per_parameter: bool = True,
        regularization_weight: float = 0.01
    ):
        """Initialize learnable scaling transform.
        
        Args:
            parameter_shapes: Dictionary mapping parameter names to shapes
            init_scale: Initial scaling value
            per_parameter: If True, learn separate scales per parameter
            regularization_weight: L2 regularization for scale parameters
        """
        super().__init__(regularization_weight)
        
        self.parameter_shapes = parameter_shapes
        self.init_scale = init_scale
        self.per_parameter = per_parameter
        
        # Initialize learnable scaling parameters
        if per_parameter:
            # Separate scaling parameter for each model parameter
            self.scales = nn.ParameterDict()
            for name, shape in parameter_shapes.items():
                # Initialize scales close to 1.0 for stability
                scale_param = torch.full(shape, init_scale, dtype=torch.float32)
                # Add small random noise to break symmetry
                scale_param += 0.1 * torch.randn_like(scale_param)
                self.scales[name] = nn.Parameter(scale_param)
        else:
            # Global scaling parameter
            self.global_scale = nn.Parameter(
                torch.tensor(init_scale, dtype=torch.float32)
            )
        
        # Constraints to prevent extreme scaling
        self.min_scale = 1e-6
        self.max_scale = 100.0
    
    def forward(self, gradient: torch.Tensor, parameter: torch.Tensor = None) -> torch.Tensor:
        """
        Apply learnable scaling to gradient.
        
        Args:
            gradient: Input gradient tensor
            parameter: Associated parameter tensor (for parameter identification)
            
        Returns:
            Scaled gradient tensor
        """
        self._update_statistics(gradient)
        
        if self.per_parameter and parameter is not None:
            # Find matching scale parameter by shape
            param_shape = tuple(parameter.shape)
            matching_scale = None
            
            # Find scale parameter with matching shape
            for name, scale in self.scales.items():
                if tuple(scale.shape) == param_shape:
                    matching_scale = scale
                    break
            
            if matching_scale is not None:
                # Clamp scaling to prevent extreme values
                clamped_scale = torch.clamp(matching_scale, self.min_scale, self.max_scale)
                scaled_gradient = gradient * clamped_scale
            else:
                # Fallback: use mean of all scales if no exact match
                avg_scale = torch.stack([s.mean() for s in self.scales.values()]).mean()
                clamped_scale = torch.clamp(avg_scale, self.min_scale, self.max_scale)
                scaled_gradient = gradient * clamped_scale
        else:
            # Apply global scaling
            if hasattr(self, 'global_scale'):
                clamped_scale = torch.clamp(self.global_scale, self.min_scale, self.max_scale)
            else:
                # Fallback to identity scaling
                clamped_scale = torch.tensor(1.0, device=gradient.device)
            
            scaled_gradient = gradient * clamped_scale
        
        return scaled_gradient
    
    def get_current_scales(self) -> Dict[str, torch.Tensor]:
        """Get current scaling values for analysis."""
        if self.per_parameter:
            return {name: torch.clamp(scale, self.min_scale, self.max_scale) 
                   for name, scale in self.scales.items()}
        else:
            return {'global': torch.clamp(self.global_scale, self.min_scale, self.max_scale)}
    
    def set_scale_bounds(self, min_scale: float, max_scale: float):
        """Update scaling bounds."""
        self.min_scale = max(min_scale, 1e-12)  # Prevent zero scaling
        self.max_scale = max_scale
    
    def get_scale_statistics(self) -> Dict[str, float]:
        """Get statistics about current scales."""
        current_scales = self.get_current_scales()
        
        if self.per_parameter:
            all_scales = torch.cat([scale.flatten() for scale in current_scales.values()])
            return {
                'mean_scale': all_scales.mean().item(),
                'std_scale': all_scales.std().item(), 
                'min_scale': all_scales.min().item(),
                'max_scale': all_scales.max().item(),
                'num_parameters': len(current_scales)
            }
        else:
            global_scale = current_scales['global'].item()
            return {
                'global_scale': global_scale,
                'num_parameters': 1
            }