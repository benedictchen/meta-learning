"""
Composite Transform
==================

Composition of multiple gradient transforms for meta-learning optimization.
Extracted from large transforms.py for better maintainability.
"""

from __future__ import annotations
from typing import Dict, List
import torch
import torch.nn as nn

from .base_transform import GradientTransform


class CompositeTransform(GradientTransform):
    """
    Composition of multiple gradient transforms.
    
    Applies multiple transforms in sequence to create complex
    gradient transformations for meta-learning.
    """
    
    def __init__(
        self,
        transforms: List[GradientTransform],
        regularization_weight: float = 0.0
    ):
        """Initialize composite transform.
        
        Args:
            transforms: List of transforms to apply in sequence
            regularization_weight: Additional regularization weight
        """
        super().__init__(regularization_weight)
        
        self.transforms = nn.ModuleList(transforms)
        self.num_transforms = len(transforms)
    
    def forward(self, gradient: torch.Tensor, parameter: torch.Tensor = None) -> torch.Tensor:
        """
        Apply all transforms in sequence.
        
        Args:
            gradient: Input gradient tensor
            parameter: Associated parameter tensor
            
        Returns:
            Transformed gradient tensor
        """
        self._update_statistics(gradient)
        
        current_gradient = gradient
        
        # Apply each transform in sequence
        for transform in self.transforms:
            current_gradient = transform(current_gradient, parameter)
        
        return current_gradient
    
    def compute_regularization_loss(self) -> torch.Tensor:
        """Compute total regularization loss from all transforms."""
        total_loss = super().compute_regularization_loss()
        
        # Add regularization from each transform
        for transform in self.transforms:
            total_loss += transform.compute_regularization_loss()
        
        return total_loss
    
    def get_transform_statistics(self) -> Dict[str, Dict[str, float]]:
        """Get statistics from all component transforms."""
        stats = {}
        for i, transform in enumerate(self.transforms):
            transform_name = f"{transform.__class__.__name__}_{i}"
            stats[transform_name] = transform.get_statistics()
        return stats
    
    def add_transform(self, transform: GradientTransform):
        """Add a new transform to the composition."""
        self.transforms.append(transform)
        self.num_transforms += 1
    
    def remove_transform(self, index: int):
        """Remove a transform by index."""
        if 0 <= index < len(self.transforms):
            del self.transforms[index]
            self.num_transforms -= 1