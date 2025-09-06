"""
TTCS Augmentation Transforms
===========================

Test-Time Augmentation transforms for improved robustness.
Extracted from large ttcs.py for better maintainability.

FIRST PUBLIC IMPLEMENTATION of Test-Time Compute Scaling!
"""

from __future__ import annotations
import torch
import torch.nn as nn
from torchvision import transforms
from typing import Optional, List, Tuple, Dict, Any
import numpy as np


def create_tta_transforms(image_size: int = 32, augmentation_strength: float = 1.0) -> transforms.Compose:
    """
    Create test-time augmentation transforms with configurable strength.
    
    Args:
        image_size: Size of input images
        augmentation_strength: Strength of augmentation (0.0 to 1.0)
        
    Returns:
        Composed transforms for test-time augmentation
    """
    strength = np.clip(augmentation_strength, 0.0, 1.0)
    
    return transforms.Compose([
        transforms.RandomResizedCrop(
            image_size, 
            scale=(1.0 - 0.1 * strength, 1.0)
        ),
        transforms.RandomHorizontalFlip(p=0.5 * strength),
        transforms.RandomRotation(
            degrees=5 * strength,
            interpolation=transforms.InterpolationMode.BILINEAR
        ),
        transforms.ColorJitter(
            brightness=0.1 * strength,
            contrast=0.1 * strength,
            saturation=0.1 * strength,
            hue=0.05 * strength
        )
    ])


def apply_tta_to_batch(batch: torch.Tensor, transforms: transforms.Compose, 
                      device: Optional[torch.device] = None) -> torch.Tensor:
    """
    Apply test-time augmentation transforms to a batch of images.
    
    Args:
        batch: Input batch tensor [N, C, H, W]
        transforms: TTA transforms to apply
        device: Device to place augmented batch on
        
    Returns:
        Augmented batch tensor
    """
    if device is None:
        device = batch.device
    
    augmented_images = []
    
    for img in batch:
        # Apply transforms (works on CPU, then move to device)
        if img.dim() == 3:  # Single image [C, H, W]
            aug_img = transforms(img.cpu())
            augmented_images.append(aug_img)
        else:  # Fallback for non-image data
            augmented_images.append(img.cpu())
    
    return torch.stack(augmented_images).to(device)


def create_adaptive_tta_pipeline(image_size: int = 32, num_augmentations: int = 4,
                               diversity_factor: float = 1.0) -> List[transforms.Compose]:
    """
    Create diverse TTA pipeline with multiple augmentation variants.
    
    Args:
        image_size: Size of input images
        num_augmentations: Number of different augmentation variants
        diversity_factor: Factor controlling diversity between variants
        
    Returns:
        List of diverse augmentation transforms
    """
    pipelines = []
    
    for i in range(num_augmentations):
        # Create varied augmentation strengths
        strength = 0.3 + (i / max(1, num_augmentations - 1)) * 0.7 * diversity_factor
        
        # Vary which augmentations are emphasized
        if i % 4 == 0:  # Geometric emphasis
            pipeline = transforms.Compose([
                transforms.RandomResizedCrop(image_size, scale=(0.85, 1.0)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=10 * strength)
            ])
        elif i % 4 == 1:  # Color emphasis
            pipeline = transforms.Compose([
                transforms.RandomResizedCrop(image_size, scale=(0.9, 1.0)),
                transforms.ColorJitter(
                    brightness=0.2 * strength,
                    contrast=0.2 * strength,
                    saturation=0.2 * strength
                )
            ])
        elif i % 4 == 2:  # Conservative augmentation
            pipeline = transforms.Compose([
                transforms.RandomResizedCrop(image_size, scale=(0.95, 1.0)),
                transforms.RandomHorizontalFlip(p=0.3)
            ])
        else:  # Aggressive augmentation
            pipeline = transforms.Compose([
                transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
                transforms.RandomHorizontalFlip(p=0.7),
                transforms.RandomRotation(degrees=15 * strength),
                transforms.ColorJitter(brightness=0.15, contrast=0.15)
            ])
        
        pipelines.append(pipeline)
    
    return pipelines


def estimate_tta_diversity(augmented_features: List[torch.Tensor]) -> Dict[str, float]:
    """
    Estimate diversity of test-time augmented features.
    
    Args:
        augmented_features: List of feature tensors from different augmentations
        
    Returns:
        Dictionary containing diversity metrics
    """
    if len(augmented_features) < 2:
        return {'diversity': 0.0, 'variance': 0.0, 'disagreement': 0.0}
    
    # Stack features [num_augmentations, batch_size, feature_dim]
    stacked_features = torch.stack(augmented_features, dim=0)
    
    # Compute pairwise cosine similarities
    similarities = []
    num_augmentations = len(augmented_features)
    
    for i in range(num_augmentations):
        for j in range(i + 1, num_augmentations):
            feat_i = stacked_features[i].flatten(start_dim=1)  # [batch_size, flattened_dim]
            feat_j = stacked_features[j].flatten(start_dim=1)
            
            # Cosine similarity
            cos_sim = torch.nn.functional.cosine_similarity(feat_i, feat_j, dim=1)
            similarities.append(cos_sim.mean().item())
    
    # Compute diversity metrics
    avg_similarity = np.mean(similarities)
    diversity = 1.0 - avg_similarity  # Higher diversity = lower similarity
    
    # Compute feature variance across augmentations
    feature_variance = stacked_features.var(dim=0).mean().item()
    
    # Compute prediction disagreement (if applicable)
    pred_variance = 0.0
    if stacked_features.size(-1) <= 1000:  # Only for reasonable feature sizes
        # Compute L2 distances between feature vectors
        distances = []
        for i in range(num_augmentations):
            for j in range(i + 1, num_augmentations):
                dist = torch.nn.functional.pairwise_distance(
                    stacked_features[i].flatten(start_dim=1),
                    stacked_features[j].flatten(start_dim=1),
                    p=2
                ).mean().item()
                distances.append(dist)
        pred_variance = np.mean(distances)
    
    return {
        'diversity': float(diversity),
        'variance': float(feature_variance),
        'disagreement': float(pred_variance),
        'avg_similarity': float(avg_similarity),
        'num_pairs': len(similarities)
    }


class TTAScheduler:
    """
    Adaptive scheduler for test-time augmentation intensity.
    
    Adjusts augmentation strength based on prediction uncertainty
    and computational budget.
    """
    
    def __init__(self, initial_strength: float = 0.5, 
                 adaptation_rate: float = 0.1,
                 min_strength: float = 0.1,
                 max_strength: float = 0.9):
        """
        Initialize TTA scheduler.
        
        Args:
            initial_strength: Starting augmentation strength
            adaptation_rate: Rate of strength adaptation
            min_strength: Minimum augmentation strength
            max_strength: Maximum augmentation strength
        """
        self.current_strength = initial_strength
        self.adaptation_rate = adaptation_rate
        self.min_strength = min_strength
        self.max_strength = max_strength
        
        # History tracking
        self.uncertainty_history = []
        self.strength_history = []
    
    def update_strength(self, uncertainty: float, budget_remaining: float = 1.0) -> float:
        """
        Update augmentation strength based on uncertainty and budget.
        
        Args:
            uncertainty: Current prediction uncertainty (0 to 1)
            budget_remaining: Remaining computational budget (0 to 1)
            
        Returns:
            Updated augmentation strength
        """
        # Higher uncertainty -> higher augmentation strength
        target_strength = self.min_strength + uncertainty * (self.max_strength - self.min_strength)
        
        # Adjust based on computational budget
        target_strength *= budget_remaining
        
        # Smooth adaptation
        self.current_strength += self.adaptation_rate * (target_strength - self.current_strength)
        
        # Clamp to bounds
        self.current_strength = np.clip(self.current_strength, self.min_strength, self.max_strength)
        
        # Update history
        self.uncertainty_history.append(uncertainty)
        self.strength_history.append(self.current_strength)
        
        # Limit history size
        if len(self.uncertainty_history) > 1000:
            self.uncertainty_history = self.uncertainty_history[-500:]
            self.strength_history = self.strength_history[-500:]
        
        return self.current_strength
    
    def get_transforms(self, image_size: int = 32) -> transforms.Compose:
        """
        Get TTA transforms with current strength.
        
        Args:
            image_size: Size of input images
            
        Returns:
            TTA transforms with current strength
        """
        return create_tta_transforms(image_size, self.current_strength)
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get scheduler statistics.
        
        Returns:
            Dictionary of scheduler statistics
        """
        if not self.uncertainty_history:
            return {'current_strength': self.current_strength}
        
        return {
            'current_strength': self.current_strength,
            'avg_uncertainty': np.mean(self.uncertainty_history),
            'avg_strength': np.mean(self.strength_history),
            'strength_std': np.std(self.strength_history),
            'adaptation_efficiency': np.corrcoef(self.uncertainty_history, self.strength_history)[0, 1] if len(self.uncertainty_history) > 1 else 0.0
        }