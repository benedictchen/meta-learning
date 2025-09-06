"""
Core TTCS Prediction Functions
==============================

Core prediction functions for Test-Time Compute Scaling.
Extracted from large ttcs.py for better maintainability.

FIRST PUBLIC IMPLEMENTATION of Test-Time Compute Scaling for meta-learning!
"""

from __future__ import annotations
import hashlib
import time
from typing import Optional, Dict, Any, Tuple, List
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms


def tta_transforms(image_size: int = 32) -> transforms.Compose:
    """
    Create test-time augmentation transforms.
    
    Args:
        image_size: Size of input images
        
    Returns:
        Composed transforms for test-time augmentation
    """
    return transforms.Compose([
        transforms.RandomResizedCrop(image_size, scale=(0.9, 1.0)),
        transforms.RandomHorizontalFlip(),
    ])


@torch.no_grad()
def ttcs_predict(encoder: nn.Module, head: nn.Module, episode, *, passes: int = 8, 
                image_size: int = 32, device=None, combine: str = "mean_prob", 
                enable_mc_dropout: bool = True, enable_tta: bool = True, **advanced_kwargs) -> torch.Tensor:
    """
    TTCS (Test-Time Compute Scaling) prediction with uncertainty estimation.
    
    This is the FIRST PUBLIC IMPLEMENTATION of Test-Time Compute Scaling for meta-learning!
    
    Features:
    - MC-Dropout for uncertainty estimation
    - Test-Time Augmentation (TTA) for images  
    - Ensemble prediction across multiple stochastic passes
    - Mean probability vs mean logit combining strategies
    
    Args:
        encoder: Feature encoder network
        head: Classification head
        episode: Episode with support and query data
        passes: Number of stochastic forward passes
        image_size: Input image size for TTA
        device: Device to run computation on
        combine: Combination strategy ('mean_prob' or 'mean_logit')
        enable_mc_dropout: Enable Monte Carlo dropout
        enable_tta: Enable test-time augmentation
        **advanced_kwargs: Additional advanced parameters
        
    Returns:
        Ensemble predictions for query set
    """
    if device is None:
        device = next(encoder.parameters()).device
    
    # Move episode to device
    support_x = episode.support_x.to(device)
    support_y = episode.support_y.to(device)
    query_x = episode.query_x.to(device)
    
    # Prepare TTA transforms if enabled
    if enable_tta:
        tta_transform = tta_transforms(image_size)
    
    # Enable dropout for MC-Dropout if enabled
    if enable_mc_dropout:
        _enable_dropout_for_inference(encoder)
        _enable_dropout_for_inference(head)
    
    predictions_list = []
    
    for pass_idx in range(passes):
        # Apply TTA to support and query if enabled
        if enable_tta:
            # Apply random augmentation to support set
            support_x_aug = torch.stack([
                tta_transform(img) if len(img.shape) == 3 else img 
                for img in support_x
            ])
            query_x_aug = torch.stack([
                tta_transform(img) if len(img.shape) == 3 else img 
                for img in query_x
            ])
        else:
            support_x_aug = support_x
            query_x_aug = query_x
        
        # Forward pass through encoder
        support_features = encoder(support_x_aug)
        query_features = encoder(query_x_aug)
        
        # Forward pass through head (prototypical or other)
        if hasattr(head, 'forward'):
            # Standard head with forward method
            logits = head(support_features, support_y, query_features)
        else:
            # Functional head or simple classifier
            # Compute prototypes
            num_classes = len(torch.unique(support_y))
            prototypes = torch.zeros(num_classes, support_features.size(-1), device=device)
            
            for class_idx in range(num_classes):
                class_mask = support_y == class_idx
                if class_mask.sum() > 0:
                    prototypes[class_idx] = support_features[class_mask].mean(dim=0)
            
            # Compute distances and logits
            distances = torch.cdist(query_features, prototypes)
            logits = -distances  # Negative distance as logits
        
        predictions_list.append(logits)
    
    # Disable dropout after inference if it was enabled
    if enable_mc_dropout:
        _disable_dropout_after_inference(encoder)
        _disable_dropout_after_inference(head)
    
    # Combine predictions
    if combine == "mean_prob":
        # Convert to probabilities, average, then back to logits
        probs_list = [torch.softmax(pred, dim=-1) for pred in predictions_list]
        mean_probs = torch.stack(probs_list).mean(dim=0)
        # Add small epsilon to avoid log(0)
        epsilon = 1e-8
        ensemble_logits = torch.log(mean_probs + epsilon)
    elif combine == "mean_logit":
        # Average logits directly
        ensemble_logits = torch.stack(predictions_list).mean(dim=0)
    else:
        raise ValueError(f"Unknown combine strategy: {combine}")
    
    return ensemble_logits


@torch.no_grad()
def ttcs_predict_advanced(encoder: nn.Module, head: nn.Module, episode, *, passes: int = 8,
                         image_size: int = 32, device=None, combine: str = "mean_prob",
                         enable_mc_dropout: bool = True, enable_tta: bool = True,
                         uncertainty_estimation: bool = False, temperature_scaling: float = 1.0,
                         dropout_rate_override: Optional[float] = None,
                         tta_strength: float = 1.0, **kwargs) -> Dict[str, torch.Tensor]:
    """
    Advanced TTCS prediction with comprehensive uncertainty estimation and analysis.
    
    Args:
        encoder: Feature encoder network
        head: Classification head
        episode: Episode with support and query data
        passes: Number of stochastic forward passes
        image_size: Input image size for TTA
        device: Device to run computation on
        combine: Combination strategy ('mean_prob' or 'mean_logit')
        enable_mc_dropout: Enable Monte Carlo dropout
        enable_tta: Enable test-time augmentation
        uncertainty_estimation: Return uncertainty metrics
        temperature_scaling: Temperature for calibration
        dropout_rate_override: Override dropout rate for MC-Dropout
        tta_strength: Strength of test-time augmentation
        **kwargs: Additional parameters
        
    Returns:
        Dictionary containing predictions and uncertainty metrics
    """
    if device is None:
        device = next(encoder.parameters()).device
    
    # Move episode to device
    support_x = episode.support_x.to(device)
    support_y = episode.support_y.to(device)
    query_x = episode.query_x.to(device)
    
    # Override dropout rates if specified
    original_dropout_rates = {}
    if dropout_rate_override is not None:
        original_dropout_rates = _override_dropout_rates(encoder, dropout_rate_override)
        original_dropout_rates.update(_override_dropout_rates(head, dropout_rate_override))
    
    # Prepare TTA transforms with specified strength
    if enable_tta:
        tta_transform = _create_adaptive_tta_transforms(image_size, tta_strength)
    
    # Enable dropout for MC-Dropout if enabled
    if enable_mc_dropout:
        _enable_dropout_for_inference(encoder)
        _enable_dropout_for_inference(head)
    
    predictions_list = []
    feature_variance_list = []
    
    for pass_idx in range(passes):
        # Apply TTA with specified strength
        if enable_tta:
            support_x_aug = _apply_tta_batch(support_x, tta_transform)
            query_x_aug = _apply_tta_batch(query_x, tta_transform)
        else:
            support_x_aug = support_x
            query_x_aug = query_x
        
        # Forward pass through encoder
        support_features = encoder(support_x_aug)
        query_features = encoder(query_x_aug)
        
        # Track feature variance for uncertainty estimation
        if uncertainty_estimation:
            feature_variance_list.append(query_features.var(dim=0))
        
        # Forward pass through head
        logits = _forward_through_head(head, support_features, support_y, query_features, device)
        
        # Apply temperature scaling
        if temperature_scaling != 1.0:
            logits = logits / temperature_scaling
        
        predictions_list.append(logits)
    
    # Restore original dropout rates if they were overridden
    if dropout_rate_override is not None:
        _restore_dropout_rates(encoder, original_dropout_rates)
        _restore_dropout_rates(head, original_dropout_rates)
    
    # Disable dropout after inference if it was enabled
    if enable_mc_dropout:
        _disable_dropout_after_inference(encoder)
        _disable_dropout_after_inference(head)
    
    # Combine predictions
    ensemble_logits = _combine_predictions(predictions_list, combine)
    
    # Prepare return dictionary
    results = {
        'logits': ensemble_logits,
        'probabilities': torch.softmax(ensemble_logits, dim=-1)
    }
    
    # Add uncertainty estimation if requested
    if uncertainty_estimation:
        uncertainty_metrics = _compute_uncertainty_metrics(
            predictions_list, feature_variance_list, combine
        )
        results.update(uncertainty_metrics)
    
    return results


def _enable_dropout_for_inference(module: nn.Module):
    """Enable dropout layers for Monte Carlo inference."""
    for m in module.modules():
        if isinstance(m, nn.Dropout):
            m.train()


def _disable_dropout_after_inference(module: nn.Module):
    """Disable dropout layers after Monte Carlo inference."""
    for m in module.modules():
        if isinstance(m, nn.Dropout):
            m.eval()


def _override_dropout_rates(module: nn.Module, dropout_rate: float) -> Dict[str, float]:
    """Override dropout rates and return original rates."""
    original_rates = {}
    for name, m in module.named_modules():
        if isinstance(m, nn.Dropout):
            original_rates[name] = m.p
            m.p = dropout_rate
    return original_rates


def _restore_dropout_rates(module: nn.Module, original_rates: Dict[str, float]):
    """Restore original dropout rates."""
    for name, m in module.named_modules():
        if isinstance(m, nn.Dropout) and name in original_rates:
            m.p = original_rates[name]


def _create_adaptive_tta_transforms(image_size: int, strength: float) -> transforms.Compose:
    """Create test-time augmentation transforms with adaptive strength."""
    return transforms.Compose([
        transforms.RandomResizedCrop(
            image_size, 
            scale=(1.0 - 0.1 * strength, 1.0)
        ),
        transforms.RandomHorizontalFlip(p=0.5 * strength),
    ])


def _apply_tta_batch(batch: torch.Tensor, transform: transforms.Compose) -> torch.Tensor:
    """Apply TTA transforms to a batch of images."""
    return torch.stack([
        transform(img) if len(img.shape) == 3 else img 
        for img in batch
    ])


def _forward_through_head(head: nn.Module, support_features: torch.Tensor, 
                         support_y: torch.Tensor, query_features: torch.Tensor,
                         device: torch.device) -> torch.Tensor:
    """Forward pass through classification head."""
    if hasattr(head, 'forward'):
        # Standard head with forward method
        return head(support_features, support_y, query_features)
    else:
        # Functional head - compute prototypes
        num_classes = len(torch.unique(support_y))
        prototypes = torch.zeros(num_classes, support_features.size(-1), device=device)
        
        for class_idx in range(num_classes):
            class_mask = support_y == class_idx
            if class_mask.sum() > 0:
                prototypes[class_idx] = support_features[class_mask].mean(dim=0)
        
        # Compute distances and logits
        distances = torch.cdist(query_features, prototypes)
        return -distances  # Negative distance as logits


def _combine_predictions(predictions_list: List[torch.Tensor], combine: str) -> torch.Tensor:
    """Combine multiple predictions using specified strategy."""
    if combine == "mean_prob":
        # Convert to probabilities, average, then back to logits
        probs_list = [torch.softmax(pred, dim=-1) for pred in predictions_list]
        mean_probs = torch.stack(probs_list).mean(dim=0)
        # Add small epsilon to avoid log(0)
        epsilon = 1e-8
        return torch.log(mean_probs + epsilon)
    elif combine == "mean_logit":
        # Average logits directly
        return torch.stack(predictions_list).mean(dim=0)
    else:
        raise ValueError(f"Unknown combine strategy: {combine}")


def _compute_uncertainty_metrics(predictions_list: List[torch.Tensor], 
                                feature_variance_list: List[torch.Tensor],
                                combine: str) -> Dict[str, torch.Tensor]:
    """Compute comprehensive uncertainty metrics."""
    # Convert predictions to probabilities
    probs_list = [torch.softmax(pred, dim=-1) for pred in predictions_list]
    prob_stack = torch.stack(probs_list)  # [passes, batch_size, num_classes]
    
    # Predictive uncertainty (entropy of mean prediction)
    mean_probs = prob_stack.mean(dim=0)
    predictive_entropy = -torch.sum(mean_probs * torch.log(mean_probs + 1e-8), dim=-1)
    
    # Aleatoric uncertainty (mean entropy of predictions)
    entropies = -torch.sum(prob_stack * torch.log(prob_stack + 1e-8), dim=-1)
    aleatoric_entropy = entropies.mean(dim=0)
    
    # Epistemic uncertainty (difference)
    epistemic_entropy = predictive_entropy - aleatoric_entropy
    
    # Prediction variance
    prediction_variance = prob_stack.var(dim=0).mean(dim=-1)
    
    # Confidence (max probability)
    confidence = mean_probs.max(dim=-1)[0]
    
    uncertainty_metrics = {
        'predictive_entropy': predictive_entropy,
        'aleatoric_entropy': aleatoric_entropy, 
        'epistemic_entropy': epistemic_entropy,
        'prediction_variance': prediction_variance,
        'confidence': confidence,
        'all_predictions': prob_stack
    }
    
    # Add feature variance if available
    if feature_variance_list:
        feature_variance_stack = torch.stack(feature_variance_list)
        uncertainty_metrics['feature_variance'] = feature_variance_stack.mean(dim=0)
    
    return uncertainty_metrics