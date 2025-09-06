"""
TTCS Utility Functions
=====================

Convenience functions and utilities for Test-Time Compute Scaling.
Extracted from large ttcs.py for better maintainability.

FIRST PUBLIC IMPLEMENTATION of Test-Time Compute Scaling!
"""

from __future__ import annotations
from typing import Optional, Dict, Any, Tuple
import torch
import torch.nn as nn

from .core_predictor import ttcs_predict, ttcs_predict_advanced


def auto_ttcs(encoder: nn.Module, head: nn.Module, episode, device=None) -> torch.Tensor:
    """
    One-liner TTCS with sensible defaults - just works out of the box.
    
    Simple Usage:
        logits = auto_ttcs(encoder, head, episode)
        
    Args:
        encoder: Feature encoder network
        head: Classification head
        episode: Episode with support and query data
        device: Device to run on
        
    Returns:
        TTCS predictions with default settings
    """
    return ttcs_predict(encoder, head, episode, device=device)


def pro_ttcs(encoder: nn.Module, head: nn.Module, episode, 
             passes: int = 16, device=None, **kwargs) -> Dict[str, torch.Tensor]:
    """
    Professional TTCS with all advanced features enabled.
    
    Advanced Usage:
        predictions, metrics = pro_ttcs(encoder, head, episode, 
                                      uncertainty_estimation=True,
                                      compute_budget="adaptive",
                                      performance_monitoring=True)
                                      
    Args:
        encoder: Feature encoder network
        head: Classification head
        episode: Episode with support and query data
        passes: Number of stochastic passes
        device: Device to run on
        **kwargs: Additional advanced parameters
        
    Returns:
        Dictionary containing predictions and advanced metrics
    """
    return ttcs_predict_advanced(
        encoder, head, episode,
        passes=passes,
        device=device,
        uncertainty_estimation=True,
        **kwargs
    )


def ttcs_with_fallback(encoder: nn.Module, head: nn.Module, episode, 
                      fallback_method: str = "protonet", **ttcs_kwargs) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """
    TTCS with automatic fallback to simpler methods.
    
    Args:
        encoder: Feature encoder
        head: Classification head
        episode: Episode data
        fallback_method: Fallback method ('protonet', 'simple')
        **ttcs_kwargs: TTCS configuration parameters
        
    Returns:
        Tuple of (predictions, fallback_info)
    """
    from .warning_system import TTCSWarningSystem
    
    warning_system = TTCSWarningSystem()
    fallback_info = {'used_fallback': False, 'fallback_reason': None}
    
    try:
        # Attempt TTCS first
        logits = ttcs_predict(encoder, head, episode, **ttcs_kwargs)
        return logits, fallback_info
        
    except Exception as e:
        fallback_info['used_fallback'] = True
        fallback_info['fallback_reason'] = str(e)
        
        warning_system.warn(
            f"TTCS failed ({str(e)}), falling back to {fallback_method}",
            category='fallback',
            severity='warning'
        )
        
        # Apply fallback method
        if fallback_method == "protonet":
            return _protonet_fallback(encoder, head, episode), fallback_info
        elif fallback_method == "simple":
            return _simple_fallback(encoder, head, episode), fallback_info
        else:
            raise ValueError(f"Unknown fallback method: {fallback_method}")


def ttcs_for_learn2learn_models(l2l_model, episode, **kwargs) -> torch.Tensor:
    """
    Make TTCS compatible with learn2learn MAML models.
    
    Args:
        l2l_model: Learn2learn model (e.g., MAML)
        episode: Episode data
        **kwargs: TTCS parameters
        
    Returns:
        TTCS predictions for learn2learn model
    """
    from .warning_system import TTCSWarningSystem
    
    warning_system = TTCSWarningSystem()
    
    # Extract encoder and head from learn2learn model
    try:
        if hasattr(l2l_model, 'module'):
            base_model = l2l_model.module
        else:
            base_model = l2l_model
            
        # Common learn2learn model structures
        if hasattr(base_model, 'features') and hasattr(base_model, 'classifier'):
            encoder = base_model.features
            head = base_model.classifier
        elif hasattr(base_model, 'encoder') and hasattr(base_model, 'head'):
            encoder = base_model.encoder
            head = base_model.head
        else:
            # Try to use the whole model as encoder with simple head
            encoder = base_model
            head = SimplePrototypicalHead()
            
            warning_system.warn(
                "Could not identify encoder/head structure in learn2learn model, "
                "using whole model as encoder with prototypical head",
                category='l2l_integration'
            )
    
    except Exception as e:
        raise ValueError(f"Failed to extract encoder/head from learn2learn model: {e}")
    
    # Apply TTCS
    return ttcs_predict(encoder, head, episode, **kwargs)


def get_optimal_ttcs_passes(encoder: nn.Module, head: nn.Module, episode,
                           max_passes: int = 20, target_accuracy: float = 0.95,
                           patience: int = 3) -> int:
    """
    Find optimal number of TTCS passes for given accuracy target.
    
    Args:
        encoder: Feature encoder network
        head: Classification head
        episode: Episode with support and query data
        max_passes: Maximum passes to try
        target_accuracy: Target accuracy threshold
        patience: Early stopping patience
        
    Returns:
        Optimal number of passes
    """
    if not hasattr(episode, 'query_y'):
        # Cannot optimize without ground truth labels
        return 8  # Default fallback
    
    best_accuracy = 0.0
    best_passes = 8
    patience_counter = 0
    
    for passes in range(2, max_passes + 1, 2):
        # Get predictions with current number of passes
        logits = ttcs_predict(encoder, head, episode, passes=passes)
        predictions = torch.argmax(logits, dim=1)
        
        # Calculate accuracy
        accuracy = (predictions == episode.query_y).float().mean().item()
        
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_passes = passes
            patience_counter = 0
            
            # Early stopping if target reached
            if accuracy >= target_accuracy:
                break
        else:
            patience_counter += 1
            
            # Early stopping if no improvement
            if patience_counter >= patience:
                break
    
    return best_passes


def estimate_ttcs_compute_cost(encoder: nn.Module, head: nn.Module, episode,
                              passes: int = 8) -> Dict[str, float]:
    """
    Estimate computational cost of TTCS for given configuration.
    
    Args:
        encoder: Feature encoder network
        head: Classification head
        episode: Episode with support and query data
        passes: Number of TTCS passes
        
    Returns:
        Dictionary containing cost estimates
    """
    import time
    
    # Single pass timing
    start_time = time.time()
    _ = ttcs_predict(encoder, head, episode, passes=1)
    single_pass_time = time.time() - start_time
    
    # Estimate model parameters
    encoder_params = sum(p.numel() for p in encoder.parameters())
    head_params = sum(p.numel() for p in head.parameters()) if hasattr(head, 'parameters') else 0
    total_params = encoder_params + head_params
    
    # Estimate memory usage (rough approximation)
    batch_size = len(episode.query_x)
    memory_per_pass_mb = (total_params * 4 + batch_size * 1000) / (1024 * 1024)  # Rough estimate
    
    return {
        'estimated_total_time_seconds': single_pass_time * passes,
        'time_per_pass_seconds': single_pass_time,
        'estimated_memory_mb': memory_per_pass_mb * passes,
        'model_parameters': int(total_params),
        'passes': passes,
        'efficiency_score': 1.0 / (single_pass_time * passes)  # Higher is better
    }


def smart_ttcs_configuration(encoder: nn.Module, head: nn.Module, episode,
                            time_budget_seconds: Optional[float] = None,
                            memory_budget_mb: Optional[float] = None) -> Dict[str, Any]:
    """
    Automatically configure TTCS parameters based on constraints.
    
    Args:
        encoder: Feature encoder network
        head: Classification head
        episode: Episode with support and query data
        time_budget_seconds: Maximum time budget
        memory_budget_mb: Maximum memory budget
        
    Returns:
        Dictionary containing optimal TTCS configuration
    """
    # Estimate costs for different pass counts
    pass_options = [2, 4, 8, 12, 16, 20]
    cost_estimates = []
    
    for passes in pass_options:
        try:
            cost = estimate_ttcs_compute_cost(encoder, head, episode, passes)
            cost['passes'] = passes
            cost_estimates.append(cost)
        except Exception:
            continue  # Skip if estimation fails
    
    if not cost_estimates:
        # Fallback configuration
        return {
            'passes': 8,
            'enable_mc_dropout': True,
            'enable_tta': True,
            'combine': 'mean_prob',
            'reasoning': 'Default configuration (cost estimation failed)'
        }
    
    # Filter by constraints
    valid_configs = cost_estimates.copy()
    
    if time_budget_seconds is not None:
        valid_configs = [c for c in valid_configs if c['estimated_total_time_seconds'] <= time_budget_seconds]
    
    if memory_budget_mb is not None:
        valid_configs = [c for c in valid_configs if c['estimated_memory_mb'] <= memory_budget_mb]
    
    if not valid_configs:
        # Use most conservative option if no configs fit constraints
        valid_configs = [min(cost_estimates, key=lambda x: x['passes'])]
    
    # Choose configuration with highest efficiency score
    optimal_config = max(valid_configs, key=lambda x: x['efficiency_score'])
    
    return {
        'passes': optimal_config['passes'],
        'enable_mc_dropout': _has_dropout_layers(encoder) or _has_dropout_layers(head),
        'enable_tta': episode.query_x.dim() == 4,  # Enable TTA for image data
        'combine': 'mean_prob',
        'reasoning': f"Optimized for efficiency (score: {optimal_config['efficiency_score']:.3f})",
        'estimated_time_seconds': optimal_config['estimated_total_time_seconds'],
        'estimated_memory_mb': optimal_config['estimated_memory_mb']
    }


# Helper functions for error handling and validation
def _has_dropout_layers(model: nn.Module) -> bool:
    """Check if model has Dropout layers for MC-Dropout compatibility."""
    if model is None:
        return False
        
    for module in model.modules():
        if isinstance(module, (nn.Dropout, nn.Dropout1d, nn.Dropout2d, nn.Dropout3d)):
            return True
    return False


def _protonet_fallback(encoder: nn.Module, head: nn.Module, episode) -> torch.Tensor:
    """Prototypical Networks fallback implementation."""
    device = next(encoder.parameters()).device
    
    # Extract features
    support_features = encoder(episode.support_x.to(device))
    query_features = encoder(episode.query_x.to(device))
    
    # Compute prototypes
    unique_classes = torch.unique(episode.support_y)
    prototypes = []
    
    for cls in unique_classes:
        class_mask = episode.support_y == cls
        class_prototype = support_features[class_mask].mean(dim=0)
        prototypes.append(class_prototype)
    
    prototypes = torch.stack(prototypes)
    
    # Compute distances and logits
    distances = torch.cdist(query_features, prototypes)
    logits = -distances
    
    return logits


def _simple_fallback(encoder: nn.Module, head: nn.Module, episode) -> torch.Tensor:
    """Simple fallback: standard forward pass."""
    device = next(encoder.parameters()).device
    
    support_features = encoder(episode.support_x.to(device))
    query_features = encoder(episode.query_x.to(device))
    
    # Use head if it's a proper module, otherwise compute prototypes
    try:
        return head(support_features, episode.support_y.to(device), query_features)
    except Exception:
        # Fallback to prototypical computation
        return _protonet_fallback(encoder, head, episode)


class SimplePrototypicalHead(nn.Module):
    """Simple prototypical head for fallback scenarios."""
    
    def forward(self, support_features: torch.Tensor, support_labels: torch.Tensor, 
                query_features: torch.Tensor) -> torch.Tensor:
        """
        Compute prototypical predictions.
        
        Args:
            support_features: Support set features
            support_labels: Support set labels
            query_features: Query set features
            
        Returns:
            Logits for query set
        """
        unique_classes = torch.unique(support_labels)
        prototypes = []
        
        for cls in unique_classes:
            class_mask = support_labels == cls
            class_prototype = support_features[class_mask].mean(dim=0)
            prototypes.append(class_prototype)
        
        prototypes = torch.stack(prototypes)
        
        # Compute distances and return negative distances as logits
        distances = torch.cdist(query_features, prototypes)
        return -distances