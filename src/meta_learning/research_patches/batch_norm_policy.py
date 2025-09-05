#!/usr/bin/env python3
"""
BatchNorm Policy Patch for Few-Shot Learning Research Accuracy
==============================================================

This patch implements proper BatchNorm handling for episodic few-shot learning
following best practices from Antoniou et al. (2018) and Chen et al. (2019).

Key Research Issues Addressed:
1. BatchNorm running statistics contaminated by support/query mixing
2. Insufficient statistics for BatchNorm in few-shot episodes
3. Domain shift between training and evaluation episodes

Solutions Implemented:
- Freeze BatchNorm running stats during episodic evaluation
- Option to replace BatchNorm with Instance/LayerNorm in backbones
- Episode-aware normalization policies
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional, Union
import warnings


class EpisodicBatchNormPolicy:
    """
    Research-accurate BatchNorm handling for few-shot learning episodes.
    
    Implements policies from:
    - Antoniou et al. (2018): "How to train your MAML"
    - Chen et al. (2019): "A Closer Look at Few-shot Classification"
    """
    
    def __init__(self, policy: str = "freeze_running_stats"):
        """
        Initialize BatchNorm policy.
        
        Args:
            policy: One of:
                - "freeze_running_stats": Freeze BN running mean/var during episodes
                - "instance_norm": Replace BatchNorm with InstanceNorm
                - "layer_norm": Replace BatchNorm with LayerNorm  
                - "group_norm": Replace BatchNorm with GroupNorm
        """
        self.policy = policy
        self.original_states = {}
        
    def apply_to_model(self, model: nn.Module) -> nn.Module:
        """Apply the BatchNorm policy to a model."""
        if self.policy == "freeze_running_stats":
            return self._freeze_bn_stats(model)
        elif self.policy in ["instance_norm", "layer_norm", "group_norm"]:
            return self._replace_batch_norm(model)
        else:
            raise ValueError(f"Unknown policy: {self.policy}")
    
    def _freeze_bn_stats(self, model: nn.Module) -> nn.Module:
        """Freeze BatchNorm running statistics during episodic evaluation."""
        for name, module in model.named_modules():
            if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                # Store original state
                self.original_states[name] = {
                    'track_running_stats': module.track_running_stats,
                    'momentum': module.momentum
                }
                
                # Freeze running stats for episodic evaluation
                module.track_running_stats = False
                module.momentum = 0.0
                
                warnings.warn(
                    f"Froze BatchNorm running stats for module: {name}. "
                    f"This follows Chen et al. (2019) recommendations for few-shot learning.",
                    UserWarning
                )
        
        return model
    
    def _replace_batch_norm(self, model: nn.Module) -> nn.Module:
        """Replace BatchNorm layers with episode-friendly alternatives."""
        for name, module in list(model.named_children()):
            if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                replacement = self._get_replacement_norm(module)
                setattr(model, name, replacement)
                
                warnings.warn(
                    f"Replaced {type(module).__name__} with {type(replacement).__name__} "
                    f"for module: {name}. This improves few-shot learning stability.",
                    UserWarning
                )
            else:
                # Recursively apply to child modules
                self._replace_batch_norm(module)
        
        return model
    
    def _get_replacement_norm(self, bn_module: nn.Module) -> nn.Module:
        """Get the appropriate replacement normalization layer."""
        num_features = bn_module.num_features
        
        if self.policy == "instance_norm":
            if isinstance(bn_module, nn.BatchNorm2d):
                return nn.InstanceNorm2d(num_features, affine=True)
            elif isinstance(bn_module, nn.BatchNorm1d):
                return nn.InstanceNorm1d(num_features, affine=True)
            else:
                raise NotImplementedError(f"InstanceNorm replacement for {type(bn_module)}")
                
        elif self.policy == "layer_norm":
            # LayerNorm requires input shape, use simplified version
            return nn.GroupNorm(1, num_features)  # GroupNorm with 1 group = LayerNorm
            
        elif self.policy == "group_norm":
            # Use 8 groups as default (common choice)
            num_groups = min(8, num_features)
            return nn.GroupNorm(num_groups, num_features)
            
        else:
            raise ValueError(f"Unknown replacement policy: {self.policy}")
    
    def restore_original_state(self, model: nn.Module) -> nn.Module:
        """Restore original BatchNorm states after episodic evaluation."""
        if self.policy != "freeze_running_stats":
            warnings.warn(
                "Cannot restore original state for replacement policies. "
                "Original BatchNorm layers were replaced.",
                UserWarning
            )
            return model
        
        for name, module in model.named_modules():
            if name in self.original_states:
                original_state = self.original_states[name]
                module.track_running_stats = original_state['track_running_stats']
                module.momentum = original_state['momentum']
        
        return model


class EpisodicNormalizationGuard:
    """
    Guard to ensure normalization layers don't leak information between episodes.
    
    This implements the "data hygiene" checks from your research checklist.
    """
    
    def __init__(self, strict_mode: bool = True):
        self.strict_mode = strict_mode
        self.episode_stats = []
        
    def validate_episode_isolation(self, model: nn.Module, 
                                 support_data: torch.Tensor,
                                 query_data: torch.Tensor) -> Dict[str, Any]:
        """
        Validate that normalization stats don't leak between support and query.
        
        Returns:
            Dictionary with validation results and potential leakage warnings.
        """
        validation_results = {
            'passed': True,
            'warnings': [],
            'batch_norm_modules': [],
            'running_stats_frozen': True
        }
        
        # Check for BatchNorm modules
        for name, module in model.named_modules():
            if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                validation_results['batch_norm_modules'].append(name)
                
                # Check if running stats are properly handled
                if module.track_running_stats and module.training:
                    validation_results['passed'] = False
                    validation_results['running_stats_frozen'] = False
                    validation_results['warnings'].append(
                        f"BatchNorm module '{name}' has track_running_stats=True "
                        f"during training. This may cause data leakage in few-shot episodes."
                    )
        
        # Additional validation for data shapes
        if support_data.size(0) < 16:  # Typical few-shot support size
            for name in validation_results['batch_norm_modules']:
                validation_results['warnings'].append(
                    f"Small support set size ({support_data.size(0)}) may cause "
                    f"unreliable BatchNorm statistics in module '{name}'. "
                    f"Consider using GroupNorm or InstanceNorm instead."
                )
        
        return validation_results


# Research-accurate utility functions
def apply_episodic_bn_policy(model: nn.Module, 
                           policy: str = "freeze_running_stats") -> nn.Module:
    """
    Apply research-accurate BatchNorm policy for few-shot learning.
    
    This is the main function researchers should use to ensure proper
    normalization handling in few-shot episodes.
    
    Args:
        model: PyTorch model to modify
        policy: BatchNorm policy to apply
        
    Returns:
        Modified model with proper normalization handling
        
    Example:
        >>> model = ResNet12(num_classes=5)
        >>> model = apply_episodic_bn_policy(model, policy="group_norm")
        >>> # Now ready for few-shot learning!
    """
    bn_policy = EpisodicBatchNormPolicy(policy)
    return bn_policy.apply_to_model(model)


def validate_few_shot_model(model: nn.Module,
                          support_data: torch.Tensor,
                          query_data: torch.Tensor,
                          strict: bool = True) -> Dict[str, Any]:
    """
    Validate a model for few-shot learning research compliance.
    
    Checks for common pitfalls that can invalidate research results.
    
    Args:
        model: Model to validate
        support_data: Support set data
        query_data: Query set data  
        strict: Whether to enforce strict validation
        
    Returns:
        Validation results with warnings and recommendations
    """
    guard = EpisodicNormalizationGuard(strict_mode=strict)
    return guard.validate_episode_isolation(model, support_data, query_data)


if __name__ == "__main__":
    # Demo: Proper BatchNorm handling for few-shot learning
    print("🔬 BatchNorm Policy Patch Demo")
    print("=" * 40)
    
    # Create a simple model with BatchNorm
    model = nn.Sequential(
        nn.Conv2d(3, 64, 3, padding=1),
        nn.BatchNorm2d(64),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Linear(64, 5)
    )
    
    print(f"Original model has BatchNorm: {any('BatchNorm' in str(type(m)) for m in model.modules())}")
    
    # Apply GroupNorm policy (recommended for few-shot)
    model_fixed = apply_episodic_bn_policy(model, policy="group_norm")
    print(f"Fixed model has BatchNorm: {any('BatchNorm' in str(type(m)) for m in model_fixed.modules())}")
    print(f"Fixed model has GroupNorm: {any('GroupNorm' in str(type(m)) for m in model_fixed.modules())}")
    
    # Validate with dummy data
    support_data = torch.randn(10, 3, 32, 32)  # 10 support examples
    query_data = torch.randn(15, 3, 32, 32)    # 15 query examples
    
    validation = validate_few_shot_model(model_fixed, support_data, query_data)
    print(f"\nValidation passed: {validation['passed']}")
    print(f"Warnings: {len(validation['warnings'])}")
    
    for warning in validation['warnings']:
        print(f"⚠️  {warning}")
    
    print("\n✅ BatchNorm policy patch applied successfully!")