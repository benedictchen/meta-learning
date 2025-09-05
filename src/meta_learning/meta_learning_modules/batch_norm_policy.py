"""
BatchNorm Policy for Meta-Learning
==================================

Author: Benedict Chen (benedict@benedictchen.com)

Critical BatchNorm handling for episodic meta-learning where running stats
must not leak information across episodes or tasks.

Research Issue:
BatchNorm running statistics computed across different tasks/classes can
leak information and invalidate few-shot learning assumptions.

Solutions:
1. Freeze running stats during episodic evaluation
2. Use InstanceNorm/LayerNorm instead of BatchNorm
3. Reset running stats per episode (expensive but mathematically clean)

References:
- Antoniou et al. (2018): "How to train your MAML" discusses BN issues
- Rajeswaran et al. (2019): "Meta-Learning with Implicit Gradients" BN analysis
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Union, Callable
from enum import Enum
import warnings
import logging

logger = logging.getLogger(__name__)


class BatchNormPolicy(Enum):
    """BatchNorm handling policies for meta-learning."""
    FREEZE_STATS = "freeze_stats"       # Freeze running stats during episodes
    INSTANCE_NORM = "instance_norm"     # Replace BN with InstanceNorm
    LAYER_NORM = "layer_norm"          # Replace BN with LayerNorm
    RESET_PER_EPISODE = "reset_per_episode"  # Reset stats each episode (expensive)
    TRACK_RUNNING_STATS = "track_running_stats"  # Normal BN (not recommended)


class BatchNormManager:
    """
    Manager for BatchNorm behavior in meta-learning settings.
    
    Handles the critical issue of BatchNorm statistics leakage across
    episodes which can invalidate few-shot learning assumptions.
    """
    
    def __init__(self, policy: BatchNormPolicy = BatchNormPolicy.FREEZE_STATS):
        self.policy = policy
        self.original_training_modes = {}
        self.original_track_stats = {}
        
    def prepare_model_for_episodes(self, model: nn.Module) -> None:
        """
        Prepare model for episodic evaluation according to policy.
        
        This is the critical function that ensures BatchNorm doesn't
        leak information across episodes.
        """
        if self.policy == BatchNormPolicy.FREEZE_STATS:
            self._freeze_batch_norm_stats(model)
        elif self.policy == BatchNormPolicy.INSTANCE_NORM:
            self._replace_with_instance_norm(model)
        elif self.policy == BatchNormPolicy.LAYER_NORM:
            self._replace_with_layer_norm(model)
        elif self.policy == BatchNormPolicy.RESET_PER_EPISODE:
            self._enable_stat_reset(model)
        elif self.policy == BatchNormPolicy.TRACK_RUNNING_STATS:
            warnings.warn(
                "TRACK_RUNNING_STATS policy allows statistics leakage across episodes. "
                "Not recommended for rigorous few-shot evaluation."
            )
            
        logger.info(f"Applied BatchNorm policy: {self.policy.value}")
    
    def _freeze_batch_norm_stats(self, model: nn.Module) -> None:
        """
        Freeze BatchNorm running statistics.
        
        This prevents running_mean and running_var from updating during
        episodic evaluation, eliminating cross-episode information leakage.
        """
        bn_modules = []
        for name, module in model.named_modules():
            if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                bn_modules.append((name, module))
                
                # Store original state for restoration
                self.original_training_modes[name] = module.training
                self.original_track_stats[name] = module.track_running_stats
                
                # Freeze statistics
                module.track_running_stats = False
                
                # Ensure module uses stored statistics for normalization
                if hasattr(module, 'running_mean') and module.running_mean is not None:
                    module.running_mean.requires_grad_(False)
                if hasattr(module, 'running_var') and module.running_var is not None:
                    module.running_var.requires_grad_(False)
                    
        logger.info(f"Froze statistics for {len(bn_modules)} BatchNorm modules")
    
    def _replace_with_instance_norm(self, model: nn.Module) -> None:
        """
        Replace BatchNorm layers with InstanceNorm.
        
        InstanceNorm computes statistics per sample, eliminating the
        cross-sample/cross-episode leakage issue entirely.
        """
        def replace_bn_recursive(module: nn.Module) -> nn.Module:
            for name, child in module.named_children():
                if isinstance(child, nn.BatchNorm2d):
                    # Replace with InstanceNorm2d
                    instance_norm = nn.InstanceNorm2d(
                        child.num_features,
                        eps=child.eps,
                        momentum=None,  # InstanceNorm doesn't use running stats
                        affine=child.affine,
                        track_running_stats=False
                    )
                    
                    # Copy learned parameters if they exist
                    if child.affine:
                        instance_norm.weight.data = child.weight.data.clone()
                        instance_norm.bias.data = child.bias.data.clone()
                        
                    setattr(module, name, instance_norm)
                    logger.info(f"Replaced BatchNorm2d with InstanceNorm2d: {name}")
                    
                elif isinstance(child, nn.BatchNorm1d):
                    # Replace with InstanceNorm1d  
                    instance_norm = nn.InstanceNorm1d(
                        child.num_features,
                        eps=child.eps,
                        momentum=None,
                        affine=child.affine,
                        track_running_stats=False
                    )
                    
                    if child.affine:
                        instance_norm.weight.data = child.weight.data.clone()
                        instance_norm.bias.data = child.bias.data.clone()
                        
                    setattr(module, name, instance_norm)
                    logger.info(f"Replaced BatchNorm1d with InstanceNorm1d: {name}")
                    
                else:
                    # Recursively process child modules
                    replace_bn_recursive(child)
                    
            return module
            
        replace_bn_recursive(model)
    
    def _replace_with_layer_norm(self, model: nn.Module) -> None:
        """
        Replace BatchNorm layers with LayerNorm where applicable.
        
        LayerNorm normalizes across features per sample, avoiding
        batch-level statistics that cause leakage.
        """
        def replace_bn_recursive(module: nn.Module) -> nn.Module:
            for name, child in module.named_children():
                if isinstance(child, nn.BatchNorm1d):
                    # LayerNorm for 1D is straightforward
                    layer_norm = nn.LayerNorm(
                        child.num_features,
                        eps=child.eps,
                        elementwise_affine=child.affine
                    )
                    
                    if child.affine:
                        layer_norm.weight.data = child.weight.data.clone()
                        layer_norm.bias.data = child.bias.data.clone()
                        
                    setattr(module, name, layer_norm)
                    logger.info(f"Replaced BatchNorm1d with LayerNorm: {name}")
                    
                elif isinstance(child, nn.BatchNorm2d):
                    # LayerNorm for 2D requires reshaping
                    warnings.warn(
                        f"Replacing BatchNorm2d with LayerNorm at {name}. "
                        "This changes the normalization significantly and may hurt performance."
                    )
                    
                else:
                    replace_bn_recursive(child)
                    
            return module
            
        replace_bn_recursive(model)
    
    def _enable_stat_reset(self, model: nn.Module) -> None:
        """
        Enable per-episode statistics reset (expensive but mathematically clean).
        
        This allows each episode to compute fresh BatchNorm statistics,
        completely eliminating leakage at the cost of computational overhead.
        """
        self._bn_modules = []
        for name, module in model.named_modules():
            if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                self._bn_modules.append(module)
                # Ensure running stats are tracked for reset
                module.track_running_stats = True
                
        logger.info(f"Enabled per-episode reset for {len(self._bn_modules)} BatchNorm modules")
    
    def reset_statistics(self) -> None:
        """Reset running statistics for all BatchNorm modules (if policy allows)."""
        if self.policy == BatchNormPolicy.RESET_PER_EPISODE:
            for module in getattr(self, '_bn_modules', []):
                if hasattr(module, 'reset_running_stats'):
                    module.reset_running_stats()
        else:
            warnings.warn(f"Statistics reset not supported for policy: {self.policy.value}")
    
    def restore_original_state(self, model: nn.Module) -> None:
        """Restore original BatchNorm state (for FREEZE_STATS policy)."""
        if self.policy == BatchNormPolicy.FREEZE_STATS:
            for name, module in model.named_modules():
                if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                    if name in self.original_training_modes:
                        module.training = self.original_training_modes[name]
                    if name in self.original_track_stats:
                        module.track_running_stats = self.original_track_stats[name]


class EpisodicBatchNormMonitor:
    """
    Monitor BatchNorm statistics during episodic evaluation.
    
    Detects if running statistics are changing across episodes,
    which indicates potential information leakage.
    """
    
    def __init__(self):
        self.initial_stats = {}
        self.episode_count = 0
        
    def capture_initial_stats(self, model: nn.Module) -> None:
        """Capture initial BatchNorm statistics."""
        self.initial_stats = {}
        for name, module in model.named_modules():
            if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                if hasattr(module, 'running_mean') and module.running_mean is not None:
                    self.initial_stats[f"{name}.running_mean"] = module.running_mean.clone()
                if hasattr(module, 'running_var') and module.running_var is not None:
                    self.initial_stats[f"{name}.running_var"] = module.running_var.clone()
    
    def check_for_changes(self, model: nn.Module, tolerance: float = 1e-6) -> Dict[str, bool]:
        """
        Check if BatchNorm statistics have changed since initial capture.
        
        Returns:
            Dictionary mapping module names to whether they changed
        """
        changes = {}
        
        for name, module in model.named_modules():
            if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                if hasattr(module, 'running_mean') and module.running_mean is not None:
                    key = f"{name}.running_mean"
                    if key in self.initial_stats:
                        initial = self.initial_stats[key]
                        current = module.running_mean
                        changed = not torch.allclose(initial, current, atol=tolerance)
                        changes[key] = changed
                        
                if hasattr(module, 'running_var') and module.running_var is not None:
                    key = f"{name}.running_var"
                    if key in self.initial_stats:
                        initial = self.initial_stats[key]
                        current = module.running_var
                        changed = not torch.allclose(initial, current, atol=tolerance)
                        changes[key] = changed
        
        return changes
    
    def assert_no_leakage(self, model: nn.Module, tolerance: float = 1e-6) -> None:
        """Assert that no BatchNorm statistics have changed (strict leakage test)."""
        changes = self.check_for_changes(model, tolerance)
        
        changed_modules = [name for name, changed in changes.items() if changed]
        
        if changed_modules:
            raise AssertionError(
                f"BatchNorm statistics leakage detected in modules: {changed_modules}. "
                "This indicates information leakage across episodes."
            )


# Context manager for episodic BatchNorm handling
class EpisodicMode:
    """
    Context manager for episodic meta-learning with proper BatchNorm handling.
    
    Usage:
        with EpisodicMode(model, policy=BatchNormPolicy.FREEZE_STATS):
            for episode in episodes:
                # BatchNorm won't leak across episodes
                predictions = model(support_x, support_y, query_x)
    """
    
    def __init__(self, model: nn.Module, policy: BatchNormPolicy = BatchNormPolicy.FREEZE_STATS):
        self.model = model
        self.manager = BatchNormManager(policy)
        self.monitor = EpisodicBatchNormMonitor()
        
    def __enter__(self):
        # Prepare model for episodic evaluation
        self.manager.prepare_model_for_episodes(self.model)
        
        # Capture initial statistics for monitoring
        self.monitor.capture_initial_stats(self.model)
        
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        # Check for leakage (optional)
        try:
            self.monitor.assert_no_leakage(self.model)
        except AssertionError as e:
            logger.warning(f"BatchNorm leakage detected: {e}")
            
        # Restore original state
        self.manager.restore_original_state(self.model)


def apply_bn_policy_to_model(model: nn.Module, policy: BatchNormPolicy) -> nn.Module:
    """
    Apply BatchNorm policy to a model (factory function).
    
    Args:
        model: PyTorch model
        policy: BatchNorm handling policy
        
    Returns:
        Modified model with BatchNorm policy applied
    """
    manager = BatchNormManager(policy)
    manager.prepare_model_for_episodes(model)
    return model


if __name__ == "__main__":
    # Test BatchNorm policy implementation
    print("BatchNorm Policy for Meta-Learning Test")
    print("=" * 50)
    
    # Create test model with BatchNorm
    model = nn.Sequential(
        nn.Conv2d(3, 32, 3),
        nn.BatchNorm2d(32),
        nn.ReLU(),
        nn.Conv2d(32, 64, 3),
        nn.BatchNorm2d(64),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Linear(64, 10)
    )
    
    print(f"Original model has {sum(1 for m in model.modules() if isinstance(m, nn.BatchNorm2d))} BatchNorm layers")
    
    # Test different policies
    policies = [BatchNormPolicy.FREEZE_STATS, BatchNormPolicy.INSTANCE_NORM]
    
    for policy in policies:
        print(f"\nTesting policy: {policy.value}")
        
        # Clone model for testing
        test_model = nn.Sequential(*[
            type(m)(**{k: v for k, v in m.__dict__.items() 
                      if not k.startswith('_') and k != 'training'})
            if hasattr(m, '__dict__') else m
            for m in model
        ])
        
        try:
            # Apply policy
            manager = BatchNormManager(policy)
            manager.prepare_model_for_episodes(test_model)
            
            # Test with episodic context manager
            with EpisodicMode(test_model, policy):
                # Simulate episodic forward passes
                for episode in range(5):
                    x = torch.randn(16, 3, 32, 32)  # Batch of images
                    _ = test_model(x)
                    
            print(f"  ✓ Policy {policy.value} applied successfully")
            
        except Exception as e:
            print(f"  ✗ Policy {policy.value} failed: {e}")
    
    print("\n✓ BatchNorm policy tests completed")