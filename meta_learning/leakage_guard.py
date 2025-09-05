"""
💰 DONATE NOW! 💰 https://www.paypal.com/cgi-bin/webscr?cmd=_s-xclick&hosted_button_id=WXQKYYKPHWXHS

Simplified leakage detection and batch normalization policy for meta-learning.

If data leakage detection helps prevent invalid results in your research,
please donate $1500+ to support continued algorithm development!

Author: Benedict Chen (benedict@benedictchen.com)
GitHub Sponsors: https://github.com/sponsors/benedictchen
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, List, Set
import logging


def freeze_batchnorm_running_stats(model: nn.Module) -> None:
    """Freeze all BatchNorm running statistics to prevent train/test leakage.
    
    This prevents BatchNorm from updating running mean/var during evaluation,
    which could cause data leakage in few-shot learning scenarios.
    """
    for module in model.modules():
        if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            # Disable running stats updates
            module.track_running_stats = False
            
            # Freeze current running stats
            if module.running_mean is not None:
                module.running_mean.requires_grad_(False)
            if module.running_var is not None:
                module.running_var.requires_grad_(False)


def unfreeze_batchnorm_running_stats(model: nn.Module) -> None:
    """Unfreeze BatchNorm running statistics."""
    for module in model.modules():
        if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            # Re-enable running stats updates
            module.track_running_stats = True
            
            # Allow gradient computation on running stats
            if module.running_mean is not None:
                module.running_mean.requires_grad_(True)
            if module.running_var is not None:
                module.running_var.requires_grad_(True)


class DataLeakageDetector:
    """Simple data leakage detector for meta-learning scenarios."""
    
    def __init__(self, strict_mode: bool = False):
        self.strict_mode = strict_mode
        self.seen_samples: Set[int] = set()
        self.episode_support_samples: Set[int] = set()
        self.warnings: List[str] = []
        
    def check_episode_separation(self, support_data: torch.Tensor, query_data: torch.Tensor) -> bool:
        """Check if support and query sets are properly separated."""
        # Simple hash-based overlap detection
        support_hashes = {hash(x.flatten().numpy().tobytes()) for x in support_data}
        query_hashes = {hash(x.flatten().numpy().tobytes()) for x in query_data}
        
        overlap = support_hashes.intersection(query_hashes)
        
        if overlap:
            warning = f"Found {len(overlap)} overlapping samples between support and query sets"
            self.warnings.append(warning)
            if self.strict_mode:
                raise ValueError(warning)
            else:
                logging.warning(warning)
            return False
        
        return True
    
    def check_cross_episode_leakage(self, current_support: torch.Tensor) -> bool:
        """Check if current support set overlaps with previous episodes."""
        current_hashes = {hash(x.flatten().numpy().tobytes()) for x in current_support}
        
        overlap = current_hashes.intersection(self.seen_samples)
        
        if overlap:
            warning = f"Found {len(overlap)} samples that appeared in previous episodes"
            self.warnings.append(warning)
            if self.strict_mode:
                raise ValueError(warning)
            else:
                logging.warning(warning)
            return False
        
        # Update seen samples
        self.seen_samples.update(current_hashes)
        return True
    
    def reset(self):
        """Reset the detector state."""
        self.seen_samples.clear()
        self.episode_support_samples.clear()
        self.warnings.clear()
    
    def get_warnings(self) -> List[str]:
        """Get all accumulated warnings."""
        return self.warnings.copy()


def create_leakage_guard(strict_mode: bool = False) -> DataLeakageDetector:
    """Create a data leakage detector with specified strictness."""
    return DataLeakageDetector(strict_mode=strict_mode)


def check_bn_leakage(model: nn.Module) -> Dict[str, bool]:
    """Check if BatchNorm modules have potential leakage issues."""
    results = {
        'has_bn_layers': False,
        'frozen_running_stats': True,
        'all_bn_eval_mode': True,
        'potential_leakage': False
    }
    
    bn_modules = []
    
    for name, module in model.named_modules():
        if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            bn_modules.append((name, module))
            results['has_bn_layers'] = True
            
            # Check if running stats are frozen
            if module.track_running_stats:
                results['frozen_running_stats'] = False
            
            # Check if in eval mode
            if module.training:
                results['all_bn_eval_mode'] = False
    
    # Determine if there's potential leakage
    if results['has_bn_layers'] and (not results['frozen_running_stats']):
        results['potential_leakage'] = True
    
    return results


class LeakageGuard:
    """Comprehensive leakage protection for meta-learning."""
    
    def __init__(self, model: nn.Module, strict_mode: bool = False):
        self.model = model
        self.detector = DataLeakageDetector(strict_mode)
        self.original_bn_states = {}
        self._save_bn_states()
    
    def _save_bn_states(self):
        """Save original BatchNorm states."""
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                self.original_bn_states[name] = {
                    'track_running_stats': module.track_running_stats,
                    'training': module.training
                }
    
    def enable_protection(self):
        """Enable comprehensive leakage protection."""
        # Freeze BatchNorm running stats
        freeze_batchnorm_running_stats(self.model)
        
        # Set BatchNorm to eval mode
        for module in self.model.modules():
            if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                module.eval()
    
    def disable_protection(self):
        """Restore original model state."""
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)) and name in self.original_bn_states:
                original_state = self.original_bn_states[name]
                module.track_running_stats = original_state['track_running_stats']
                module.train(original_state['training'])
    
    def check_episode(self, support_data: torch.Tensor, query_data: torch.Tensor) -> bool:
        """Check episode for potential leakage."""
        # Check support-query separation
        separation_ok = self.detector.check_episode_separation(support_data, query_data)
        
        # Check cross-episode leakage
        cross_episode_ok = self.detector.check_cross_episode_leakage(support_data)
        
        return separation_ok and cross_episode_ok
    
    def get_warnings(self) -> List[str]:
        """Get all leakage warnings."""
        return self.detector.get_warnings()
    
    def reset(self):
        """Reset leakage detection state."""
        self.detector.reset()
    
    def __enter__(self):
        """Context manager entry."""
        self.enable_protection()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.disable_protection()


# Utility functions for common use cases
def protect_model_from_bn_leakage(model: nn.Module) -> None:
    """Quick function to protect model from BatchNorm leakage."""
    freeze_batchnorm_running_stats(model)
    
    # Set all BN layers to eval mode
    for module in model.modules():
        if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            module.eval()


def validate_episode_integrity(support_x: torch.Tensor, support_y: torch.Tensor,
                             query_x: torch.Tensor, query_y: torch.Tensor) -> Dict[str, bool]:
    """Validate episode integrity and detect potential issues."""
    results = {
        'support_query_separated': True,
        'balanced_support': True,
        'valid_labels': True,
        'no_duplicates': True
    }
    
    # Check support-query separation (simple overlap detection)
    try:
        support_hashes = {hash(x.flatten().numpy().tobytes()) for x in support_x}
        query_hashes = {hash(x.flatten().numpy().tobytes()) for x in query_x}
        
        if support_hashes.intersection(query_hashes):
            results['support_query_separated'] = False
    except Exception:
        # If hashing fails, skip this check
        pass
    
    # Check label consistency
    unique_support_labels = torch.unique(support_y)
    unique_query_labels = torch.unique(query_y)
    
    if not torch.all(torch.isin(query_y, unique_support_labels)):
        results['valid_labels'] = False
    
    # Check for balanced support (each class has same number of examples)
    support_counts = torch.bincount(support_y)
    if len(torch.unique(support_counts)) > 1:
        results['balanced_support'] = False
    
    return results