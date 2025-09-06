"""
TTCS Warning System
==================

Warning system for Test-Time Compute Scaling with deduplication and severity levels.
Extracted from large ttcs.py for better maintainability.

FIRST PUBLIC IMPLEMENTATION of Test-Time Compute Scaling!
"""

from __future__ import annotations
from typing import Dict, Set, Optional
import warnings


class TTCSWarningSystem:
    """
    Warning system for TTCS with deduplication and severity levels.
    
    Prevents spam warnings while ensuring important issues are communicated.
    """
    
    def __init__(self):
        """Initialize warning system with empty state."""
        self.issued_warnings: Set[str] = set()
        self.warning_counts: Dict[str, int] = {}
        self.suppressed_categories: Set[str] = set()
    
    def warn(self, message: str, category: str = 'general', severity: str = 'warning') -> None:
        """
        Issue warning with deduplication.
        
        Args:
            message: Warning message to display
            category: Warning category for grouping
            severity: Severity level ('info', 'warning', 'error')
            
        Raises:
            RuntimeError: If severity is 'error'
        """
        # Skip if category is suppressed
        if category in self.suppressed_categories:
            return
        
        warning_key = f"{category}:{hash(message)}"
        
        if warning_key not in self.issued_warnings:
            self.issued_warnings.add(warning_key)
            self.warning_counts[warning_key] = 1
            
            full_message = f"[TTCS {severity.upper()}] {message}"
            
            if severity == 'error':
                raise RuntimeError(full_message)
            elif severity == 'warning':
                warnings.warn(full_message, UserWarning, stacklevel=2)
            elif severity == 'info':
                # For info messages, just print (don't use warnings module)
                print(f"ℹ️  {full_message}")
        else:
            self.warning_counts[warning_key] += 1
            
            # Optionally warn about repeated issues (but less frequently)
            if self.warning_counts[warning_key] in [5, 10, 25, 50, 100]:
                summary_msg = f"[TTCS REPEATED] {message} (occurred {self.warning_counts[warning_key]} times)"
                if severity != 'error':  # Don't repeat error raises
                    warnings.warn(summary_msg, UserWarning, stacklevel=2)
    
    def suppress_category(self, category: str) -> None:
        """
        Suppress all warnings from a specific category.
        
        Args:
            category: Category to suppress
        """
        self.suppressed_categories.add(category)
    
    def unsuppress_category(self, category: str) -> None:
        """
        Re-enable warnings from a specific category.
        
        Args:
            category: Category to re-enable
        """
        self.suppressed_categories.discard(category)
    
    def get_warning_summary(self) -> Dict[str, int]:
        """
        Get summary of all issued warnings.
        
        Returns:
            Dictionary mapping warning keys to occurrence counts
        """
        return self.warning_counts.copy()
    
    def get_category_summary(self) -> Dict[str, int]:
        """
        Get summary of warnings by category.
        
        Returns:
            Dictionary mapping categories to total warning counts
        """
        category_counts = {}
        
        for warning_key, count in self.warning_counts.items():
            category = warning_key.split(':', 1)[0]
            category_counts[category] = category_counts.get(category, 0) + count
        
        return category_counts
    
    def reset(self) -> None:
        """Reset all warning state."""
        self.issued_warnings.clear()
        self.warning_counts.clear()
    
    def has_warnings(self, category: Optional[str] = None) -> bool:
        """
        Check if any warnings have been issued.
        
        Args:
            category: Optional category to check specifically
            
        Returns:
            True if warnings have been issued
        """
        if category is None:
            return len(self.warning_counts) > 0
        
        return any(key.startswith(f"{category}:") for key in self.warning_counts)
    
    def get_most_common_warnings(self, n: int = 5) -> list:
        """
        Get the most frequently occurring warnings.
        
        Args:
            n: Number of top warnings to return
            
        Returns:
            List of (warning_key, count) tuples sorted by frequency
        """
        return sorted(self.warning_counts.items(), key=lambda x: x[1], reverse=True)[:n]


# Global warning system instance for convenience
_global_warning_system = TTCSWarningSystem()


def ttcs_warn(message: str, category: str = 'general', severity: str = 'warning') -> None:
    """
    Convenience function for issuing TTCS warnings.
    
    Args:
        message: Warning message
        category: Warning category  
        severity: Severity level
    """
    _global_warning_system.warn(message, category, severity)


def suppress_ttcs_warnings(category: str) -> None:
    """
    Suppress TTCS warnings from a specific category.
    
    Args:
        category: Category to suppress
    """
    _global_warning_system.suppress_category(category)


def unsuppress_ttcs_warnings(category: str) -> None:
    """
    Re-enable TTCS warnings from a specific category.
    
    Args:
        category: Category to re-enable
    """
    _global_warning_system.unsuppress_category(category)


def get_ttcs_warning_summary() -> Dict[str, int]:
    """
    Get summary of all TTCS warnings.
    
    Returns:
        Dictionary of warning counts by category
    """
    return _global_warning_system.get_category_summary()


def reset_ttcs_warnings() -> None:
    """Reset all TTCS warning state."""
    _global_warning_system.reset()


class TTCSConfigurationValidator:
    """
    Validator for TTCS configuration parameters.
    
    Provides comprehensive validation with helpful error messages.
    """
    
    def __init__(self, warning_system: Optional[TTCSWarningSystem] = None):
        """
        Initialize validator.
        
        Args:
            warning_system: Warning system to use (defaults to global)
        """
        self.warning_system = warning_system or _global_warning_system
    
    def validate_passes(self, passes: int) -> None:
        """
        Validate number of passes parameter.
        
        Args:
            passes: Number of stochastic passes
            
        Raises:
            ValueError: If passes is invalid
        """
        if not isinstance(passes, int):
            raise ValueError(f"passes must be an integer, got {type(passes)}")
        
        if passes < 1:
            raise ValueError(f"passes must be >= 1, got {passes}")
        
        if passes > 50:
            self.warning_system.warn(
                f"passes={passes} is very high and may be slow. Consider passes <= 20 for practical use.",
                category='configuration',
                severity='warning'
            )
        
        if passes < 5:
            self.warning_system.warn(
                f"passes={passes} may be too low for reliable uncertainty estimation. Consider passes >= 8.",
                category='configuration',
                severity='warning'
            )
    
    def validate_combine_strategy(self, combine: str) -> None:
        """
        Validate combination strategy parameter.
        
        Args:
            combine: Combination strategy
            
        Raises:
            ValueError: If combine strategy is invalid
        """
        valid_strategies = ['mean_prob', 'mean_logit']
        
        if combine not in valid_strategies:
            raise ValueError(f"combine must be one of {valid_strategies}, got '{combine}'")
    
    def validate_model_compatibility(self, encoder, head) -> None:
        """
        Validate model compatibility with TTCS.
        
        Args:
            encoder: Feature encoder model
            head: Classification head
        """
        import torch.nn as nn
        
        # Check encoder
        if not hasattr(encoder, '__call__'):
            raise ValueError(f"encoder must be callable, got {type(encoder)}")
        
        if not isinstance(encoder, nn.Module):
            self.warning_system.warn(
                "encoder is not a torch.nn.Module - some TTCS features may not work properly",
                category='compatibility',
                severity='warning'
            )
        
        # Check head
        if not hasattr(head, '__call__'):
            raise ValueError(f"head must be callable, got {type(head)}")
        
        # Check for dropout layers (needed for MC-Dropout)
        if isinstance(encoder, nn.Module) and not self._has_dropout_layers(encoder):
            if isinstance(head, nn.Module) and not self._has_dropout_layers(head):
                self.warning_system.warn(
                    "No Dropout layers found in encoder or head. "
                    "MC-Dropout will have limited effect. Consider adding Dropout layers.",
                    category='mc_dropout',
                    severity='info'
                )
    
    def validate_episode_data(self, episode) -> None:
        """
        Validate episode data structure.
        
        Args:
            episode: Episode data object
            
        Raises:
            ValueError: If episode data is invalid
        """
        required_attrs = ['support_x', 'support_y', 'query_x']
        
        for attr in required_attrs:
            if not hasattr(episode, attr):
                raise ValueError(f"episode must have '{attr}' attribute")
        
        # Check tensor compatibility
        try:
            import torch
            
            if not isinstance(episode.support_x, torch.Tensor):
                raise ValueError(f"episode.support_x must be a torch.Tensor, got {type(episode.support_x)}")
            
            if not isinstance(episode.support_y, torch.Tensor):
                raise ValueError(f"episode.support_y must be a torch.Tensor, got {type(episode.support_y)}")
            
            if not isinstance(episode.query_x, torch.Tensor):
                raise ValueError(f"episode.query_x must be a torch.Tensor, got {type(episode.query_x)}")
            
            # Check dimensionality
            if episode.support_x.dim() < 2:
                raise ValueError(f"episode.support_x must have at least 2 dimensions, got {episode.support_x.dim()}")
            
            if episode.query_x.dim() < 2:
                raise ValueError(f"episode.query_x must have at least 2 dimensions, got {episode.query_x.dim()}")
            
            # Check support/query compatibility
            if episode.support_x.shape[1:] != episode.query_x.shape[1:]:
                raise ValueError(
                    f"support_x and query_x must have same feature dimensions. "
                    f"Got support_x: {episode.support_x.shape[1:]} vs query_x: {episode.query_x.shape[1:]}"
                )
                
        except ImportError:
            # PyTorch not available, skip tensor-specific validation
            pass
    
    def _has_dropout_layers(self, model) -> bool:
        """Check if model has dropout layers."""
        import torch.nn as nn
        
        if not isinstance(model, nn.Module):
            return False
        
        for module in model.modules():
            if isinstance(module, (nn.Dropout, nn.Dropout1d, nn.Dropout2d, nn.Dropout3d)):
                return True
        return False


# Convenience validator instance
ttcs_validator = TTCSConfigurationValidator()