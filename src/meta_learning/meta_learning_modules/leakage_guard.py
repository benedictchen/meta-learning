"""
Leakage Guard for Meta-Learning
===============================

Author: Benedict Chen (benedict@benedictchen.com)

Critical leakage prevention for meta-learning where normalization statistics,
feature statistics, or class statistics must NEVER leak across the train/test split.

Research Issue:
In few-shot learning, any statistics computed on training classes that influence
test class evaluation violates the fundamental few-shot assumption and creates
data leakage that invalidates results.

Common Leakage Sources:
1. Global normalization statistics computed on ALL classes (including test)
2. Feature statistics (mean/std) computed across train+test classes  
3. BatchNorm running statistics updated during test episodes
4. Class prototype statistics computed on mixed train/test data
5. Gradient statistics from optimizers (Adam moments) carrying over

References:
- Vinyals et al. (2016): "Matching Networks" - strict train/test isolation
- Snell et al. (2017): "Prototypical Networks" - per-episode statistics only
- Finn et al. (2017): "MAML" - no cross-episode parameter sharing
"""

import torch
import torch.nn as nn
from typing import Dict, Set, List, Optional, Any, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
import warnings
import logging
from contextlib import contextmanager

logger = logging.getLogger(__name__)


class LeakageType(Enum):
    """Types of leakage that can occur in meta-learning."""
    NORMALIZATION_STATS = "normalization_stats"      # Global mean/std from train+test
    BATCHNORM_RUNNING_STATS = "batchnorm_running"    # BN stats across episodes
    OPTIMIZER_MOMENTS = "optimizer_moments"          # Adam/SGD momentum from train
    CLASS_STATISTICS = "class_statistics"           # Statistics across train+test classes
    FEATURE_STATISTICS = "feature_statistics"       # Global feature stats
    GRADIENT_ACCUMULATION = "gradient_accumulation" # Gradients across episodes


@dataclass
class LeakageViolation:
    """Record of a detected leakage violation."""
    violation_type: LeakageType
    source_location: str
    description: str
    severity: str  # "critical", "high", "medium", "low"
    suggested_fix: str
    data_snapshot: Optional[Dict[str, Any]] = None


class LeakageGuard:
    """
    Guard against data leakage in meta-learning experiments.
    
    Monitors and prevents common sources of leakage that can
    invalidate few-shot learning results.
    """
    
    def __init__(self, 
                 strict_mode: bool = True,
                 track_normalization: bool = True,
                 track_batchnorm: bool = True,
                 track_optimizer: bool = True):
        self.strict_mode = strict_mode
        self.track_normalization = track_normalization
        self.track_batchnorm = track_batchnorm
        self.track_optimizer = track_optimizer
        
        # Tracking state
        self.violations: List[LeakageViolation] = []
        self.registered_stats = {}
        self.episode_boundaries = []
        self.train_classes: Optional[Set[int]] = None
        self.test_classes: Optional[Set[int]] = None
        
    def register_train_test_split(self, train_classes: List[int], test_classes: List[int]):
        """
        Register the official train/test class split.
        
        Critical for detecting cross-split leakage.
        """
        self.train_classes = set(train_classes)
        self.test_classes = set(test_classes)
        
        # Validate split integrity
        overlap = self.train_classes.intersection(self.test_classes)
        if overlap:
            self._record_violation(
                LeakageType.CLASS_STATISTICS,
                "train_test_split",
                f"Train/test classes overlap: {overlap}",
                "critical",
                "Ensure train_classes and test_classes are disjoint sets"
            )
            
        logger.info(f"Registered split: {len(train_classes)} train, {len(test_classes)} test classes")
    
    def validate_normalization_stats(self, 
                                   stats: Dict[str, torch.Tensor],
                                   data_classes: List[int],
                                   operation_name: str) -> bool:
        """
        Validate that normalization statistics don't leak across train/test split.
        
        Args:
            stats: Dictionary of normalization statistics (mean, std, etc.)
            data_classes: Classes that contributed to these statistics
            operation_name: Name of the operation for error reporting
            
        Returns:
            True if no leakage detected, False otherwise
        """
        if not self.track_normalization:
            return True
            
        if self.train_classes is None or self.test_classes is None:
            warnings.warn("Train/test split not registered. Cannot validate normalization stats.")
            return True
            
        data_class_set = set(data_classes)
        
        # Check for cross-split contamination
        has_train = bool(data_class_set.intersection(self.train_classes))
        has_test = bool(data_class_set.intersection(self.test_classes))
        
        if has_train and has_test:
            self._record_violation(
                LeakageType.NORMALIZATION_STATS,
                operation_name,
                f"Normalization stats computed on both train and test classes: "
                f"train={data_class_set.intersection(self.train_classes)}, "
                f"test={data_class_set.intersection(self.test_classes)}",
                "critical",
                "Compute separate normalization statistics for train and test splits"
            )
            return False
            
        # Store stats for monitoring
        self.registered_stats[operation_name] = {
            'stats': {k: v.clone().detach() for k, v in stats.items()},
            'classes': data_classes,
            'timestamp': torch.tensor(len(self.episode_boundaries))
        }
        
        return True
    
    def check_batchnorm_leakage(self, model: nn.Module, episode_boundary: bool = False) -> List[LeakageViolation]:
        """
        Check for BatchNorm statistics leakage across episodes.
        
        Returns list of violations found.
        """
        violations = []
        
        if not self.track_batchnorm:
            return violations
            
        if episode_boundary:
            self.episode_boundaries.append(len(self.violations))
            
        for name, module in model.named_modules():
            if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                # Check if module is in training mode but tracking stats
                if module.training and module.track_running_stats:
                    violation = LeakageViolation(
                        violation_type=LeakageType.BATCHNORM_RUNNING_STATS,
                        source_location=f"model.{name}",
                        description=f"BatchNorm module in training mode with track_running_stats=True "
                                  f"will accumulate statistics across episodes",
                        severity="high",
                        suggested_fix="Use BatchNormPolicy.FREEZE_STATS or switch to InstanceNorm"
                    )
                    violations.append(violation)
                    
        return violations
    
    def validate_episode_isolation(self,
                                 support_classes: List[int],
                                 query_classes: List[int],
                                 episode_id: str) -> bool:
        """
        Validate that episode maintains proper class isolation.
        
        Args:
            support_classes: Classes in support set
            query_classes: Classes in query set  
            episode_id: Identifier for the episode
            
        Returns:
            True if isolation maintained, False otherwise
        """
        support_set = set(support_classes)
        query_set = set(query_classes)
        
        # Check that support and query have same classes (standard few-shot)
        if support_set != query_set:
            self._record_violation(
                LeakageType.CLASS_STATISTICS,
                f"episode_{episode_id}",
                f"Support and query sets have different classes: "
                f"support={support_set}, query={query_set}",
                "medium",
                "Ensure support and query sets contain the same N classes"
            )
            return False
            
        # Check that episode classes don't span train/test split
        if self.train_classes is not None and self.test_classes is not None:
            has_train = bool(support_set.intersection(self.train_classes))
            has_test = bool(support_set.intersection(self.test_classes))
            
            if has_train and has_test:
                self._record_violation(
                    LeakageType.CLASS_STATISTICS,
                    f"episode_{episode_id}",
                    f"Episode spans train/test split: "
                    f"train_classes={support_set.intersection(self.train_classes)}, "
                    f"test_classes={support_set.intersection(self.test_classes)}",
                    "critical",
                    "Generate episodes from single split (train OR test, never both)"
                )
                return False
                
        return True
    
    def monitor_optimizer_state(self, optimizer: torch.optim.Optimizer, 
                              episode_boundary: bool = False) -> List[LeakageViolation]:
        """
        Monitor optimizer state for cross-episode leakage.
        
        Adam/AdamW momentum terms can carry information across episodes.
        """
        violations = []
        
        if not self.track_optimizer:
            return violations
            
        if episode_boundary and hasattr(optimizer, 'state') and optimizer.state:
            # Check if optimizer has accumulated state
            for param_group in optimizer.param_groups:
                for param in param_group['params']:
                    if param in optimizer.state:
                        state = optimizer.state[param]
                        if 'exp_avg' in state or 'exp_avg_sq' in state:
                            violation = LeakageViolation(
                                violation_type=LeakageType.OPTIMIZER_MOMENTS,
                                source_location="optimizer_state",
                                description=f"Optimizer has accumulated moments from previous episodes. "
                                          f"State keys: {list(state.keys())}",
                                severity="medium",
                                suggested_fix="Reset optimizer state between episodes: optimizer.state.clear()"
                            )
                            violations.append(violation)
                            break
                            
        return violations
    
    def _record_violation(self, 
                         violation_type: LeakageType,
                         source: str,
                         description: str,
                         severity: str,
                         suggested_fix: str):
        """Record a leakage violation."""
        violation = LeakageViolation(
            violation_type=violation_type,
            source_location=source,
            description=description,
            severity=severity,
            suggested_fix=suggested_fix
        )
        
        self.violations.append(violation)
        
        if self.strict_mode and severity == "critical":
            raise ValueError(f"CRITICAL LEAKAGE DETECTED: {description}")
        elif severity in ["critical", "high"]:
            logger.error(f"LEAKAGE: {description}")
        else:
            logger.warning(f"Potential leakage: {description}")
    
    def assert_no_leakage(self, max_severity: str = "medium") -> None:
        """
        Assert that no leakage violations above max_severity have occurred.
        
        Args:
            max_severity: Maximum allowed severity ("low", "medium", "high", "critical")
        """
        severity_levels = {"low": 1, "medium": 2, "high": 3, "critical": 4}
        max_level = severity_levels[max_severity]
        
        violations = [v for v in self.violations 
                     if severity_levels[v.severity] > max_level]
        
        if violations:
            violation_summary = "\n".join([
                f"- {v.violation_type.value}: {v.description} (fix: {v.suggested_fix})"
                for v in violations
            ])
            raise AssertionError(
                f"Leakage violations detected:\n{violation_summary}"
            )
    
    def get_leakage_report(self) -> Dict[str, Any]:
        """Get comprehensive leakage analysis report."""
        violations_by_type = {}
        for violation in self.violations:
            vtype = violation.violation_type.value
            if vtype not in violations_by_type:
                violations_by_type[vtype] = []
            violations_by_type[vtype].append(violation)
            
        severity_counts = {}
        for violation in self.violations:
            severity = violation.severity
            severity_counts[severity] = severity_counts.get(severity, 0) + 1
            
        return {
            'total_violations': len(self.violations),
            'violations_by_type': violations_by_type,
            'severity_counts': severity_counts,
            'train_classes': list(self.train_classes) if self.train_classes else None,
            'test_classes': list(self.test_classes) if self.test_classes else None,
            'episode_boundaries': len(self.episode_boundaries),
            'registered_stats': list(self.registered_stats.keys())
        }


@contextmanager
def leakage_guard_context(guard: LeakageGuard, 
                         train_classes: Optional[List[int]] = None,
                         test_classes: Optional[List[int]] = None):
    """
    Context manager for leakage-guarded meta-learning evaluation.
    
    Usage:
        with leakage_guard_context(guard, train_classes, test_classes):
            # All operations monitored for leakage
            results = evaluate_few_shot_model(model, episodes)
    """
    if train_classes and test_classes:
        guard.register_train_test_split(train_classes, test_classes)
        
    try:
        yield guard
    finally:
        # Final leakage check
        report = guard.get_leakage_report()
        if report['total_violations'] > 0:
            logger.warning(f"Leakage guard detected {report['total_violations']} violations")
            
            # Log critical/high violations
            critical_high = sum(1 for v in guard.violations 
                              if v.severity in ['critical', 'high'])
            if critical_high > 0:
                logger.error(f"{critical_high} critical/high severity leakage violations")


def create_safe_normalizer(train_data: torch.Tensor, 
                          train_classes: List[int],
                          guard: Optional[LeakageGuard] = None) -> Callable[[torch.Tensor], torch.Tensor]:
    """
    Create a normalizer that only uses training data statistics.
    
    This prevents test data leakage in normalization.
    
    Args:
        train_data: Training data to compute statistics from
        train_classes: Classes in training data (for leakage validation)
        guard: Optional leakage guard for monitoring
        
    Returns:
        Normalization function that applies train-only statistics
    """
    # Compute statistics only on training data
    train_mean = train_data.mean(dim=0)
    train_std = train_data.std(dim=0)
    
    # Prevent division by zero
    train_std = torch.where(train_std < 1e-8, torch.ones_like(train_std), train_std)
    
    # Validate with leakage guard
    if guard is not None:
        stats = {'mean': train_mean, 'std': train_std}
        guard.validate_normalization_stats(stats, train_classes, "safe_normalizer")
    
    def normalize_fn(data: torch.Tensor) -> torch.Tensor:
        """Apply train-only normalization statistics."""
        return (data - train_mean) / train_std
        
    return normalize_fn


if __name__ == "__main__":
    # Test leakage guard functionality
    print("Leakage Guard for Meta-Learning Test")
    print("=" * 50)
    
    # Create guard
    guard = LeakageGuard(strict_mode=False)
    
    # Test train/test split registration
    train_classes = [0, 1, 2, 3, 4]
    test_classes = [5, 6, 7, 8, 9]
    guard.register_train_test_split(train_classes, test_classes)
    
    # Test normalization validation (valid case)
    train_data = torch.randn(100, 64)
    stats = {'mean': train_data.mean(0), 'std': train_data.std(0)}
    valid = guard.validate_normalization_stats(stats, train_classes, "train_normalization")
    print(f"Train-only normalization valid: {valid}")
    
    # Test normalization validation (leakage case)
    mixed_classes = train_classes + test_classes
    valid = guard.validate_normalization_stats(stats, mixed_classes, "mixed_normalization")
    print(f"Mixed train+test normalization valid: {valid}")
    
    # Test episode isolation
    valid_episode = guard.validate_episode_isolation([0, 1, 2], [0, 1, 2], "episode_1")
    print(f"Valid episode isolation: {valid_episode}")
    
    # Test cross-split episode (should fail)
    invalid_episode = guard.validate_episode_isolation([0, 1, 5], [0, 1, 5], "episode_2")
    print(f"Cross-split episode valid: {invalid_episode}")
    
    # Test safe normalizer
    normalizer = create_safe_normalizer(train_data, train_classes, guard)
    test_data = torch.randn(50, 64)
    normalized_test = normalizer(test_data)
    print(f"Safe normalization applied to test data shape: {normalized_test.shape}")
    
    # Get final report
    report = guard.get_leakage_report()
    print(f"\nLeakage Report:")
    print(f"  Total violations: {report['total_violations']}")
    print(f"  Severity counts: {report['severity_counts']}")
    print(f"  Violations by type: {list(report['violations_by_type'].keys())}")
    
    # Test context manager
    print("\nTesting leakage guard context...")
    with leakage_guard_context(LeakageGuard(), train_classes, test_classes) as ctx_guard:
        # Simulate some operations
        ctx_guard.validate_episode_isolation([0, 1], [0, 1], "context_episode")
        
    print("✓ Leakage guard tests completed")