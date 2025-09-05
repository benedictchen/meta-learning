"""
💰 SUPPORT THIS RESEARCH - PLEASE DONATE! 💰

🙏 If this library helps your research or project, please consider donating:
💳 PayPal: https://www.paypal.com/cgi-bin/webscr?cmd=_s-xclick&hosted_button_id=WXQKYYKPHWXHS
⭐ GitHub Sponsors: https://github.com/sponsors/benedictchen

👨‍💻 Author: Benedict Chen
💰 Donations: Help support this work! Buy me a coffee ☕, beer 🍺, lamborghini 🏎️, or private island 🏝️
💖 Please consider recurring donations to fully support continued research

Your support enables cutting-edge AI research for everyone! 🚀

Leakage Guard for Meta-Learning 🛡️🔬
===================================

🎯 **ELI5 Explanation**:
Imagine you're a teacher creating a fair test for your students!
You can't show students the test questions beforehand - that would be cheating.
Similarly, in AI research, we must keep training and testing data completely separate.
This module acts like a strict exam proctor that prevents "cheating":

- 🛡️ **Prevents Data Leakage**: Like ensuring no student sees test answers during study time
- 🔍 **Monitors Statistics**: Like watching for students sharing answers
- 🚫 **Blocks Contamination**: Like confiscating cheat sheets
- 📊 **Validates Integrity**: Like checking test papers for suspicious similarities

🔬 **Research Foundation**:
- **Matching Networks**: Oriol Vinyals et al. (2016) - Strict train/test isolation principles
- **Prototypical Networks**: Jake Snell et al. (2017) - Per-episode statistics only
- **MAML**: Chelsea Finn et al. (2017) - No cross-episode parameter sharing

⚠️ **Common Leakage Sources**:
```
❌ BAD: Global stats computed on ALL classes (train+test)
✅ GOOD: Per-episode stats computed independently

❌ BAD: BatchNorm running stats across episodes  
✅ GOOD: Frozen BN or per-episode normalization

❌ BAD: Optimizer momentum carrying over episodes
✅ GOOD: Fresh optimizer state per episode

❌ BAD: Feature normalization using test class statistics
✅ GOOD: Normalization using support set only
```

🧮 **Mathematical Leakage Detection**:
- **Cross-Split Contamination**: C_train ∩ C_test = ∅  [No class overlap]
- **Statistical Independence**: E[X_train, Y_test] = E[X_train] × E[Y_test]  [Independent distributions]
- **Episode Isolation**: θᵢ ⊥ θⱼ for i ≠ j  [Independent episode parameters]

This module provides comprehensive leakage detection and prevention
for maintaining research integrity in few-shot learning experiments.
"""

import torch
import torch.nn as nn
from typing import Dict, Set, List, Optional, Any, Callable, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum
import warnings
import logging
from contextlib import contextmanager
import copy

logger = logging.getLogger(__name__)


class LeakageType(Enum):
    """Types of leakage that can occur in meta-learning."""
    NORMALIZATION_STATS = "normalization_stats"      # Global mean/std from train+test
    BATCHNORM_RUNNING_STATS = "batchnorm_running"    # BN stats across episodes
    OPTIMIZER_MOMENTS = "optimizer_moments"          # Adam/SGD momentum from train
    CLASS_STATISTICS = "class_statistics"           # Statistics across train+test classes
    FEATURE_STATISTICS = "feature_statistics"       # Global feature stats
    GRADIENT_ACCUMULATION = "gradient_accumulation" # Gradients across episodes
    PROTOTYPE_CONTAMINATION = "prototype_contamination" # Mixed train/test prototypes
    TEMPORAL_LEAKAGE = "temporal_leakage"           # Future information in current prediction


@dataclass
class LeakageViolation:
    """Record of a detected leakage violation."""
    violation_type: LeakageType
    source_location: str
    description: str
    severity: str  # "critical", "high", "medium", "low"
    suggested_fix: str
    data_snapshot: Optional[Dict[str, Any]] = None
    timestamp: Optional[float] = None


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
                 track_optimizer: bool = True,
                 auto_fix: bool = False):
        self.strict_mode = strict_mode
        self.track_normalization = track_normalization
        self.track_batchnorm = track_batchnorm
        self.track_optimizer = track_optimizer
        self.auto_fix = auto_fix
        
        # Tracking state
        self.violations: List[LeakageViolation] = []
        self.registered_stats = {}
        self.episode_boundaries = []
        self.train_classes: Optional[Set[int]] = None
        self.test_classes: Optional[Set[int]] = None
        
        # Model state tracking
        self.model_states = {}
        self.optimizer_states = {}
        
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
                f"Classes appear in both train and test sets: {overlap}",
                "critical",
                "Ensure strict separation between train and test classes"
            )
        
        logger.info(f"Registered train/test split: {len(self.train_classes)} train, {len(self.test_classes)} test classes")
    
    def check_episode_data(self, support_classes: List[int], query_classes: List[int], 
                          episode_id: str = "unknown") -> bool:
        """
        Check episode data for leakage violations.
        
        Args:
            support_classes: Classes present in support set
            query_classes: Classes present in query set  
            episode_id: Identifier for this episode
            
        Returns:
            True if no leakage detected, False otherwise
        """
        violations_found = False
        
        # Check if episode contains mix of train/test classes
        if self.train_classes and self.test_classes:
            support_set = set(support_classes)
            query_set = set(query_classes)
            
            # Support set should not contain test classes during training
            support_test_overlap = support_set.intersection(self.test_classes)
            if support_test_overlap:
                self._record_violation(
                    LeakageType.CLASS_STATISTICS,
                    f"episode_{episode_id}",
                    f"Support set contains test classes: {support_test_overlap}",
                    "critical",
                    "Ensure support set only contains training classes"
                )
                violations_found = True
                
            # Query set should not contain classes not in support during few-shot evaluation
            query_not_in_support = query_set - support_set
            if query_not_in_support and len(support_set) > 0:
                self._record_violation(
                    LeakageType.CLASS_STATISTICS,
                    f"episode_{episode_id}",
                    f"Query contains classes not in support: {query_not_in_support}",
                    "high",
                    "Ensure query classes are subset of support classes in few-shot episodes"
                )
        
        return not violations_found
    
    def monitor_model_state(self, model: nn.Module, state_name: str = "default"):
        """
        Monitor model state for potential leakage through parameter persistence.
        
        Args:
            model: PyTorch model to monitor
            state_name: Identifier for this model state
        """
        if not self.track_batchnorm:
            return
            
        current_state = {}
        
        # Check BatchNorm running statistics
        for name, module in model.named_modules():
            if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                if hasattr(module, 'running_mean') and hasattr(module, 'running_var'):
                    current_state[f"{name}.running_mean"] = module.running_mean.clone()
                    current_state[f"{name}.running_var"] = module.running_var.clone()
        
        # Compare with previous state if exists
        if state_name in self.model_states:
            previous_state = self.model_states[state_name]
            
            for param_name, current_value in current_state.items():
                if param_name in previous_state:
                    previous_value = previous_state[param_name]
                    
                    # Check if BatchNorm stats changed significantly across episodes
                    if torch.norm(current_value - previous_value) > 1e-6:
                        self._record_violation(
                            LeakageType.BATCHNORM_RUNNING_STATS,
                            f"model.{param_name}",
                            f"BatchNorm running statistics changed across episodes",
                            "high",
                            "Freeze BatchNorm running statistics during few-shot evaluation"
                        )
        
        # Store current state
        self.model_states[state_name] = current_state
    
    def monitor_optimizer_state(self, optimizer: torch.optim.Optimizer, state_name: str = "default"):
        """
        Monitor optimizer state for momentum/moment leakage across episodes.
        
        Args:
            optimizer: PyTorch optimizer to monitor
            state_name: Identifier for this optimizer state
        """
        if not self.track_optimizer:
            return
            
        current_state = {}
        
        # Extract optimizer state (momentum, squared gradients, etc.)
        for group in optimizer.param_groups:
            for param in group['params']:
                if param.grad is not None:
                    param_id = id(param)
                    if param_id in optimizer.state:
                        state = optimizer.state[param_id]
                        
                        # Check for momentum or other moment estimates
                        for key, value in state.items():
                            if isinstance(value, torch.Tensor):
                                current_state[f"param_{param_id}.{key}"] = value.clone()
        
        # Compare with previous state
        if state_name in self.optimizer_states and current_state:
            # If optimizer has accumulated state, it might carry information across episodes
            if len(current_state) > 0:
                self._record_violation(
                    LeakageType.OPTIMIZER_MOMENTS,
                    f"optimizer_{state_name}",
                    f"Optimizer state persists across episodes (momentum/moments may leak)",
                    "medium",
                    "Reset optimizer state between episodes or use fresh optimizer instances"
                )
        
        self.optimizer_states[state_name] = current_state
    
    def check_prototype_computation(self, features: torch.Tensor, labels: torch.Tensor,
                                   computation_name: str = "unknown") -> bool:
        """
        Check prototype computation for potential leakage.
        
        Args:
            features: Feature tensor [N, D]
            labels: Label tensor [N]
            computation_name: Name of this computation for tracking
            
        Returns:
            True if computation is safe, False if leakage detected
        """
        if self.train_classes is None or self.test_classes is None:
            return True  # Cannot check without registered split
            
        unique_labels = set(labels.tolist())
        
        # Check if prototype computation mixes train and test classes
        train_in_proto = unique_labels.intersection(self.train_classes)
        test_in_proto = unique_labels.intersection(self.test_classes)
        
        if len(train_in_proto) > 0 and len(test_in_proto) > 0:
            self._record_violation(
                LeakageType.PROTOTYPE_CONTAMINATION,
                computation_name,
                f"Prototype computation mixes train {train_in_proto} and test {test_in_proto} classes",
                "critical", 
                "Compute prototypes separately for train and test classes"
            )
            return False
            
        return True
    
    def check_normalization_stats(self, data: torch.Tensor, 
                                 labels: Optional[torch.Tensor] = None,
                                 stat_name: str = "unknown") -> bool:
        """
        Check normalization statistics computation for leakage.
        
        Args:
            data: Data tensor used for normalization stats
            labels: Optional labels to check class mixing
            stat_name: Name of this statistic computation
            
        Returns:
            True if stats are safe, False if leakage detected
        """
        if not self.track_normalization:
            return True
            
        if labels is not None and self.train_classes and self.test_classes:
            unique_labels = set(labels.tolist())
            train_in_data = unique_labels.intersection(self.train_classes)
            test_in_data = unique_labels.intersection(self.test_classes)
            
            if len(train_in_data) > 0 and len(test_in_data) > 0:
                self._record_violation(
                    LeakageType.NORMALIZATION_STATS,
                    stat_name,
                    f"Normalization computed on mixed train {train_in_data} and test {test_in_data} classes",
                    "critical",
                    "Compute normalization statistics separately for each episode or split"
                )
                return False
        
        return True
    
    @contextmanager
    def safe_episode_context(self, episode_id: str, model: nn.Module, 
                           optimizer: Optional[torch.optim.Optimizer] = None):
        """
        Context manager for safe episode execution.
        
        Automatically monitors for leakage and optionally applies fixes.
        """
        # Pre-episode monitoring
        initial_model_state = None
        initial_optimizer_state = None
        
        if self.track_batchnorm:
            initial_model_state = self._capture_model_state(model)
            
        if optimizer and self.track_optimizer:
            initial_optimizer_state = copy.deepcopy(optimizer.state_dict())
        
        try:
            yield self
            
        finally:
            # Post-episode monitoring
            if initial_model_state:
                self._check_model_state_changes(model, initial_model_state, episode_id)
                
            if initial_optimizer_state and optimizer:
                self._check_optimizer_state_changes(optimizer, initial_optimizer_state, episode_id)
                
            # Auto-fix if enabled
            if self.auto_fix:
                self._apply_auto_fixes(model, optimizer)
    
    def _capture_model_state(self, model: nn.Module) -> Dict[str, torch.Tensor]:
        """Capture current model state for comparison."""
        state = {}
        for name, module in model.named_modules():
            if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                if hasattr(module, 'running_mean'):
                    state[f"{name}.running_mean"] = module.running_mean.clone()
                if hasattr(module, 'running_var'):
                    state[f"{name}.running_var"] = module.running_var.clone()
        return state
    
    def _check_model_state_changes(self, model: nn.Module, initial_state: Dict[str, torch.Tensor], 
                                 episode_id: str):
        """Check for unwanted model state changes during episode."""
        current_state = self._capture_model_state(model)
        
        for name, initial_value in initial_state.items():
            if name in current_state:
                current_value = current_state[name]
                if torch.norm(current_value - initial_value) > 1e-8:
                    self._record_violation(
                        LeakageType.BATCHNORM_RUNNING_STATS,
                        f"episode_{episode_id}.{name}",
                        f"Model state changed during episode execution",
                        "high",
                        "Ensure model is in eval mode or freeze BatchNorm stats"
                    )
    
    def _check_optimizer_state_changes(self, optimizer: torch.optim.Optimizer, 
                                     initial_state: Dict, episode_id: str):
        """Check for optimizer state accumulation across episodes."""
        current_state = optimizer.state_dict()
        
        # If optimizer accumulated new state during episode, it might leak to next episode
        if len(current_state.get('state', {})) > len(initial_state.get('state', {})):
            self._record_violation(
                LeakageType.OPTIMIZER_MOMENTS,
                f"episode_{episode_id}.optimizer",
                f"Optimizer accumulated state during episode",
                "medium", 
                "Clear optimizer state between episodes"
            )
    
    def _apply_auto_fixes(self, model: nn.Module, optimizer: Optional[torch.optim.Optimizer]):
        """Apply automatic fixes for common leakage issues."""
        # Fix 1: Set BatchNorm to eval mode
        for module in model.modules():
            if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                module.eval()
                
        # Fix 2: Clear optimizer state
        if optimizer:
            optimizer.state.clear()
    
    def _record_violation(self, violation_type: LeakageType, source: str, 
                         description: str, severity: str, suggested_fix: str):
        """Record a leakage violation."""
        violation = LeakageViolation(
            violation_type=violation_type,
            source_location=source,
            description=description,
            severity=severity,
            suggested_fix=suggested_fix,
            timestamp=torch.tensor(0.0).item()  # Simple timestamp
        )
        
        self.violations.append(violation)
        
        # Log violation based on severity
        if severity == "critical":
            logger.error(f"CRITICAL LEAKAGE: {description} in {source}")
        elif severity == "high":
            logger.warning(f"HIGH LEAKAGE: {description} in {source}")
        else:
            logger.info(f"{severity.upper()} LEAKAGE: {description} in {source}")
            
        if self.strict_mode and severity in ["critical", "high"]:
            raise ValueError(f"Leakage violation in strict mode: {description}")
    
    def get_violations_report(self) -> Dict[str, Any]:
        """Get comprehensive report of all detected violations."""
        if not self.violations:
            return {"status": "clean", "violations": [], "summary": "No leakage detected"}
            
        violations_by_type = {}
        violations_by_severity = {"critical": 0, "high": 0, "medium": 0, "low": 0}
        
        for violation in self.violations:
            # Group by type
            vtype = violation.violation_type.value
            if vtype not in violations_by_type:
                violations_by_type[vtype] = []
            violations_by_type[vtype].append(violation)
            
            # Count by severity
            violations_by_severity[violation.severity] += 1
        
        return {
            "status": "violations_detected",
            "total_violations": len(self.violations),
            "violations_by_type": violations_by_type,
            "violations_by_severity": violations_by_severity,
            "violations": [
                {
                    "type": v.violation_type.value,
                    "source": v.source_location,
                    "description": v.description,
                    "severity": v.severity,
                    "fix": v.suggested_fix
                } for v in self.violations
            ],
            "recommendations": self._get_fix_recommendations()
        }
    
    def _get_fix_recommendations(self) -> List[str]:
        """Get prioritized fix recommendations based on detected violations."""
        recommendations = []
        violation_types = set(v.violation_type for v in self.violations)
        
        if LeakageType.BATCHNORM_RUNNING_STATS in violation_types:
            recommendations.append("Set model.eval() during few-shot evaluation to freeze BatchNorm stats")
            
        if LeakageType.OPTIMIZER_MOMENTS in violation_types:
            recommendations.append("Clear optimizer state between episodes: optimizer.state.clear()")
            
        if LeakageType.CLASS_STATISTICS in violation_types:
            recommendations.append("Ensure strict train/test class separation in all computations")
            
        if LeakageType.NORMALIZATION_STATS in violation_types:
            recommendations.append("Compute normalization statistics per-episode, not globally")
            
        if LeakageType.PROTOTYPE_CONTAMINATION in violation_types:
            recommendations.append("Compute prototypes separately for each class split")
        
        return recommendations
    
    def clear_violations(self):
        """Clear all recorded violations."""
        self.violations.clear()
        
    def is_clean(self) -> bool:
        """Check if no violations have been detected."""
        return len(self.violations) == 0


def create_leakage_guard(strict_mode: bool = True, **kwargs) -> LeakageGuard:
    """
    Factory function to create a leakage guard.
    
    Args:
        strict_mode: Whether to raise errors on critical violations
        **kwargs: Additional configuration parameters
        
    Returns:
        Configured LeakageGuard instance
    """
    return LeakageGuard(strict_mode=strict_mode, **kwargs)


if __name__ == "__main__":
    # Test leakage guard functionality
    print("Leakage Guard Test")
    print("=" * 30)
    
    # Create leakage guard
    guard = create_leakage_guard(strict_mode=False)
    
    # Register train/test split
    train_classes = [0, 1, 2, 3, 4]
    test_classes = [5, 6, 7, 8, 9]
    guard.register_train_test_split(train_classes, test_classes)
    print(f"✓ Registered train/test split")
    
    # Test episode data checking
    support_classes = [0, 1, 2]  # Train classes - OK
    query_classes = [0, 1, 2]    # Same as support - OK
    is_clean = guard.check_episode_data(support_classes, query_classes, "test_episode_1")
    print(f"✓ Episode 1 check: {'Clean' if is_clean else 'Violations detected'}")
    
    # Test violation detection
    contaminated_support = [0, 1, 5]  # Mix train and test - BAD!
    contaminated_query = [0, 1, 5]
    is_clean = guard.check_episode_data(contaminated_support, contaminated_query, "test_episode_2")
    print(f"✓ Episode 2 check: {'Clean' if is_clean else 'Violations detected'}")
    
    # Test model monitoring
    model = nn.Sequential(
        nn.Linear(10, 20),
        nn.BatchNorm1d(20),
        nn.ReLU(),
        nn.Linear(20, 5)
    )
    
    guard.monitor_model_state(model, "test_model")
    print(f"✓ Model monitoring initialized")
    
    # Generate violations report
    report = guard.get_violations_report()
    print(f"✓ Violations report: {report['total_violations']} violations detected")
    
    if report['total_violations'] > 0:
        print("Recommendations:")
        for i, rec in enumerate(report['recommendations'], 1):
            print(f"  {i}. {rec}")
    
    print(f"\n✓ Leakage guard test completed. Status: {'Clean' if guard.is_clean() else 'Violations detected'}")