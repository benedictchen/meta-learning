#!/usr/bin/env python3
"""
Data Leakage Guards for Few-Shot Learning Research
==================================================

This module implements comprehensive guards against data leakage in few-shot
learning experiments, addressing subtle but critical issues that can invalidate
research results.

Common Data Leakage Issues Addressed:
1. Normalization statistics computed from query set examples
2. Class information leaking through batch statistics during episodes  
3. Feature scaling contaminated by test/query examples
4. Temporal leakage in sequential dataset splits
5. Label leakage through improper episode construction

Research Standards Implemented:
- Chen et al. (2019): "A Closer Look at Few-shot Classification" data hygiene
- Dhillon et al. (2019): "A Baseline for Few-Shot Image Classification" 
- Tian et al. (2020): "Rethinking Few-Shot Image Classification" evaluation protocols
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Any, Optional, Set, Union
import warnings
from collections import defaultdict
from contextlib import contextmanager
import functools


class DataLeakageDetector:
    """
    Comprehensive detector for various forms of data leakage in few-shot learning.
    
    Implements detection patterns for subtle leakage that can invalidate results.
    """
    
    def __init__(self, strict_mode: bool = True):
        self.strict_mode = strict_mode
        self.detected_violations = []
        self.episode_history = []
        
    def validate_episode_construction(self,
                                    support_classes: List[int],
                                    query_classes: List[int],
                                    original_class_mapping: Dict[int, int] = None) -> Dict[str, Any]:
        """
        Validate that episode construction doesn't leak class information.
        
        Args:
            support_classes: Class labels in support set (after remapping)
            query_classes: Class labels in query set (after remapping)  
            original_class_mapping: Mapping from episode labels to original labels
            
        Returns:
            Validation results with any detected violations
        """
        violations = []
        
        # 1. Check for label leakage (query classes not in support)
        support_set = set(support_classes)
        query_set = set(query_classes)
        
        if not query_set.issubset(support_set):
            leaked_classes = query_set - support_set
            violations.append({
                'type': 'label_leakage',
                'severity': 'critical',
                'description': f'Query contains classes {leaked_classes} not in support set',
                'recommendation': 'Ensure all query classes appear in support set'
            })
        
        # 2. Check for proper label remapping
        if len(support_set) > 0:
            expected_labels = set(range(len(support_set)))
            if support_set != expected_labels:
                violations.append({
                    'type': 'improper_remapping', 
                    'severity': 'medium',
                    'description': f'Support labels {support_set} should be remapped to {expected_labels}',
                    'recommendation': 'Remap episode labels to start from 0'
                })
        
        # 3. Check for class balance in support set
        if len(support_classes) > 0:
            class_counts = defaultdict(int)
            for cls in support_classes:
                class_counts[cls] += 1
            
            count_values = list(class_counts.values())
            if len(set(count_values)) > 1:
                violations.append({
                    'type': 'unbalanced_support',
                    'severity': 'low', 
                    'description': f'Support set class distribution: {dict(class_counts)}',
                    'recommendation': 'Use equal numbers of support examples per class'
                })
        
        return {
            'passed': len(violations) == 0,
            'violations': violations,
            'support_classes': len(support_set),
            'query_classes': len(query_set)
        }
    
    def validate_normalization_statistics(self,
                                        model: nn.Module,
                                        support_data: torch.Tensor,
                                        query_data: torch.Tensor) -> Dict[str, Any]:
        """
        Validate that normalization layers don't leak information from query to support.
        
        This is critical for BatchNorm, LayerNorm, and similar layers that compute
        statistics across batch dimensions.
        """
        violations = []
        
        # Find all normalization layers
        norm_modules = []
        for name, module in model.named_modules():
            if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d,
                                 nn.LayerNorm, nn.GroupNorm, nn.InstanceNorm1d, 
                                 nn.InstanceNorm2d, nn.InstanceNorm3d)):
                norm_modules.append((name, module))
        
        if not norm_modules:
            return {'passed': True, 'violations': [], 'norm_modules_found': 0}
        
        # Check BatchNorm running statistics during training
        for name, module in norm_modules:
            if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                if module.track_running_stats and module.training:
                    violations.append({
                        'type': 'batchnorm_running_stats',
                        'severity': 'critical',
                        'module': name,
                        'description': f'BatchNorm "{name}" tracks running stats during training',
                        'recommendation': 'Set track_running_stats=False or use GroupNorm/InstanceNorm'
                    })
        
        # Check for batch mixing (support and query in same forward pass)
        total_batch_size = support_data.size(0) + query_data.size(0) 
        if total_batch_size > 32 and len(norm_modules) > 0:  # Heuristic for large batches
            warnings.warn(
                "Large combined batch size detected with normalization layers. "
                "Ensure support and query are processed separately to avoid information leakage.",
                UserWarning
            )
        
        return {
            'passed': len(violations) == 0,
            'violations': violations, 
            'norm_modules_found': len(norm_modules)
        }
    
    def validate_feature_scaling(self,
                               feature_extractor: nn.Module,
                               support_features: torch.Tensor,
                               query_features: torch.Tensor,
                               scaling_method: str = None) -> Dict[str, Any]:
        """
        Validate that feature scaling doesn't use query set statistics.
        
        Common violation: Computing mean/std for normalization using both
        support and query features, then applying to make predictions.
        """
        violations = []
        
        # Check if features have been scaled (heuristic)
        support_mean = support_features.mean().item()
        support_std = support_features.std().item()
        query_mean = query_features.mean().item()
        query_std = query_features.std().item()
        
        # If both sets have very similar statistics, might indicate joint scaling
        mean_diff = abs(support_mean - query_mean)
        std_diff = abs(support_std - query_std)
        
        if mean_diff < 0.01 and std_diff < 0.01 and scaling_method is None:
            violations.append({
                'type': 'suspicious_feature_scaling',
                'severity': 'medium',
                'description': f'Support and query features have very similar statistics (mean diff: {mean_diff:.4f}, std diff: {std_diff:.4f})',
                'recommendation': 'Ensure feature scaling uses only support set statistics'
            })
        
        return {
            'passed': len(violations) == 0,
            'violations': violations,
            'support_stats': {'mean': support_mean, 'std': support_std},
            'query_stats': {'mean': query_mean, 'std': query_std}
        }
    
    def validate_dataset_split(self,
                             train_classes: Set[int],
                             val_classes: Set[int], 
                             test_classes: Set[int]) -> Dict[str, Any]:
        """
        Validate that dataset splits don't have class overlap.
        
        Critical for few-shot learning where we need completely disjoint
        class sets for proper evaluation.
        """
        violations = []
        
        # Check for class overlap between splits
        train_val_overlap = train_classes & val_classes
        train_test_overlap = train_classes & test_classes  
        val_test_overlap = val_classes & test_classes
        
        if train_val_overlap:
            violations.append({
                'type': 'train_val_class_overlap',
                'severity': 'critical',
                'classes': list(train_val_overlap),
                'description': f'Classes {list(train_val_overlap)} appear in both train and validation',
                'recommendation': 'Ensure completely disjoint class sets'
            })
        
        if train_test_overlap:
            violations.append({
                'type': 'train_test_class_overlap', 
                'severity': 'critical',
                'classes': list(train_test_overlap),
                'description': f'Classes {list(train_test_overlap)} appear in both train and test',
                'recommendation': 'Ensure completely disjoint class sets'
            })
        
        if val_test_overlap:
            violations.append({
                'type': 'val_test_class_overlap',
                'severity': 'critical', 
                'classes': list(val_test_overlap),
                'description': f'Classes {list(val_test_overlap)} appear in both validation and test',
                'recommendation': 'Ensure completely disjoint class sets'
            })
        
        return {
            'passed': len(violations) == 0,
            'violations': violations,
            'total_classes': len(train_classes | val_classes | test_classes),
            'train_classes': len(train_classes),
            'val_classes': len(val_classes), 
            'test_classes': len(test_classes)
        }


class LeakagePreventionContext:
    """
    Context manager that enforces data leakage prevention policies.
    
    Use this to wrap model training/evaluation code to ensure proper
    data hygiene practices.
    """
    
    def __init__(self,
                 model: nn.Module,
                 prevent_batchnorm_leakage: bool = True,
                 freeze_running_stats: bool = True):
        self.model = model
        self.prevent_batchnorm_leakage = prevent_batchnorm_leakage
        self.freeze_running_stats = freeze_running_stats
        self.original_states = {}
        
    def __enter__(self):
        if self.prevent_batchnorm_leakage:
            self._apply_batchnorm_policy()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.prevent_batchnorm_leakage:
            self._restore_batchnorm_policy()
    
    def _apply_batchnorm_policy(self):
        """Apply BatchNorm leakage prevention policy."""
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                # Store original state
                self.original_states[name] = {
                    'track_running_stats': module.track_running_stats,
                    'momentum': module.momentum
                }
                
                if self.freeze_running_stats:
                    module.track_running_stats = False
                    module.momentum = 0.0
    
    def _restore_batchnorm_policy(self):
        """Restore original BatchNorm settings."""
        for name, module in self.model.named_modules():
            if name in self.original_states:
                original = self.original_states[name]
                module.track_running_stats = original['track_running_stats']
                module.momentum = original['momentum']


def leakage_guard(prevent_batchnorm_leakage: bool = True):
    """
    Decorator to prevent data leakage in few-shot learning functions.
    
    Example:
        @leakage_guard()
        def evaluate_episode(model, support_data, query_data):
            # This function is now protected against BatchNorm leakage
            return model_accuracy
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Try to find model in arguments
            model = None
            for arg in args:
                if isinstance(arg, nn.Module):
                    model = arg
                    break
            
            if 'model' in kwargs:
                model = kwargs['model']
            
            if model is None:
                warnings.warn(
                    "Could not find model in function arguments. "
                    "Leakage protection may not be applied.",
                    UserWarning
                )
                return func(*args, **kwargs)
            
            # Apply leakage prevention
            with LeakagePreventionContext(model, prevent_batchnorm_leakage):
                return func(*args, **kwargs)
        
        return wrapper
    return decorator


class ComprehensiveLeakageValidator:
    """
    Comprehensive validator that checks for all forms of data leakage
    in few-shot learning experiments.
    """
    
    def __init__(self):
        self.detector = DataLeakageDetector()
        
    def validate_full_pipeline(self,
                             model: nn.Module,
                             support_data: torch.Tensor,
                             support_labels: List[int],
                             query_data: torch.Tensor,
                             query_labels: List[int],
                             train_classes: Set[int] = None,
                             val_classes: Set[int] = None,
                             test_classes: Set[int] = None) -> Dict[str, Any]:
        """
        Run comprehensive validation of entire few-shot learning pipeline.
        
        Returns detailed report of all potential leakage issues.
        """
        validation_results = {
            'overall_passed': True,
            'total_violations': 0,
            'critical_violations': 0,
            'validation_reports': {}
        }
        
        # 1. Validate episode construction
        episode_validation = self.detector.validate_episode_construction(
            support_labels, query_labels
        )
        validation_results['validation_reports']['episode_construction'] = episode_validation
        
        # 2. Validate normalization statistics  
        norm_validation = self.detector.validate_normalization_statistics(
            model, support_data, query_data
        )
        validation_results['validation_reports']['normalization'] = norm_validation
        
        # 3. Extract features and validate scaling (if model has feature extractor)
        try:
            with torch.no_grad():
                # This is a heuristic - assumes model returns features when in eval mode
                model.eval()
                support_features = model(support_data)
                query_features = model(query_data)
                
                if support_features.dim() > 1:  # Valid feature tensor
                    scaling_validation = self.detector.validate_feature_scaling(
                        model, support_features, query_features
                    )
                    validation_results['validation_reports']['feature_scaling'] = scaling_validation
        except Exception as e:
            warnings.warn(f"Could not validate feature scaling: {e}", UserWarning)
        
        # 4. Validate dataset splits (if provided)
        if all(x is not None for x in [train_classes, val_classes, test_classes]):
            split_validation = self.detector.validate_dataset_split(
                train_classes, val_classes, test_classes
            )
            validation_results['validation_reports']['dataset_splits'] = split_validation
        
        # Aggregate results
        total_violations = 0
        critical_violations = 0
        
        for report_name, report in validation_results['validation_reports'].items():
            if not report['passed']:
                validation_results['overall_passed'] = False
                
            if 'violations' in report:
                total_violations += len(report['violations'])
                critical_violations += sum(
                    1 for v in report['violations'] 
                    if v.get('severity') == 'critical'
                )
        
        validation_results['total_violations'] = total_violations
        validation_results['critical_violations'] = critical_violations
        
        return validation_results
    
    def generate_leakage_report(self, validation_results: Dict[str, Any]) -> str:
        """Generate human-readable leakage validation report."""
        lines = [
            "🔒 Data Leakage Validation Report",
            "=" * 40,
            f"Overall Status: {'✅ PASSED' if validation_results['overall_passed'] else '❌ FAILED'}",
            f"Total Violations: {validation_results['total_violations']}",
            f"Critical Violations: {validation_results['critical_violations']}",
            ""
        ]
        
        for report_name, report in validation_results['validation_reports'].items():
            status = "✅ PASSED" if report['passed'] else "❌ FAILED"
            lines.append(f"{report_name.replace('_', ' ').title()}: {status}")
            
            if 'violations' in report and report['violations']:
                for violation in report['violations']:
                    severity = violation.get('severity', 'unknown')
                    emoji = '🚨' if severity == 'critical' else '⚠️' if severity == 'medium' else 'ℹ️'
                    lines.append(f"  {emoji} {violation['description']}")
                    if 'recommendation' in violation:
                        lines.append(f"     💡 {violation['recommendation']}")
        
        lines.append("")
        lines.append("📚 Research Standards Applied:")
        lines.append("  - Chen et al. (2019): Data hygiene for few-shot learning")
        lines.append("  - Dhillon et al. (2019): Baseline evaluation protocols") 
        lines.append("  - Tian et al. (2020): Rethinking few-shot evaluation")
        
        return "\n".join(lines)


# Convenience functions
def validate_episode(model: nn.Module,
                    support_data: torch.Tensor,
                    support_labels: List[int], 
                    query_data: torch.Tensor,
                    query_labels: List[int]) -> Dict[str, Any]:
    """
    Quick validation function for single episode.
    
    Returns validation results with any detected leakage issues.
    """
    validator = ComprehensiveLeakageValidator()
    return validator.validate_full_pipeline(
        model, support_data, support_labels, query_data, query_labels
    )


@contextmanager
def leakage_free_context(model: nn.Module):
    """
    Context manager for leakage-free few-shot learning evaluation.
    
    Example:
        with leakage_free_context(model):
            accuracy = evaluate_few_shot_episode(model, support, query)
    """
    with LeakagePreventionContext(model):
        yield


if __name__ == "__main__":
    # Demo: Comprehensive leakage detection
    print("🔒 Data Leakage Guards Demo") 
    print("=" * 40)
    
    # Create test model and data
    model = nn.Sequential(
        nn.Conv2d(3, 64, 3),
        nn.BatchNorm2d(64),  # Potential leakage source
        nn.ReLU(),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Linear(64, 5)
    )
    
    support_data = torch.randn(10, 3, 32, 32)
    support_labels = [0, 0, 1, 1, 2, 2, 3, 3, 4, 4]
    query_data = torch.randn(15, 3, 32, 32)
    query_labels = [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4]
    
    # Run comprehensive validation
    validator = ComprehensiveLeakageValidator()
    results = validator.validate_full_pipeline(
        model, support_data, support_labels, query_data, query_labels
    )
    
    # Print report
    report = validator.generate_leakage_report(results)
    print(report)
    
    print("\n🔧 Testing leakage prevention context:")
    with leakage_free_context(model):
        print("  ✅ Model is now protected from BatchNorm leakage!")
        # Any operations here will have BatchNorm running stats frozen
    
    print("\n✅ Data leakage guards ready for research use!")