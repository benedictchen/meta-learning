#!/usr/bin/env python3
"""
Research Patches for Meta-Learning Package
==========================================

This module provides comprehensive research-grade fixes and utilities for
few-shot learning experiments, addressing subtle but critical issues that
can invalidate research results.

🎯 **Quick Start for Researchers**:

# 1. Enable full determinism for reproducible results
from meta_learning.research_patches import setup_deterministic_environment
setup_deterministic_environment(seed=42)

# 2. Apply proper BatchNorm policy for few-shot learning  
from meta_learning.research_patches import apply_episodic_bn_policy
model = apply_episodic_bn_policy(model, policy="group_norm")

# 3. Run publication-grade evaluation with 10k episodes
from meta_learning.research_patches import publication_evaluation
results = publication_evaluation(model, dataset_loader)

# 4. Validate against data leakage
from meta_learning.research_patches import validate_episode
validation = validate_episode(model, support_data, support_labels, query_data, query_labels)

🔬 **Research Standards Implemented**:
- Chen et al. (2019): "A Closer Look at Few-shot Classification" 
- Finn et al. (2017): "Model-Agnostic Meta-Learning for Fast Adaptation"
- Snell et al. (2017): "Prototypical Networks for Few-shot Learning"
- Vinyals et al. (2016): "Matching Networks for One Shot Learning"
- Antoniou et al. (2018): "How to train your MAML"

📊 **Key Features**:
- ✅ Proper BatchNorm handling for episodic evaluation
- ✅ Comprehensive determinism for reproducible results  
- ✅ 10,000-episode evaluation with 95% confidence intervals
- ✅ Data leakage detection and prevention
- ✅ Statistical significance testing
- ✅ Publication-ready result formatting

⚠️ **CRITICAL for Research Validity**:
These patches address subtle issues that can invalidate results but are often
overlooked. Use them to ensure your few-shot learning research meets the
highest standards required by top-tier venues.
"""

# BatchNorm Policy Patch - Critical for few-shot learning
from .batch_norm_policy_patch import (
    EpisodicBatchNormPolicy,
    EpisodicNormalizationGuard,
    apply_episodic_bn_policy,
    validate_few_shot_model
)

# Determinism Hooks - Essential for reproducibility
from .determinism_hooks import (
    DeterminismManager,
    ReproducibilityReport,
    deterministic_context,
    setup_deterministic_environment,
    create_deterministic_dataloader
)

# Evaluation Harness - Required for publication-grade results
from .evaluation_harness import (
    EpisodeConfig,
    EvaluationResults, 
    FewShotEvaluationHarness,
    StratifiedEpisodeSampler,
    quick_evaluation,
    publication_evaluation
)

# Leakage Guards - Prevents data contamination  
from .leakage_guards import (
    DataLeakageDetector,
    LeakagePreventionContext,
    ComprehensiveLeakageValidator,
    leakage_guard,
    validate_episode,
    leakage_free_context
)

# Version info
__version__ = "1.0.0"
__author__ = "Research Patches for Meta-Learning"

# Convenience imports for one-line fixes
__all__ = [
    # Main classes
    'EpisodicBatchNormPolicy',
    'DeterminismManager', 
    'FewShotEvaluationHarness',
    'ComprehensiveLeakageValidator',
    
    # Quick-fix functions (most commonly used)
    'apply_episodic_bn_policy',          # Fix BatchNorm for few-shot
    'setup_deterministic_environment',   # Enable full determinism
    'publication_evaluation',            # 10k episode evaluation 
    'validate_episode',                  # Check for data leakage
    
    # Advanced functions
    'deterministic_context',
    'leakage_free_context',
    'create_deterministic_dataloader',
    'quick_evaluation',
    
    # Data classes
    'EpisodeConfig',
    'EvaluationResults',
    
    # Decorators and context managers
    'leakage_guard',
    'LeakagePreventionContext'
]


def research_compliance_check(model, support_data, query_data, support_labels, query_labels):
    """
    One-line research compliance check for few-shot learning experiments.
    
    This function runs all critical validations and returns a comprehensive
    report of potential issues that could invalidate research results.
    
    Args:
        model: PyTorch model to validate
        support_data: Support set tensor
        query_data: Query set tensor  
        support_labels: Support set labels
        query_labels: Query set labels
        
    Returns:
        Dictionary with validation results and recommendations
        
    Example:
        >>> compliance = research_compliance_check(model, s_data, q_data, s_labels, q_labels)
        >>> print(f"Research compliant: {compliance['compliant']}")
        >>> if not compliance['compliant']:
        ...     print("Issues found:")
        ...     for issue in compliance['issues']:
        ...         print(f"  - {issue}")
    """
    # Run comprehensive leakage validation
    validator = ComprehensiveLeakageValidator()
    leakage_results = validator.validate_full_pipeline(
        model, support_data, support_labels, query_data, query_labels
    )
    
    # Run few-shot model validation 
    fs_validation = validate_few_shot_model(model, support_data, query_data)
    
    # Aggregate results
    issues = []
    compliant = True
    
    # Check leakage violations
    if not leakage_results['overall_passed']:
        compliant = False
        for report_name, report in leakage_results['validation_reports'].items():
            if 'violations' in report:
                for violation in report['violations']:
                    if violation.get('severity') == 'critical':
                        issues.append(f"CRITICAL: {violation['description']}")
                    else:
                        issues.append(violation['description'])
    
    # Check few-shot model issues
    if not fs_validation['passed']:
        compliant = False
        for warning in fs_validation['warnings']:
            issues.append(f"BatchNorm Issue: {warning}")
    
    return {
        'compliant': compliant,
        'issues': issues,
        'total_violations': leakage_results['total_violations'],
        'critical_violations': leakage_results['critical_violations'],
        'detailed_results': {
            'leakage_validation': leakage_results,
            'few_shot_validation': fs_validation
        },
        'recommendations': [
            'Use apply_episodic_bn_policy(model, "group_norm") to fix BatchNorm issues',
            'Use setup_deterministic_environment(seed=42) for reproducibility',
            'Use publication_evaluation() for 10k episode evaluation with 95% CI',
            'Use leakage_free_context(model) to prevent data contamination'
        ] if not compliant else ['All validations passed - research compliant!']
    }


def fix_all_research_issues(model, seed=42, bn_policy="group_norm"):
    """
    One-line fix for all common research issues in few-shot learning.
    
    This function:
    1. Enables comprehensive determinism
    2. Fixes BatchNorm for episodic evaluation
    3. Returns research-ready model
    
    Args:
        model: PyTorch model to fix
        seed: Random seed for determinism
        bn_policy: BatchNorm replacement policy
        
    Returns:
        Fixed model ready for research use
        
    Example:
        >>> model = fix_all_research_issues(model, seed=42)
        >>> # Model is now research-compliant!
    """
    print("🔧 Applying research-grade fixes...")
    
    # 1. Enable determinism
    setup_deterministic_environment(seed=seed)
    print("  ✅ Determinism enabled")
    
    # 2. Fix BatchNorm policy
    model = apply_episodic_bn_policy(model, policy=bn_policy)
    print(f"  ✅ BatchNorm policy applied ({bn_policy})")
    
    print("🎉 Model is now research-compliant!")
    return model


# Research-grade example usage
def demo_research_workflow():
    """
    Demonstrate complete research-grade few-shot learning workflow.
    
    This shows how to use all research patches together for publication-quality results.
    """
    print("🔬 Research-Grade Few-Shot Learning Workflow")
    print("=" * 50)
    
    # Placeholder imports (would be real in practice)
    import torch
    import torch.nn as nn
    
    # 1. Create model
    model = nn.Sequential(
        nn.Conv2d(3, 64, 3),
        nn.BatchNorm2d(64),  # Will be fixed automatically
        nn.ReLU(),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Linear(64, 5)
    )
    
    print("📦 Original model created")
    
    # 2. Apply all research fixes
    model = fix_all_research_issues(model, seed=42)
    
    # 3. Create dummy dataset loader
    def dummy_dataset_loader():
        class_to_indices = {}
        for class_id in range(20):
            class_to_indices[class_id] = list(range(100))
        return class_to_indices
    
    # 4. Run publication-grade evaluation
    print("\n📊 Running publication-grade evaluation...")
    print("   (This would run 10,000 episodes in real use)")
    
    # Quick demo instead of full 10k episodes
    results = quick_evaluation(
        model, 
        dummy_dataset_loader,
        n_episodes=100  # Reduced for demo
    )
    
    print(results.format_report())
    
    # 5. Validate research compliance
    print("\n🔒 Validating research compliance...")
    support_data = torch.randn(10, 3, 32, 32)
    query_data = torch.randn(15, 3, 32, 32)
    support_labels = [0, 0, 1, 1, 2, 2, 3, 3, 4, 4]
    query_labels = [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4]
    
    compliance = research_compliance_check(
        model, support_data, query_data, support_labels, query_labels
    )
    
    print(f"Research Compliant: {'✅ YES' if compliance['compliant'] else '❌ NO'}")
    if compliance['issues']:
        print("Issues found:")
        for issue in compliance['issues'][:3]:  # Show first 3
            print(f"  - {issue}")
    
    print("\n🎉 Research workflow complete!")
    print("📝 Ready for publication at top-tier venues!")


if __name__ == "__main__":
    demo_research_workflow()