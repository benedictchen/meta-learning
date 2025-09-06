#!/usr/bin/env python3
"""Standalone test for advanced math utilities - bypasses package dependency issues"""

import sys
sys.path.insert(0, 'src')

import torch
import pytest
from meta_learning.core.math_utils import (
    batched_prototype_computation, 
    adaptive_temperature_scaling,
    numerical_stability_monitor,
    adaptive_temperature_scaling_supervised,
    mixed_precision_distances,
    batch_aware_prototype_computation
)

def test_adaptive_temperature_scaling_supervised():
    """Test supervised adaptive temperature scaling"""
    print("Testing adaptive_temperature_scaling_supervised...")
    
    # Create synthetic logits and targets
    logits = torch.randn(16, 3)
    targets = torch.randint(0, 3, (16,))
    
    # Test basic functionality
    scaled_logits = adaptive_temperature_scaling_supervised(logits, targets)
    
    assert scaled_logits.shape == (16, 3), f"Expected (16, 3), got {scaled_logits.shape}"
    assert torch.is_tensor(scaled_logits), "Output should be a tensor"
    print(f"✅ Scaled logits shape: {scaled_logits.shape}")

def test_mixed_precision_distances():
    """Test mixed precision distance computation"""
    print("Testing mixed_precision_distances...")
    
    query = torch.randn(8, 64, dtype=torch.float32)
    prototypes = torch.randn(5, 64, dtype=torch.float32)
    
    distances = mixed_precision_distances(query, prototypes)
    
    assert distances.shape == (8, 5), f"Expected (8, 5), got {distances.shape}"
    assert torch.all(distances >= 0), "Distances should be non-negative"
    print(f"✅ Distance shape: {distances.shape}")

def test_batch_aware_prototype_computation():
    """Test batch-aware prototype computation"""
    print("Testing batch_aware_prototype_computation...")
    
    support_x = torch.randn(100, 50)  # Large enough to trigger memory awareness
    support_y = torch.randint(0, 5, (100,))
    
    prototypes = batch_aware_prototype_computation(support_x, support_y, memory_budget=0.5)
    
    assert prototypes.shape == (5, 50), f"Expected (5, 50), got {prototypes.shape}"
    assert torch.is_tensor(prototypes), "Output should be a tensor"
    print(f"✅ Prototypes: {prototypes.shape}")

def test_numerical_stability_monitor():
    """Test numerical stability monitoring"""
    print("Testing numerical_stability_monitor...")
    
    # Create a tensor with potential numerical issues
    tensor = torch.tensor([1e-8, 1e8, float('inf'), -float('inf'), 0.0])
    
    metrics = numerical_stability_monitor(tensor, "test_tensor")
    
    assert isinstance(metrics, dict), "Should return a dictionary"
    assert 'max_abs' in metrics, "Should include max_abs metric"
    assert 'min_abs' in metrics, "Should include min_abs metric"
    print(f"✅ Stability metrics: {list(metrics.keys())}")

def test_integration_scenario():
    """Test integration of multiple utilities"""
    print("Testing integration scenario...")
    
    # Simulate few-shot learning scenario
    support_x = torch.randn(20, 32)
    support_y = torch.randint(0, 4, (20,))
    query_x = torch.randn(12, 32)
    query_y = torch.randint(0, 4, (12,))
    
    # Compute prototypes with batch awareness
    prototypes, proto_metrics = batch_aware_prototype_computation(support_x, support_y, max_batch_size=10)
    
    # Compute distances with mixed precision
    distances, dtype_used = mixed_precision_distances(query_x, prototypes)
    
    # Apply supervised temperature scaling
    logits, temperature, temp_metrics = adaptive_temperature_scaling_supervised(
        query_x, support_x, support_y, query_y, initial_temp=1.0
    )
    
    # Monitor numerical stability
    stable, stability_metrics = numerical_stability_monitor(logits)
    
    print(f"✅ Integration test completed:")
    print(f"  - Prototypes: {prototypes.shape}")
    print(f"  - Temperature: {temperature:.4f}")
    print(f"  - Stable: {stable}")
    print(f"  - Used {proto_metrics['num_batches']} batches, {dtype_used} precision")

def run_all_tests():
    """Run all tests and report results"""
    tests = [
        test_adaptive_temperature_scaling_supervised,
        test_mixed_precision_distances,
        test_batch_aware_prototype_computation,
        test_numerical_stability_monitor,
        test_integration_scenario
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"❌ {test.__name__} failed: {e}")
            failed += 1
    
    print(f"\n🎯 Test Results: {passed}/{len(tests)} passed, {failed} failed")
    return failed == 0

if __name__ == "__main__":
    print("Running standalone advanced math utilities tests...\n")
    success = run_all_tests()
    sys.exit(0 if success else 1)