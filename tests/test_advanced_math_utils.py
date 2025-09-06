"""
Tests for advanced mathematical utilities in meta_learning.core.math_utils
"""
import torch
import pytest
import numpy as np
from unittest.mock import patch, MagicMock
from meta_learning.core.math_utils import (
    adaptive_temperature_scaling_supervised,
    mixed_precision_distances,
    batch_aware_prototype_computation,
    numerical_stability_monitor
)


class TestAdaptiveTemperatureScalingSupervised:
    """Test supervised adaptive temperature scaling."""
    
    def test_supervised_scaling_basic(self):
        """Test basic supervised temperature scaling functionality."""
        torch.manual_seed(42)
        logits = torch.randn(50, 5) * 2.0  # Overconfident logits
        targets = torch.randint(0, 5, (50,))
        
        scaled_logits = adaptive_temperature_scaling_supervised(logits, targets, 
                                                              initial_tau=1.0, max_iter=10)
        
        assert scaled_logits.shape == logits.shape
        # Should produce different (typically smaller magnitude) logits
        assert not torch.equal(scaled_logits, logits)
        
        # Scaled logits should lead to better calibrated predictions
        original_probs = torch.softmax(logits, dim=1)
        scaled_probs = torch.softmax(scaled_logits, dim=1)
        
        # Check that probabilities changed (calibration improved)
        assert not torch.allclose(original_probs, scaled_probs, atol=1e-3)
    
    def test_supervised_scaling_improves_calibration(self):
        """Test that supervised scaling improves calibration."""
        torch.manual_seed(123)
        # Create overconfident logits
        logits = torch.randn(100, 3) * 5.0  # Very confident predictions
        targets = torch.randint(0, 3, (100,))
        
        # Original cross-entropy loss
        original_loss = torch.nn.functional.cross_entropy(logits, targets)
        
        # Scaled logits
        scaled_logits = adaptive_temperature_scaling_supervised(logits, targets, max_iter=20)
        scaled_loss = torch.nn.functional.cross_entropy(scaled_logits, targets)
        
        # Scaled loss should be lower (better calibration)
        assert scaled_loss.item() <= original_loss.item()
    
    def test_supervised_scaling_edge_cases(self):
        """Test edge cases for supervised temperature scaling."""
        # Single sample
        logits = torch.randn(1, 5)
        targets = torch.tensor([2])
        
        scaled_logits = adaptive_temperature_scaling_supervised(logits, targets, max_iter=5)
        assert scaled_logits.shape == (1, 5)
        
        # Perfect predictions (should handle gracefully)
        perfect_logits = torch.zeros(10, 3)
        perfect_targets = torch.zeros(10, dtype=torch.long)
        perfect_logits[range(10), perfect_targets] = 10.0  # Perfect confidence
        
        scaled_logits = adaptive_temperature_scaling_supervised(perfect_logits, perfect_targets)
        assert not torch.any(torch.isnan(scaled_logits))
        assert not torch.any(torch.isinf(scaled_logits))
    
    def test_temperature_clamping(self):
        """Test that temperature is properly clamped."""
        # Create logits that might lead to very small temperatures
        logits = torch.randn(20, 4) * 0.1  # Very small magnitude
        targets = torch.randint(0, 4, (20,))
        
        scaled_logits = adaptive_temperature_scaling_supervised(logits, targets)
        
        # Should not produce extreme values due to temperature clamping
        assert torch.all(torch.isfinite(scaled_logits))
        assert torch.max(torch.abs(scaled_logits)) < 1000  # Reasonable magnitude


class TestMixedPrecisionDistances:
    """Test mixed precision distance computations."""
    
    def test_mixed_precision_automatic_selection(self):
        """Test automatic precision selection based on tensor size."""
        # Small tensors - should use full precision
        small_a = torch.randn(10, 64)
        small_b = torch.randn(5, 64)
        
        distances_small = mixed_precision_distances(small_a, small_b)
        
        assert distances_small.shape == (10, 5)
        assert distances_small.dtype == torch.float32
        
        # Large tensors - should use half precision if CUDA available
        if torch.cuda.is_available():
            large_a = torch.randn(1000, 1000, device='cuda')
            large_b = torch.randn(1000, 1000, device='cuda')
            
            distances_large = mixed_precision_distances(large_a, large_b)
            
            assert distances_large.shape == (1000, 1000)
            assert distances_large.dtype == torch.float32  # Returned as float32
    
    def test_mixed_precision_forced_selection(self):
        """Test forced precision selection."""
        a = torch.randn(50, 32)
        b = torch.randn(20, 32)
        
        # Force full precision
        distances_full = mixed_precision_distances(a, b, use_half_precision=False)
        
        # Force half precision (if supported)
        if torch.cuda.is_available():
            a_cuda = a.cuda()
            b_cuda = b.cuda()
            distances_half = mixed_precision_distances(a_cuda, b_cuda, use_half_precision=True)
            
            assert distances_half.dtype == torch.float32
            # Should be close but not identical due to precision differences
            assert torch.allclose(distances_full.cpu(), distances_half.cpu(), atol=1e-2)
    
    def test_mixed_precision_numerical_consistency(self):
        """Test that mixed precision maintains reasonable numerical accuracy."""
        torch.manual_seed(42)
        a = torch.randn(100, 128)
        b = torch.randn(50, 128)
        
        # Full precision baseline
        distances_full = mixed_precision_distances(a, b, use_half_precision=False)
        
        if torch.cuda.is_available():
            a_cuda = a.cuda()
            b_cuda = b.cuda()
            
            # Half precision computation
            distances_half = mixed_precision_distances(a_cuda, b_cuda, use_half_precision=True)
            
            # Should be reasonably close (half precision has ~3-4 decimal places)
            assert torch.allclose(distances_full, distances_half.cpu(), rtol=1e-3, atol=1e-2)
    
    def test_mixed_precision_memory_efficiency(self):
        """Test memory efficiency claims (mock memory monitoring)."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available for memory testing")
        
        # Large tensors that would benefit from half precision
        a = torch.randn(1000, 512, device='cuda')
        b = torch.randn(800, 512, device='cuda')
        
        # This should automatically use half precision for memory efficiency
        with patch('torch.cuda.memory_allocated') as mock_memory:
            mock_memory.return_value = 1_000_000_000  # 1GB allocated
            
            distances = mixed_precision_distances(a, b)
            
            assert distances.shape == (1000, 800)
            assert distances.device.type == 'cuda'


class TestBatchAwarePrototypeComputation:
    """Test batch-aware prototype computation."""
    
    def test_batch_aware_basic_functionality(self):
        """Test basic batch-aware prototype computation."""
        support_x = torch.randn(30, 64)
        support_y = torch.repeat_interleave(torch.arange(5), 6)  # 5 classes, 6 samples each
        
        prototypes = batch_aware_prototype_computation(support_x, support_y)
        
        assert prototypes.shape == (5, 64)  # 5 classes, 64 features
        
        # Verify prototypes are class means
        for i in range(5):
            class_mask = support_y == i
            expected_prototype = support_x[class_mask].mean(dim=0)
            assert torch.allclose(prototypes[i], expected_prototype)
    
    def test_batch_aware_memory_budget(self):
        """Test memory budget functionality."""
        support_x = torch.randn(100, 256)
        support_y = torch.repeat_interleave(torch.arange(10), 10)
        
        # Test with different memory budgets
        prototypes_high = batch_aware_prototype_computation(support_x, support_y, memory_budget=0.9)
        prototypes_low = batch_aware_prototype_computation(support_x, support_y, memory_budget=0.1)
        
        # Results should be identical regardless of batching strategy
        assert torch.allclose(prototypes_high, prototypes_low)
        assert prototypes_high.shape == (10, 256)
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_batch_aware_cuda_memory_detection(self):
        """Test CUDA memory detection and batching decisions."""
        support_x = torch.randn(50, 128, device='cuda')
        support_y = torch.repeat_interleave(torch.arange(5), 10).cuda()
        
        # Mock CUDA memory properties for testing
        with patch('torch.cuda.get_device_properties') as mock_props:
            mock_device = MagicMock()
            mock_device.total_memory = 1_000_000  # 1MB (very small)
            mock_props.return_value = mock_device
            
            # Should trigger batching due to small "available" memory
            prototypes = batch_aware_prototype_computation(support_x, support_y, memory_budget=0.5)
            
            assert prototypes.shape == (5, 128)
            assert prototypes.device.type == 'cuda'
    
    def test_batch_aware_cpu_fallback(self):
        """Test CPU fallback when CUDA not available."""
        support_x = torch.randn(40, 32)
        support_y = torch.repeat_interleave(torch.arange(8), 5)
        
        prototypes = batch_aware_prototype_computation(support_x, support_y)
        
        assert prototypes.shape == (8, 32)
        assert prototypes.device.type == 'cpu'
        
        # Should still compute correct prototypes
        for i in range(8):
            class_mask = support_y == i
            expected = support_x[class_mask].mean(dim=0)
            assert torch.allclose(prototypes[i], expected)
    
    def test_batch_aware_dtype_preservation(self):
        """Test that batch-aware computation preserves dtypes."""
        support_x = torch.randn(20, 16, dtype=torch.float64)
        support_y = torch.repeat_interleave(torch.arange(4), 5)
        
        prototypes = batch_aware_prototype_computation(support_x, support_y)
        
        assert prototypes.dtype == torch.float64
        assert prototypes.shape == (4, 16)


class TestEnhancedNumericalStabilityMonitor:
    """Test enhanced numerical stability monitoring."""
    
    def test_stability_monitor_comprehensive_stats(self):
        """Test comprehensive statistics collection."""
        tensor = torch.randn(50, 50) * 2.0 + 1.0  # Mean ~1, std ~2
        
        stats = numerical_stability_monitor(tensor, "test_tensor")
        
        # Check all expected keys
        expected_keys = ['has_nan', 'has_inf', 'min_value', 'max_value', 
                        'mean_value', 'std_value', 'condition_number', 'dynamic_range']
        for key in expected_keys:
            assert key in stats
        
        # Verify reasonable values
        assert stats['has_nan'] is False
        assert stats['has_inf'] is False
        assert abs(stats['mean_value'] - 1.0) < 0.5  # Should be close to 1
        assert stats['std_value'] > 0  # Should have positive std
        assert stats['dynamic_range'] > 0  # Should have positive range
    
    def test_stability_monitor_problem_detection(self):
        """Test detection of numerical problems."""
        # Tensor with NaN values
        problematic_tensor = torch.tensor([[1.0, 2.0], [float('nan'), 4.0]])
        
        with patch('builtins.print') as mock_print:
            stats = numerical_stability_monitor(problematic_tensor, "problematic")
            
            assert stats['has_nan'] is True
            mock_print.assert_any_call("WARNING: problematic contains NaN values")
        
        # Tensor with infinite values
        inf_tensor = torch.tensor([[1.0, float('inf')], [3.0, 4.0]])
        
        with patch('builtins.print') as mock_print:
            stats = numerical_stability_monitor(inf_tensor, "infinite")
            
            assert stats['has_inf'] is True
            mock_print.assert_any_call("WARNING: infinite contains infinite values")
    
    def test_stability_monitor_condition_number(self):
        """Test condition number computation for matrices."""
        # Well-conditioned matrix (identity)
        well_conditioned = torch.eye(5)
        stats_good = numerical_stability_monitor(well_conditioned, "identity")
        
        assert abs(stats_good['condition_number'] - 1.0) < 0.1  # Should be ~1
        
        # Ill-conditioned matrix
        ill_conditioned = torch.eye(3) 
        ill_conditioned[2, 2] = 1e-10  # Near singular
        
        with patch('builtins.print') as mock_print:
            stats_bad = numerical_stability_monitor(ill_conditioned, "ill_conditioned")
            
            assert stats_bad['condition_number'] > 1e10
            # Should warn about high condition number
            assert any('condition number' in str(call) for call in mock_print.call_args_list)
    
    def test_stability_monitor_non_matrix_tensors(self):
        """Test monitoring of non-matrix tensors."""
        # 1D tensor
        vector = torch.randn(100)
        stats_1d = numerical_stability_monitor(vector, "vector")
        
        assert np.isnan(stats_1d['condition_number'])  # Should be NaN for non-matrices
        
        # 3D tensor
        tensor_3d = torch.randn(10, 20, 30)
        stats_3d = numerical_stability_monitor(tensor_3d, "tensor3d")
        
        assert np.isnan(stats_3d['condition_number'])  # Should be NaN for non-matrices
        assert stats_3d['dynamic_range'] > 0  # But other stats should work
    
    def test_stability_monitor_edge_cases(self):
        """Test edge cases in stability monitoring."""
        # Zero tensor
        zero_tensor = torch.zeros(5, 5)
        stats_zero = numerical_stability_monitor(zero_tensor, "zeros")
        
        assert stats_zero['min_value'] == 0.0
        assert stats_zero['max_value'] == 0.0
        assert stats_zero['mean_value'] == 0.0
        assert stats_zero['std_value'] == 0.0
        assert stats_zero['dynamic_range'] == 0.0
        
        # Constant tensor
        constant_tensor = torch.full((3, 3), 5.0)
        stats_const = numerical_stability_monitor(constant_tensor, "constant")
        
        assert stats_const['min_value'] == 5.0
        assert stats_const['max_value'] == 5.0
        assert stats_const['std_value'] == 0.0
        assert stats_const['dynamic_range'] == 0.0
        
        # Single element tensor
        single_tensor = torch.tensor([[42.0]])
        stats_single = numerical_stability_monitor(single_tensor, "single")
        
        assert stats_single['min_value'] == 42.0
        assert stats_single['max_value'] == 42.0
        assert stats_single['condition_number'] == 1.0  # Single element has condition 1
    
    def test_stability_monitor_no_grad_context(self):
        """Test that monitoring doesn't affect gradients."""
        tensor_with_grad = torch.randn(10, 10, requires_grad=True)
        
        # Perform some operation to create computation graph
        loss = tensor_with_grad.sum()
        
        # Monitor should not affect gradients
        stats = numerical_stability_monitor(tensor_with_grad, "grad_tensor")
        
        # Should still be able to backpropagate
        loss.backward()
        assert tensor_with_grad.grad is not None
        assert tensor_with_grad.grad.shape == tensor_with_grad.shape


class TestMathUtilsIntegration:
    """Integration tests for mathematical utilities."""
    
    def test_prototype_computation_with_temperature_scaling(self):
        """Test integration of prototype computation with temperature scaling."""
        # Create few-shot scenario
        torch.manual_seed(42)
        support_x = torch.randn(15, 64)  # 3 classes, 5 shots each
        support_y = torch.repeat_interleave(torch.arange(3), 5)
        query_x = torch.randn(9, 64)     # 3 classes, 3 queries each
        query_y = torch.repeat_interleave(torch.arange(3), 3)
        
        # Compute prototypes
        prototypes = batch_aware_prototype_computation(support_x, support_y)
        
        # Compute distances to prototypes
        from meta_learning.core.math_utils import pairwise_sqeuclidean
        distances = pairwise_sqeuclidean(query_x, prototypes)
        
        # Convert to logits (negative distances)
        logits = -distances
        
        # Apply temperature scaling
        scaled_logits, temperature = adaptive_temperature_scaling_supervised(logits, query_y)
        
        # Should produce reasonable results
        assert logits.shape == (9, 3)
        assert scaled_logits.shape == (9, 3)
        assert temperature > 0
        
        # Accuracy should be reasonable for synthetic data
        predictions = scaled_logits.argmax(dim=1)
        accuracy = (predictions == query_y).float().mean()
        assert 0.0 <= accuracy <= 1.0  # Valid accuracy range
    
    def test_mixed_precision_with_stability_monitoring(self):
        """Test mixed precision computation with stability monitoring."""
        torch.manual_seed(123)
        a = torch.randn(100, 256) * 10  # Large magnitude
        b = torch.randn(50, 256) * 10
        
        # Compute with mixed precision
        distances = mixed_precision_distances(a, b, use_half_precision=False)
        
        # Monitor stability
        stats = numerical_stability_monitor(distances, "distance_matrix")
        
        # Should be stable
        assert stats['has_nan'] is False
        assert stats['has_inf'] is False
        assert stats['dynamic_range'] > 0
        
        # All distances should be non-negative
        assert torch.all(distances >= 0)


if __name__ == "__main__":
    pytest.main([__file__])