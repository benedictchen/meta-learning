"""
Tests for mathematical utilities in meta_learning.core.math_utils
"""
import torch
import pytest
import numpy as np
from meta_learning.core.math_utils import (
    pairwise_sqeuclidean, cosine_logits, batched_prototype_computation,
    adaptive_temperature_scaling, numerical_stability_monitor, _eps_like
)


class TestPairwiseDistances:
    """Test pairwise distance computations."""
    
    def test_pairwise_sqeuclidean_basic(self):
        """Test basic squared Euclidean distance computation."""
        a = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        b = torch.tensor([[0.0, 0.0], [1.0, 1.0]])
        
        distances = pairwise_sqeuclidean(a, b)
        
        assert distances.shape == (2, 2)
        # Distance from [1,0] to [0,0] should be 1
        assert torch.allclose(distances[0, 0], torch.tensor(1.0))
        # Distance from [0,1] to [1,1] should be 1  
        assert torch.allclose(distances[1, 1], torch.tensor(1.0))
    
    def test_pairwise_sqeuclidean_numerical_stability(self):
        """Test numerical stability with identical points."""
        a = torch.tensor([[1.0, 2.0, 3.0]])
        b = a.clone()  # Identical points
        
        distances = pairwise_sqeuclidean(a, b)
        
        # Should be exactly 0, not small negative due to floating point
        assert torch.all(distances >= 0)
        assert torch.allclose(distances, torch.zeros_like(distances))
    
    def test_cosine_logits_basic(self):
        """Test cosine similarity logit computation."""
        a = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        b = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        
        logits = cosine_logits(a, b, tau=1.0)
        
        assert logits.shape == (2, 2)
        # Identical vectors should have cosine similarity of 1
        assert torch.allclose(logits[0, 0], torch.tensor(1.0))
        assert torch.allclose(logits[1, 1], torch.tensor(1.0))
        # Orthogonal vectors should have cosine similarity of 0
        assert torch.allclose(logits[0, 1], torch.tensor(0.0))
    
    def test_cosine_logits_temperature(self):
        """Test temperature scaling in cosine logits."""
        a = torch.tensor([[1.0, 0.0]])
        b = torch.tensor([[1.0, 0.0]])
        
        logits_low = cosine_logits(a, b, tau=1.0)
        logits_high = cosine_logits(a, b, tau=10.0)
        
        # Higher temperature should reduce logit magnitude
        assert torch.all(torch.abs(logits_high) < torch.abs(logits_low))
    
    def test_cosine_logits_zero_norm_handling(self):
        """Test handling of zero-norm vectors."""
        a = torch.tensor([[0.0, 0.0]])  # Zero vector
        b = torch.tensor([[1.0, 0.0]])
        
        # Should not crash due to division by zero
        logits = cosine_logits(a, b, tau=1.0)
        assert not torch.any(torch.isnan(logits))


class TestPrototypeComputation:
    """Test batched prototype computation."""
    
    def test_batched_prototype_basic(self):
        """Test basic prototype computation."""
        # Create simple data: class 0 at [0,0], class 1 at [1,1]
        support_x = torch.tensor([[0.0, 0.0], [0.1, -0.1], [1.0, 1.0], [0.9, 1.1]])
        support_y = torch.tensor([0, 0, 1, 1])
        
        prototypes = batched_prototype_computation(support_x, support_y)
        
        assert prototypes.shape == (2, 2)  # 2 classes, 2 features
        # Class 0 prototype should be near [0.05, -0.05]
        assert torch.allclose(prototypes[0], torch.tensor([0.05, -0.05]), atol=1e-4)
        # Class 1 prototype should be near [0.95, 1.05]
        assert torch.allclose(prototypes[1], torch.tensor([0.95, 1.05]), atol=1e-4)
    
    def test_batched_prototype_non_contiguous_labels(self):
        """Test with non-contiguous label values."""
        support_x = torch.tensor([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
        support_y = torch.tensor([10, 20, 10])  # Non-contiguous labels
        
        prototypes = batched_prototype_computation(support_x, support_y)
        
        assert prototypes.shape == (2, 2)  # 2 unique classes
        # Should handle label remapping correctly
        assert not torch.any(torch.isnan(prototypes))
    
    def test_batched_prototype_single_sample_per_class(self):
        """Test with single sample per class."""
        support_x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        support_y = torch.tensor([0, 1])
        
        prototypes = batched_prototype_computation(support_x, support_y)
        
        # Prototypes should equal the single samples
        assert torch.allclose(prototypes[0], torch.tensor([1.0, 2.0]))
        assert torch.allclose(prototypes[1], torch.tensor([3.0, 4.0]))


class TestAdaptiveTemperatureScaling:
    """Test adaptive temperature scaling."""
    
    def test_adaptive_temperature_basic(self):
        """Test basic temperature adaptation."""
        # Create logits with some spread
        logits = torch.tensor([[2.0, 1.0, 0.5, 0.0, -0.5]])
        
        scaled_logits, temperature = adaptive_temperature_scaling(logits)
        
        assert scaled_logits.shape == logits.shape
        assert isinstance(temperature, float)
        assert temperature > 0
    
    def test_adaptive_temperature_target_entropy(self):
        """Test temperature scaling with specific target entropy."""
        logits = torch.tensor([[10.0, 0.0, 0.0]])  # Very confident
        target_entropy = 0.5
        
        scaled_logits, temperature = adaptive_temperature_scaling(logits, target_entropy)
        
        # Should scale down the confident prediction
        assert temperature > 1.0  # Should increase temperature to reduce confidence
        assert torch.max(scaled_logits) < torch.max(logits)
    
    def test_adaptive_temperature_empty_input(self):
        """Test handling of empty input."""
        logits = torch.empty(0, 5)
        
        scaled_logits, temperature = adaptive_temperature_scaling(logits)
        
        assert scaled_logits.shape == logits.shape
        assert temperature == 1.0


class TestNumericalStabilityMonitor:
    """Test numerical stability monitoring."""
    
    def test_numerical_monitor_normal_tensor(self):
        """Test monitoring of normal tensor."""
        tensor = torch.randn(10, 10)
        
        metrics = numerical_stability_monitor(tensor, "test_operation")
        
        assert metrics["operation"] == "test_operation"
        assert metrics["has_nan"] is False
        assert metrics["has_inf"] is False
        assert "dynamic_range" in metrics
        assert isinstance(metrics["dynamic_range"], float)
    
    def test_numerical_monitor_problematic_tensor(self):
        """Test monitoring of tensor with numerical issues."""
        tensor = torch.tensor([[1.0, float('nan')], [float('inf'), 2.0]])
        
        metrics = numerical_stability_monitor(tensor, "problematic_operation")
        
        assert metrics["has_nan"] is True
        assert metrics["has_inf"] is True
    
    def test_numerical_monitor_matrix_condition(self):
        """Test condition number computation for matrices."""
        # Create a well-conditioned matrix
        tensor = torch.eye(3)
        
        metrics = numerical_stability_monitor(tensor, "matrix_test")
        
        assert "condition_number" in metrics
        # Identity matrix should be well-conditioned (condition number ≈ 1)
        assert abs(metrics["condition_number"] - 1.0) < 0.1
    
    def test_numerical_monitor_with_gradients(self):
        """Test monitoring tensor with gradients."""
        tensor = torch.randn(5, 5, requires_grad=True)
        # Create some dummy computation to generate gradients
        loss = tensor.sum()
        loss.backward()
        
        metrics = numerical_stability_monitor(tensor, "gradient_test")
        
        assert "gradient_norm" in metrics
        assert isinstance(metrics["gradient_norm"], float)
        assert metrics["gradient_has_nan"] is False
    
    def test_numerical_monitor_empty_tensor(self):
        """Test handling of empty tensor."""
        tensor = torch.empty(0)
        
        metrics = numerical_stability_monitor(tensor, "empty_test")
        
        assert "warning" in metrics
        assert metrics["warning"] == "empty_tensor"


class TestUtilityFunctions:
    """Test utility functions."""
    
    def test_eps_like(self):
        """Test epsilon tensor creation."""
        tensor = torch.randn(3, 3, dtype=torch.float32)
        eps = _eps_like(tensor, 1e-6)
        
        assert eps.dtype == tensor.dtype
        assert eps.device == tensor.device
        assert abs(eps.item() - 1e-6) < 1e-12  # Allow for floating point precision
    
    def test_eps_like_different_devices(self):
        """Test epsilon creation with different devices."""
        if torch.cuda.is_available():
            tensor_cuda = torch.randn(2, 2, device='cuda')
            eps_cuda = _eps_like(tensor_cuda)
            assert eps_cuda.device.type == 'cuda'
        
        tensor_cpu = torch.randn(2, 2, device='cpu')
        eps_cpu = _eps_like(tensor_cpu)
        assert eps_cpu.device.type == 'cpu'


if __name__ == "__main__":
    pytest.main([__file__])