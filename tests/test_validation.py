"""
Tests for input validation and error handling utilities.
"""

import pytest
import torch
import warnings
from unittest.mock import Mock

from meta_learning.validation import (
    ValidationError, ConfigurationWarning, validate_episode_tensors,
    validate_few_shot_configuration, validate_distance_metric,
    validate_temperature_parameter, validate_learning_rate,
    validate_model_parameters, validate_maml_config, validate_uncertainty_config,
    validate_optimizer_config, validate_episodic_config, validate_regularization_config,
    validate_complete_config, ValidationContext, check_episode_quality,
    warn_if_suboptimal_config
)


class TestEpisodeValidation:
    """Test episode tensor validation."""
    
    def test_valid_episode_tensors(self):
        """Test validation with valid episode tensors."""
        support_x = torch.randn(10, 64)
        support_y = torch.arange(5).repeat(2)
        query_x = torch.randn(15, 64)
        query_y = torch.arange(5).repeat(3)
        
        # Should not raise any exception
        validate_episode_tensors(support_x, support_y, query_x, query_y)
    
    def test_invalid_tensor_types(self):
        """Test validation with invalid tensor types."""
        support_x = torch.randn(10, 64)
        support_y = torch.arange(5).repeat(2)
        query_x = torch.randn(15, 64)
        query_y = [0, 1, 2, 3, 4] * 3  # List instead of tensor
        
        with pytest.raises(ValidationError, match="query_y must be a torch.Tensor"):
            validate_episode_tensors(support_x, support_y, query_x, query_y)
    
    def test_dimension_mismatch(self):
        """Test validation with dimension mismatches."""
        support_x = torch.randn(10, 64)
        support_y = torch.arange(5).repeat(2)
        query_x = torch.randn(15, 32)  # Different feature dimension
        query_y = torch.arange(5).repeat(3)
        
        with pytest.raises(ValidationError, match="feature dimensions don't match"):
            validate_episode_tensors(support_x, support_y, query_x, query_y)
    
    def test_batch_size_mismatch(self):
        """Test validation with batch size mismatches."""
        support_x = torch.randn(10, 64)
        support_y = torch.arange(8)  # Wrong size
        query_x = torch.randn(15, 64)
        query_y = torch.arange(5).repeat(3)
        
        with pytest.raises(ValidationError, match="batch sizes don't match"):
            validate_episode_tensors(support_x, support_y, query_x, query_y)
    
    def test_negative_labels(self):
        """Test validation with negative labels."""
        support_x = torch.randn(10, 64)
        support_y = torch.tensor([-1, 0, 1, 2, 3] * 2)  # Contains negative
        query_x = torch.randn(15, 64)
        query_y = torch.arange(5).repeat(3)
        
        with pytest.raises(ValidationError, match="contains negative labels"):
            validate_episode_tensors(support_x, support_y, query_x, query_y)
    
    def test_missing_classes_in_support(self):
        """Test validation when query has classes not in support."""
        support_x = torch.randn(6, 64)
        support_y = torch.tensor([0, 0, 1, 1, 2, 2])  # Classes 0, 1, 2
        query_x = torch.randn(8, 64)
        query_y = torch.tensor([0, 1, 2, 3, 0, 1, 2, 3])  # Includes class 3
        
        with pytest.raises(ValidationError, match="Query set contains classes not in support"):
            validate_episode_tensors(support_x, support_y, query_x, query_y)


class TestFewShotConfiguration:
    """Test few-shot configuration validation."""
    
    def test_valid_configuration(self):
        """Test validation with valid configuration."""
        validate_few_shot_configuration(5, 2, 10)
    
    def test_invalid_n_way(self):
        """Test validation with invalid n_way."""
        with pytest.raises(ValidationError, match="n_way must be a positive integer"):
            validate_few_shot_configuration(0, 2)
        
        with pytest.raises(ValidationError, match="n_way must be a positive integer"):
            validate_few_shot_configuration(-1, 2)
        
        with pytest.raises(ValidationError, match="n_way must be a positive integer"):
            validate_few_shot_configuration(2.5, 2)
    
    def test_invalid_k_shot(self):
        """Test validation with invalid k_shot."""
        with pytest.raises(ValidationError, match="k_shot must be a positive integer"):
            validate_few_shot_configuration(5, 0)
    
    def test_invalid_n_query(self):
        """Test validation with invalid n_query."""
        with pytest.raises(ValidationError, match="n_query must be a positive integer"):
            validate_few_shot_configuration(5, 2, 0)
    
    def test_high_n_way_warning(self):
        """Test warning for high n_way."""
        with pytest.warns(ConfigurationWarning, match="n_way=150 is unusually high"):
            validate_few_shot_configuration(150, 2)
    
    def test_high_k_shot_warning(self):
        """Test warning for high k_shot."""
        with pytest.warns(ConfigurationWarning, match="k_shot=100 is unusually high"):
            validate_few_shot_configuration(5, 100)


class TestDistanceMetricValidation:
    """Test distance metric validation."""
    
    def test_valid_distance_metrics(self):
        """Test validation with valid distance metrics."""
        for distance in ["sqeuclidean", "euclidean", "cosine", "manhattan", "dot"]:
            validate_distance_metric(distance)
    
    def test_invalid_distance_metric(self):
        """Test validation with invalid distance metric."""
        with pytest.raises(ValidationError, match="Invalid distance metric 'invalid'"):
            validate_distance_metric("invalid")


class TestTemperatureValidation:
    """Test temperature parameter validation."""
    
    def test_valid_temperature(self):
        """Test validation with valid temperature."""
        validate_temperature_parameter(1.0, "sqeuclidean")
    
    def test_invalid_temperature(self):
        """Test validation with invalid temperature."""
        with pytest.raises(ValidationError, match="tau must be a positive number"):
            validate_temperature_parameter(0, "sqeuclidean")
        
        with pytest.raises(ValidationError, match="tau must be a positive number"):
            validate_temperature_parameter(-1.0, "cosine")
    
    def test_high_cosine_temperature_warning(self):
        """Test warning for high temperature with cosine distance."""
        with pytest.warns(ConfigurationWarning, match="tau=15.0 is very high for cosine distance"):
            validate_temperature_parameter(15.0, "cosine")
    
    def test_high_euclidean_temperature_warning(self):
        """Test warning for high temperature with squared Euclidean distance."""
        with pytest.warns(ConfigurationWarning, match="tau=200.0 is very high for squared Euclidean"):
            validate_temperature_parameter(200.0, "sqeuclidean")


class TestLearningRateValidation:
    """Test learning rate validation."""
    
    def test_valid_learning_rate(self):
        """Test validation with valid learning rate."""
        validate_learning_rate(0.01)
    
    def test_invalid_learning_rate(self):
        """Test validation with invalid learning rate."""
        with pytest.raises(ValidationError, match="learning rate must be a positive number"):
            validate_learning_rate(0)
        
        with pytest.raises(ValidationError, match="learning rate must be a positive number"):
            validate_learning_rate(-0.01)
    
    def test_high_learning_rate_warning(self):
        """Test warning for high learning rate."""
        with pytest.warns(ConfigurationWarning, match="learning rate=2.0 is very high"):
            validate_learning_rate(2.0)
    
    def test_low_learning_rate_warning(self):
        """Test warning for very low learning rate."""
        with pytest.warns(ConfigurationWarning, match="learning rate=1e-07 is very low"):
            validate_learning_rate(1e-7)


class TestModelValidation:
    """Test model parameter validation."""
    
    def test_valid_model(self):
        """Test validation with valid model."""
        model = torch.nn.Linear(10, 5)
        validate_model_parameters(model)
    
    def test_invalid_model_type(self):
        """Test validation with invalid model type."""
        with pytest.raises(ValidationError, match="model must be a torch.nn.Module"):
            validate_model_parameters("not a model")
    
    def test_model_no_parameters_warning(self):
        """Test warning for model with no parameters."""
        model = torch.nn.Identity()
        with pytest.warns(ConfigurationWarning, match="Model has no parameters"):
            validate_model_parameters(model)
    
    def test_model_batchnorm_warning(self):
        """Test warning for model with BatchNorm layers."""
        model = torch.nn.Sequential(
            torch.nn.Linear(10, 5),
            torch.nn.BatchNorm1d(5)
        )
        with pytest.warns(ConfigurationWarning, match="Model contains BatchNorm layers"):
            validate_model_parameters(model)


class TestMAMLConfigValidation:
    """Test MAML configuration validation."""
    
    def test_valid_maml_config(self):
        """Test validation with valid MAML configuration."""
        validate_maml_config(0.01, 3, 0.001)
    
    def test_invalid_inner_lr(self):
        """Test validation with invalid inner learning rate."""
        with pytest.raises(ValidationError, match="inner_lr must be a positive number"):
            validate_maml_config(0, 3, 0.001)
    
    def test_invalid_inner_steps(self):
        """Test validation with invalid inner steps."""
        with pytest.raises(ValidationError, match="inner_steps must be a positive integer"):
            validate_maml_config(0.01, 0, 0.001)
    
    def test_high_inner_lr_warning(self):
        """Test warning for high inner learning rate."""
        with pytest.warns(ConfigurationWarning, match="inner_lr=0.2 is very high for MAML"):
            validate_maml_config(0.2, 3, 0.001)
    
    def test_low_inner_lr_warning(self):
        """Test warning for very low inner learning rate."""
        with pytest.warns(ConfigurationWarning, match="inner_lr=1e-06 is very low for MAML"):
            validate_maml_config(1e-6, 3, 0.001)
    
    def test_high_inner_steps_warning(self):
        """Test warning for high number of inner steps."""
        with pytest.warns(ConfigurationWarning, match="inner_steps=15 is higher than typical"):
            validate_maml_config(0.01, 15, 0.001)
    
    def test_high_lr_ratio_warning(self):
        """Test warning for high learning rate ratio."""
        with pytest.warns(ConfigurationWarning, match="outer_lr/inner_lr ratio is very high"):
            validate_maml_config(0.001, 3, 0.05)


class TestUncertaintyConfigValidation:
    """Test uncertainty estimation configuration validation."""
    
    def test_valid_uncertainty_config(self):
        """Test validation with valid uncertainty configuration."""
        validate_uncertainty_config("monte_carlo_dropout", 10, 0.1)
    
    def test_invalid_method(self):
        """Test validation with invalid uncertainty method."""
        with pytest.raises(ValidationError, match="Invalid uncertainty method 'invalid'"):
            validate_uncertainty_config("invalid", 10, 0.1)
    
    def test_invalid_n_samples(self):
        """Test validation with invalid number of samples."""
        with pytest.raises(ValidationError, match="n_samples must be a positive integer"):
            validate_uncertainty_config("monte_carlo_dropout", 0, 0.1)
    
    def test_invalid_dropout_rate(self):
        """Test validation with invalid dropout rate."""
        with pytest.raises(ValidationError, match="dropout_rate must be in"):
            validate_uncertainty_config("monte_carlo_dropout", 10, 1.5)
    
    def test_low_samples_warning(self):
        """Test warning for low number of samples."""
        with pytest.warns(ConfigurationWarning, match="n_samples=3 is low for Monte Carlo dropout"):
            validate_uncertainty_config("monte_carlo_dropout", 3, 0.1)
    
    def test_low_dropout_warning(self):
        """Test warning for very low dropout rate."""
        with pytest.warns(ConfigurationWarning, match="dropout_rate=0.01 is very low"):
            validate_uncertainty_config("monte_carlo_dropout", 10, 0.01)


class TestOptimizerConfigValidation:
    """Test optimizer configuration validation."""
    
    def test_valid_optimizer_config(self):
        """Test validation with valid optimizer configuration."""
        validate_optimizer_config("adam", 0.001, 1e-4, 0.9)
    
    def test_invalid_optimizer_type(self):
        """Test validation with invalid optimizer type."""
        with pytest.raises(ValidationError, match="Invalid optimizer type 'invalid'"):
            validate_optimizer_config("invalid", 0.001)
    
    def test_invalid_weight_decay(self):
        """Test validation with invalid weight decay."""
        with pytest.raises(ValidationError, match="weight_decay must be non-negative"):
            validate_optimizer_config("adam", 0.001, -0.1)
    
    def test_invalid_momentum(self):
        """Test validation with invalid momentum."""
        with pytest.raises(ValidationError, match="momentum must be in"):
            validate_optimizer_config("sgd", 0.001, 0.0, 1.5)
    
    def test_invalid_adam_beta(self):
        """Test validation with invalid Adam beta parameters."""
        with pytest.raises(ValidationError, match="adam beta1 must be in"):
            validate_optimizer_config("adam", 0.001, 0.0, 0.0, beta1=1.5)


class TestEpisodicConfigValidation:
    """Test episodic training configuration validation."""
    
    def test_valid_episodic_config(self):
        """Test validation with valid episodic configuration."""
        validate_episodic_config(1000, 32, 100, 100, 500)
    
    def test_invalid_n_tasks(self):
        """Test validation with invalid number of tasks."""
        with pytest.raises(ValidationError, match="n_tasks must be a positive integer"):
            validate_episodic_config(0, 32, 100, 100, 500)
    
    def test_invalid_meta_batch_size(self):
        """Test validation with invalid meta batch size."""
        with pytest.raises(ValidationError, match="meta_batch_size must be a positive integer"):
            validate_episodic_config(1000, 0, 100, 100, 500)
    
    def test_meta_batch_size_too_large(self):
        """Test validation with meta batch size larger than number of tasks."""
        with pytest.raises(ValidationError, match="meta_batch_size.*cannot exceed n_tasks"):
            validate_episodic_config(100, 200, 100, 100, 500)
    
    def test_low_val_episodes_warning(self):
        """Test warning for low number of validation episodes."""
        with pytest.warns(ConfigurationWarning, match="val_episodes=50 may be too small"):
            validate_episodic_config(1000, 32, 100, 50, 500)
    
    def test_low_test_episodes_warning(self):
        """Test warning for low number of test episodes."""
        with pytest.warns(ConfigurationWarning, match="test_episodes=100 may be too small"):
            validate_episodic_config(1000, 32, 100, 100, 100)


class TestCompleteConfigValidation:
    """Test complete configuration validation."""
    
    def test_valid_complete_config(self):
        """Test validation with valid complete configuration."""
        config = {
            "n_way": 5,
            "k_shot": 2,
            "n_query": 10,
            "distance": "sqeuclidean",
            "tau": 1.0,
            "inner_lr": 0.01,
            "inner_steps": 3,
            "outer_lr": 0.001,
            "uncertainty_method": "monte_carlo_dropout",
            "optimizer": "adam",
            "n_tasks": 1000,
            "meta_batch_size": 32
        }
        validate_complete_config(config)
    
    def test_partial_config_validation(self):
        """Test validation with partial configuration."""
        config = {
            "n_way": 5,
            "k_shot": 2,
            "distance": "cosine",
            "tau": 2.0
        }
        validate_complete_config(config)
    
    def test_invalid_complete_config(self):
        """Test validation with invalid complete configuration."""
        config = {
            "n_way": 0,  # Invalid
            "k_shot": 2,
            "distance": "invalid"  # Invalid
        }
        with pytest.raises(ValidationError):
            validate_complete_config(config)


class TestEpisodeQuality:
    """Test episode quality analysis."""
    
    def test_check_episode_quality(self):
        """Test episode quality checking."""
        support_x = torch.randn(10, 64)
        support_y = torch.arange(5).repeat(2)
        query_x = torch.randn(15, 64)
        query_y = torch.arange(5).repeat(3)
        
        metrics = check_episode_quality(support_x, support_y, query_x, query_y)
        
        assert "support_class_balance" in metrics
        assert "query_class_balance" in metrics
        assert "support_is_balanced" in metrics
        assert "query_is_balanced" in metrics
        assert "n_way" in metrics
        assert "k_shot" in metrics
        assert "support_has_nan" in metrics
        assert "support_has_inf" in metrics
        assert "query_has_nan" in metrics
        assert "query_has_inf" in metrics
        
        assert metrics["n_way"] == 5
        assert metrics["k_shot"] == 2
        assert metrics["support_is_balanced"] == True
        assert metrics["query_is_balanced"] == True


class TestValidationContext:
    """Test validation context manager."""
    
    def test_validation_context_warnings_disabled(self):
        """Test validation context with warnings disabled."""
        with ValidationContext(warnings_enabled=False):
            # This should not emit a warning
            validate_few_shot_configuration(150, 2)
    
    def test_validation_context_warnings_enabled(self):
        """Test validation context with warnings enabled."""
        with ValidationContext(warnings_enabled=True):
            with pytest.warns(ConfigurationWarning):
                validate_few_shot_configuration(150, 2)


class TestSuboptimalConfigWarnings:
    """Test suboptimal configuration warnings."""
    
    def test_challenging_few_shot_warning(self):
        """Test warning for challenging few-shot configuration."""
        with pytest.warns(ConfigurationWarning, match="10-way 1-shot is very challenging"):
            warn_if_suboptimal_config(10, 1, "sqeuclidean", 1.0)
    
    def test_low_cosine_tau_warning(self):
        """Test warning for low temperature with cosine distance."""
        with pytest.warns(ConfigurationWarning, match="tau=0.05 is very low for cosine distance"):
            warn_if_suboptimal_config(5, 2, "cosine", 0.05)
    
    def test_very_low_euclidean_tau_warning(self):
        """Test warning for very low temperature with squared Euclidean distance."""
        with pytest.warns(ConfigurationWarning, match="tau=0.005 is very low for squared Euclidean"):
            warn_if_suboptimal_config(5, 2, "sqeuclidean", 0.005)


class TestValidationEdgeCases:
    """Test edge cases and boundary conditions for validation."""
    
    def test_empty_tensors(self):
        """Test validation with empty tensors."""
        support_x = torch.empty(0, 64)
        support_y = torch.empty(0, dtype=torch.long)
        query_x = torch.empty(0, 64)
        query_y = torch.empty(0, dtype=torch.long)
        
        with pytest.raises(ValidationError, match="Empty tensors are not allowed"):
            validate_episode_tensors(support_x, support_y, query_x, query_y)
    
    def test_single_sample_episode(self):
        """Test validation with single sample episodes."""
        support_x = torch.randn(1, 64)
        support_y = torch.tensor([0])
        query_x = torch.randn(1, 64)
        query_y = torch.tensor([0])
        
        # Should pass validation but might trigger quality warnings
        validate_episode_tensors(support_x, support_y, query_x, query_y)
    
    def test_extremely_high_dimensional_features(self):
        """Test validation with very high dimensional features."""
        support_x = torch.randn(10, 10000)  # Very high dimensional
        support_y = torch.arange(5).repeat(2)
        query_x = torch.randn(15, 10000)
        query_y = torch.arange(5).repeat(3)
        
        # Should pass validation
        validate_episode_tensors(support_x, support_y, query_x, query_y)
    
    def test_non_contiguous_tensors(self):
        """Test validation with non-contiguous tensors."""
        base_x = torch.randn(20, 64)
        base_y = torch.arange(10).repeat(2)
        
        # Create non-contiguous tensors using transpose and select
        support_x = base_x[:10].t().t()  # Non-contiguous
        support_y = base_y[:10]
        query_x = base_x[10:].t().t()  # Non-contiguous
        query_y = base_y[10:]
        
        assert not support_x.is_contiguous()
        assert not query_x.is_contiguous()
        
        # Should still pass validation
        validate_episode_tensors(support_x, support_y, query_x, query_y)
    
    def test_mixed_device_tensors(self):
        """Test validation with tensors on different devices."""
        support_x = torch.randn(10, 64)  # CPU
        support_y = torch.arange(5).repeat(2)  # CPU
        query_x = torch.randn(15, 64)  # CPU
        
        if torch.cuda.is_available():
            query_y = torch.arange(5).repeat(3).cuda()  # GPU
            
            with pytest.raises(ValidationError, match="All tensors must be on the same device"):
                validate_episode_tensors(support_x, support_y, query_x, query_y)
        else:
            # Skip test if CUDA not available
            pytest.skip("CUDA not available")
    
    def test_extreme_label_values(self):
        """Test validation with extreme label values."""
        support_x = torch.randn(4, 64)
        support_y = torch.tensor([0, 1000000, 2000000, 3000000])  # Very large labels
        query_x = torch.randn(4, 64)
        query_y = torch.tensor([0, 1000000, 2000000, 3000000])
        
        # Should pass validation despite large values
        validate_episode_tensors(support_x, support_y, query_x, query_y)
    
    def test_floating_point_labels(self):
        """Test validation rejects floating point labels."""
        support_x = torch.randn(4, 64)
        support_y = torch.tensor([0.0, 1.0, 2.0, 3.0])  # Float labels
        query_x = torch.randn(4, 64)
        query_y = torch.tensor([0.0, 1.0, 2.0, 3.0])
        
        with pytest.raises(ValidationError, match="Labels must be integer type"):
            validate_episode_tensors(support_x, support_y, query_x, query_y)
    
    def test_unicode_distance_metrics(self):
        """Test distance metric validation with unicode strings."""
        # Test with unicode characters
        with pytest.raises(ValidationError, match="Unsupported distance metric"):
            validate_distance_metric("cοsine")  # Contains unicode 'ο' instead of 'o'
    
    def test_temperature_extreme_values(self):
        """Test temperature validation with extreme values."""
        # Test with very small positive values
        validate_temperature_parameter(1e-100, "cosine")
        
        # Test with very large values
        validate_temperature_parameter(1e100, "cosine")
        
        # Test with values close to zero
        validate_temperature_parameter(1e-15, "sqeuclidean")
    
    def test_model_validation_edge_cases(self):
        """Test model validation edge cases."""
        # Test with model having mixed parameter types
        class MixedModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(10, 5)
                self.linear.weight.data.fill_(1.0)
                self.linear.bias.data[0] = float('inf')  # One inf parameter
        
        mixed_model = MixedModel()
        
        with pytest.raises(ValidationError, match="Model contains infinite parameters"):
            validate_model_parameters(mixed_model)
    
    def test_maml_config_boundary_conditions(self):
        """Test MAML configuration at boundary values."""
        # Test at exactly the warning thresholds
        validate_maml_config(0.1, 10, 0.001)  # At boundaries, should not warn
        
        # Test just over thresholds
        with pytest.warns(ConfigurationWarning):
            validate_maml_config(0.11, 11, 0.001)  # Should warn
    
    def test_few_shot_config_type_coercion(self):
        """Test few-shot configuration with values that could be coerced."""
        # Test with numpy integers
        import numpy as np
        
        n_way = np.int32(5)
        k_shot = np.int64(2)
        n_query = np.int16(10)
        
        # Should handle numpy integer types
        validate_few_shot_configuration(n_way, k_shot, n_query)
    
    def test_validation_with_gradients_enabled(self):
        """Test validation behavior with autograd enabled."""
        with torch.enable_grad():
            support_x = torch.randn(10, 64, requires_grad=True)
            support_y = torch.arange(5).repeat(2)
            query_x = torch.randn(15, 64, requires_grad=True)
            query_y = torch.arange(5).repeat(3)
            
            # Should pass validation even with gradients
            validate_episode_tensors(support_x, support_y, query_x, query_y)


class TestValidationMemoryEfficiency:
    """Test validation functions don't consume excessive memory."""
    
    def test_large_tensor_validation(self):
        """Test validation with large tensors doesn't copy data."""
        # Create large tensors
        large_support_x = torch.randn(1000, 512)
        large_support_y = torch.arange(100).repeat(10)
        large_query_x = torch.randn(2000, 512)
        large_query_y = torch.arange(100).repeat(20)
        
        # Get initial memory usage
        initial_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
        
        # Validation should not significantly increase memory
        validate_episode_tensors(large_support_x, large_support_y, large_query_x, large_query_y)
        
        final_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
        memory_increase = final_memory - initial_memory
        
        # Memory increase should be minimal (less than 1MB for bookkeeping)
        assert memory_increase < 1024 * 1024  # Less than 1MB
    
    def test_repeated_validation_memory_leak(self):
        """Test repeated validation calls don't leak memory."""
        support_x = torch.randn(50, 64)
        support_y = torch.arange(10).repeat(5)
        query_x = torch.randn(100, 64)
        query_y = torch.arange(10).repeat(10)
        
        initial_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
        
        # Run validation many times
        for _ in range(100):
            validate_episode_tensors(support_x, support_y, query_x, query_y)
        
        final_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
        memory_increase = final_memory - initial_memory
        
        # Should not significantly increase memory
        assert memory_increase < 512 * 1024  # Less than 512KB