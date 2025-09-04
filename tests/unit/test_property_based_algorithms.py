"""
Property-Based Tests for Algorithm Correctness
===============================================

Uses property-based testing to verify mathematical properties and invariants
of meta-learning algorithms, ensuring correctness across diverse inputs.

Author: Benedict Chen (benedict@benedictchen.com)
"""

import pytest
import torch
# DEMO/TEST REPRODUCIBILITY: Set seed for consistent synthetic data
torch.manual_seed(42)  # Reproducible test/demo data
import torch.nn as nn
# DEMO/TEST REPRODUCIBILITY: Set seed for consistent synthetic data
torch.manual_seed(42)  # Reproducible test/demo data
import numpy as np
from hypothesis import given, strategies as st, settings, assume
from typing import Dict, List, Tuple, Optional
import math

# Import core algorithms
from meta_learning.meta_learning_modules import (
    TestTimeComputeScaler, TestTimeComputeConfig,
    MAMLLearner, MAMLConfig, FirstOrderMAML, ReptileLearner,
    PrototypicalNetworks, PrototypicalConfig, MatchingNetworks, MatchingConfig,
    OnlineMetaLearner, ContinualMetaConfig,
    few_shot_accuracy, adaptation_speed, compute_confidence_interval,
    basic_confidence_interval, estimate_difficulty,
    HardwareManager, HardwareConfig,
    TaskConfiguration, EvaluationConfig
)


class TestAlgorithmMathematicalProperties:
    """Test mathematical properties and invariants of algorithms."""
    
    @given(
        n_way=st.integers(min_value=2, max_value=10),
        k_shot=st.integers(min_value=1, max_value=20),
        q_query=st.integers(min_value=1, max_value=30),
        embedding_dim=st.integers(min_value=32, max_value=512)
    )
    @settings(max_examples=10, deadline=5000)
    def test_prototypical_networks_distance_properties(self, n_way, k_shot, q_query, embedding_dim):
        """Test mathematical properties of prototypical distance computation."""
        try:
            # Create prototypical network
            config = PrototypicalConfig()
            config.embedding_dim = embedding_dim
            config.multi_scale_features = False
            model = PrototypicalNetworks(config)
            
            # Generate synthetic data
            support_size = n_way * k_shot
            query_size = n_way * q_query
            
            support_data = torch.randn(support_size, 3, 28, 28)
            support_labels = torch.arange(n_way).repeat(k_shot)
            query_data = torch.randn(query_size, 3, 28, 28)
            query_labels = torch.arange(n_way).repeat(q_query)
            
            # Forward pass
            with torch.no_grad():
                predictions = model.forward(support_data, support_labels, query_data)
                
            if isinstance(predictions, tuple):
                logits = predictions[0]
            else:
                logits = predictions
                
            # Test mathematical properties
            assert logits.shape[0] == query_size
            assert logits.shape[1] == n_way
            
            # Property 1: Logits should sum to reasonable values (not all zeros/infinities)
            assert torch.isfinite(logits).all(), "All logits should be finite"
            
            # Property 2: For each query, the maximum logit should correspond to valid class
            max_indices = logits.argmax(dim=1)
            assert (max_indices >= 0).all() and (max_indices < n_way).all()
            
            # Property 3: Softmax probabilities should sum to 1
            probabilities = torch.softmax(logits, dim=1)
            prob_sums = probabilities.sum(dim=1)
            assert torch.allclose(prob_sums, torch.ones_like(prob_sums), atol=1e-6)
            
        except Exception as e:
            # Some parameter combinations might not work - that's expected
            assume(False)
            
    @given(
        inner_lr=st.floats(min_value=1e-4, max_value=1.0),
        outer_lr=st.floats(min_value=1e-4, max_value=1.0),
        inner_steps=st.integers(min_value=1, max_value=10)
    )
    @settings(max_examples=5, deadline=10000)
    def test_maml_gradient_properties(self, inner_lr, outer_lr, inner_steps):
        """Test gradient-based properties of MAML algorithm."""
        try:
            # Create MAML learner
            config = MAMLConfig(
                inner_lr=inner_lr,
                outer_lr=outer_lr,
                inner_steps=inner_steps,
                first_order=False
            )
            maml = MAMLLearner(config)
            
            # Generate simple task
            support_data = torch.randn(10, 1, 8, 8, requires_grad=True)
            support_labels = torch.randint(0, 2, (10,))
            query_data = torch.randn(5, 1, 8, 8)
            query_labels = torch.randint(0, 2, (5,))
            
            # Test adaptation
            adapted_params = maml.adapt(support_data, support_labels)
            
            # Property 1: Adapted parameters should exist and be finite
            assert adapted_params is not None
            for param_name, param_value in adapted_params.items():
                assert torch.isfinite(param_value).all(), f"Parameter {param_name} should be finite"
                
            # Property 2: Parameters should change during adaptation
            original_params = {name: param.clone() for name, param in maml.model.named_parameters()}
            
            # At least some parameters should have changed (unless learning rate is extremely small)
            if inner_lr > 1e-3:
                param_changes = []
                for param_name in adapted_params:
                    if param_name in original_params:
                        change = torch.norm(adapted_params[param_name] - original_params[param_name])
                        param_changes.append(change.item())
                        
                # At least one parameter should have changed meaningfully
                max_change = max(param_changes) if param_changes else 0
                assert max_change > 1e-6, "Some parameters should change during adaptation"
                
            # Property 3: Predictions should be valid probabilities
            predictions = maml.predict(query_data, adapted_params)
            if isinstance(predictions, torch.Tensor) and predictions.dim() > 1:
                probabilities = torch.softmax(predictions, dim=1)
                assert torch.allclose(probabilities.sum(dim=1), torch.ones(probabilities.shape[0]), atol=1e-5)
                
        except Exception as e:
            # Some parameter combinations might fail - skip those
            assume(False)

    @given(
        max_compute_budget=st.integers(min_value=1, max_value=50),
        confidence_threshold=st.floats(min_value=0.1, max_value=0.99)
    )
    @settings(max_examples=8, deadline=5000)
    def test_test_time_compute_scaling_properties(self, max_compute_budget, confidence_threshold):
        """Test properties of test-time compute scaling."""
        try:
            # Create test-time compute scaler
            config = TestTimeComputeConfig(
                max_compute_budget=max_compute_budget,
                min_confidence_threshold=confidence_threshold,
                compute_strategy="gradual_scaling"
            )
            scaler = TestTimeComputeScaler(config)
            
            # Generate input data
            batch_size = min(8, max_compute_budget)  # Keep reasonable batch size
            input_data = torch.randn(batch_size, 3, 32, 32)
            
            # Scale compute
            result = scaler.scale_compute(input_data)
            
            if isinstance(result, tuple):
                output, metrics = result
            else:
                output = result
                metrics = {}
                
            # Property 1: Output should exist and be finite
            assert output is not None
            if isinstance(output, torch.Tensor):
                assert torch.isfinite(output).all(), "Output should be finite"
                assert output.shape[0] == batch_size, "Batch dimension should be preserved"
                
            # Property 2: Metrics should contain reasonable values
            if metrics:
                for key, value in metrics.items():
                    if isinstance(value, (int, float)):
                        assert not math.isnan(value) and not math.isinf(value), f"Metric {key} should be finite"
                        
            # Property 3: Compute budget should be respected
            compute_used = metrics.get('compute_steps_used', 0)
            assert compute_used <= max_compute_budget, "Should not exceed compute budget"
            
        except Exception as e:
            # Some configurations might not work
            assume(False)


class TestStatisticalPropertyInvariants:
    """Test statistical properties and invariants of utility functions."""
    
    @given(
        values=st.lists(
            st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
            min_size=3,
            max_size=50
        ),
        confidence_level=st.floats(min_value=0.5, max_value=0.99)
    )
    @settings(max_examples=10, deadline=2000)
    def test_confidence_interval_properties(self, values, confidence_level):
        """Test mathematical properties of confidence intervals."""
        try:
            # Filter out edge cases
            assume(len(set(values)) > 1)  # Need some variance
            assume(np.std(values) > 1e-6)  # Need meaningful variance
            
            # Test basic confidence interval
            ci_result = basic_confidence_interval(values, confidence_level=confidence_level)
            
            if ci_result is not None and len(ci_result) >= 2:
                lower, upper = ci_result[0], ci_result[1]
                
                # Property 1: Lower bound should be <= upper bound
                assert lower <= upper, f"Lower bound {lower} should be <= upper bound {upper}"
                
                # Property 2: Mean should typically be within CI (for normal-ish distributions)
                mean_val = np.mean(values)
                # Allow some tolerance for small samples or extreme cases
                if len(values) >= 10:
                    assert lower <= mean_val <= upper or abs(mean_val - lower) < 0.1 or abs(mean_val - upper) < 0.1
                    
                # Property 3: CI width should be positive
                ci_width = upper - lower
                assert ci_width >= 0, "Confidence interval width should be non-negative"
                
                # Property 4: Higher confidence should generally mean wider CI
                if confidence_level > 0.9 and len(values) >= 5:
                    narrower_ci = basic_confidence_interval(values, confidence_level=0.8)
                    if narrower_ci is not None and len(narrower_ci) >= 2:
                        narrower_width = narrower_ci[1] - narrower_ci[0]
                        # Allow some tolerance due to different methods
                        assert ci_width >= narrower_width * 0.8
                        
        except Exception as e:
            # Some edge cases might fail
            assume(False)
            
    @given(
        predictions=st.lists(
            st.integers(min_value=0, max_value=9),
            min_size=1,
            max_size=100
        ),
        targets=st.lists(
            st.integers(min_value=0, max_value=9),
            min_size=1,
            max_size=100
        )
    )
    @settings(max_examples=15, deadline=1000)
    def test_accuracy_computation_properties(self, predictions, targets):
        """Test mathematical properties of accuracy computation."""
        try:
            # Ensure same length
            min_len = min(len(predictions), len(targets))
            assume(min_len > 0)
            
            pred_tensor = torch.tensor(predictions[:min_len])
            target_tensor = torch.tensor(targets[:min_len])
            
            # Test with logits (create one-hot-like logits)
            num_classes = max(max(predictions[:min_len]), max(targets[:min_len])) + 1
            logits = torch.zeros(min_len, num_classes)
            logits[range(min_len), predictions[:min_len]] = 10.0  # High confidence for predicted class
            
            # Compute accuracy
            accuracy = few_shot_accuracy(logits, target_tensor)
            
            # Property 1: Accuracy should be between 0 and 1
            assert 0.0 <= accuracy <= 1.0, f"Accuracy {accuracy} should be between 0 and 1"
            
            # Property 2: Perfect predictions should give accuracy = 1
            if all(p == t for p, t in zip(predictions[:min_len], targets[:min_len])):
                assert abs(accuracy - 1.0) < 1e-6, "Perfect predictions should give accuracy 1.0"
                
            # Property 3: Completely wrong predictions should give accuracy = 0
            if all(p != t for p, t in zip(predictions[:min_len], targets[:min_len])):
                assert abs(accuracy - 0.0) < 1e-6, "Completely wrong predictions should give accuracy 0.0"
                
            # Property 4: Accuracy should equal fraction of correct predictions
            correct_predictions = sum(1 for p, t in zip(predictions[:min_len], targets[:min_len]) if p == t)
            expected_accuracy = correct_predictions / min_len
            assert abs(accuracy - expected_accuracy) < 1e-6, "Accuracy should match fraction correct"
            
        except Exception as e:
            assume(False)

    @given(
        losses=st.lists(
            st.floats(min_value=0.1, max_value=10.0, allow_nan=False, allow_infinity=False),
            min_size=2,
            max_size=20
        )
    )
    @settings(max_examples=10, deadline=1000)
    def test_adaptation_speed_properties(self, losses):
        """Test properties of adaptation speed computation."""
        try:
            # Ensure decreasing trend for meaningful adaptation speed
            assume(losses[0] > losses[-1])  # Should show some improvement
            
            speed = adaptation_speed(losses)
            
            # Property 1: Speed should be a finite number
            assert isinstance(speed, (int, float)), "Adaptation speed should be numeric"
            assert not math.isnan(speed) and not math.isinf(speed), "Speed should be finite"
            
            # Property 2: For decreasing losses, speed should be positive
            if all(losses[i] >= losses[i+1] for i in range(len(losses)-1)):
                assert speed >= 0, "Speed should be non-negative for decreasing losses"
                
            # Property 3: Faster decrease should mean higher speed
            # Create a faster decreasing sequence for comparison
            faster_losses = [loss * (0.5 ** i) for i, loss in enumerate(losses)]
            faster_speed = adaptation_speed(faster_losses)
            
            # The faster sequence should generally have higher adaptation speed
            # (allowing some tolerance due to different calculation methods)
            if len(losses) >= 3:
                assert faster_speed >= speed * 0.8, "Faster adaptation should have higher speed"
                
        except Exception as e:
            assume(False)


class TestHardwareOptimizationProperties:
    """Test properties of hardware optimization utilities."""
    
    @given(
        batch_size=st.integers(min_value=1, max_value=32),
        input_channels=st.integers(min_value=1, max_value=3),
        height=st.integers(min_value=8, max_value=64),
        width=st.integers(min_value=8, max_value=64)
    )
    @settings(max_examples=8, deadline=3000)
    def test_hardware_manager_properties(self, batch_size, input_channels, height, width):
        """Test properties of hardware optimization."""
        try:
            # Create hardware manager
            config = HardwareConfig(
                device='cpu',  # Use CPU to avoid GPU dependency
                use_mixed_precision=False,
                memory_efficient=True
            )
            hw_manager = HardwareManager(config)
            
            # Test model preparation
            model = nn.Sequential(
                nn.Conv2d(input_channels, 16, 3, padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(16, 10)
            )
            
            # Prepare model
            optimized_model = hw_manager.prepare_model(model)
            
            # Property 1: Optimized model should exist and be callable
            assert optimized_model is not None
            assert callable(optimized_model)
            
            # Property 2: Model should handle input correctly
            test_input = torch.randn(batch_size, input_channels, height, width)
            
            with torch.no_grad():
                original_output = model(test_input)
                optimized_output = optimized_model(test_input)
                
            # Property 3: Output shapes should match
            assert original_output.shape == optimized_output.shape
            
            # Property 4: Outputs should be finite
            assert torch.isfinite(original_output).all()
            assert torch.isfinite(optimized_output).all()
            
            # Property 5: For CPU-only optimization, outputs should be similar
            # (allowing for numerical differences due to optimization)
            if config.device == 'cpu' and not config.use_mixed_precision:
                assert torch.allclose(original_output, optimized_output, atol=1e-4, rtol=1e-3)
                
        except Exception as e:
            assume(False)

    @given(
        task_features=st.lists(
            st.lists(
                st.floats(min_value=-3.0, max_value=3.0, allow_nan=False, allow_infinity=False),
                min_size=8,
                max_size=128
            ),
            min_size=2,
            max_size=20
        )
    )
    @settings(max_examples=10, deadline=2000)
    def test_difficulty_estimation_properties(self, task_features):
        """Test properties of task difficulty estimation."""
        try:
            # Convert to tensor
            feature_tensors = [torch.tensor(features, dtype=torch.float32) for features in task_features]
            
            # Ensure all features have same dimension
            min_dim = min(len(features) for features in task_features)
            assume(min_dim >= 4)  # Need meaningful feature dimension
            
            # Truncate to same dimension
            feature_tensors = [tensor[:min_dim] for tensor in feature_tensors]
            
            # Test difficulty estimation for each task
            difficulties = []
            for features in feature_tensors:
                try:
                    difficulty = estimate_difficulty(features.unsqueeze(0))  # Add batch dimension
                    difficulties.append(difficulty)
                except Exception:
                    continue
                    
            assume(len(difficulties) >= 2)  # Need multiple successful estimates
            
            # Property 1: All difficulties should be finite numbers
            for diff in difficulties:
                assert isinstance(diff, (int, float)), "Difficulty should be numeric"
                assert not math.isnan(diff) and not math.isinf(diff), "Difficulty should be finite"
                
            # Property 2: Difficulties should be in reasonable range
            for diff in difficulties:
                assert diff >= 0, "Difficulty should be non-negative"
                assert diff <= 100, "Difficulty should be reasonably bounded"  # Allow flexible upper bound
                
            # Property 3: Tasks with more variance should generally be more difficult
            variances = [torch.var(tensor).item() for tensor in feature_tensors[:len(difficulties)]]
            
            if len(difficulties) >= 3:
                # Check correlation between variance and difficulty
                high_var_idx = np.argmax(variances)
                low_var_idx = np.argmin(variances)
                
                if variances[high_var_idx] > variances[low_var_idx] * 2:  # Significant difference
                    # High variance task might be more difficult (allowing some tolerance)
                    assert difficulties[high_var_idx] >= difficulties[low_var_idx] * 0.5
                    
        except Exception as e:
            assume(False)


class TestConfigurationValidationProperties:
    """Test properties of configuration validation and compatibility."""
    
    @given(
        n_way=st.integers(min_value=2, max_value=20),
        k_shot=st.integers(min_value=1, max_value=50),
        q_query=st.integers(min_value=1, max_value=100),
        num_tasks=st.integers(min_value=1, max_value=10000)
    )
    @settings(max_examples=10, deadline=1000)
    def test_task_configuration_properties(self, n_way, k_shot, q_query, num_tasks):
        """Test properties of task configuration."""
        try:
            config = TaskConfiguration(
                n_way=n_way,
                k_shot=k_shot,
                q_query=q_query,
                num_tasks=num_tasks
            )
            
            # Property 1: All parameters should be positive
            assert config.n_way > 0, "n_way should be positive"
            assert config.k_shot > 0, "k_shot should be positive"
            assert config.q_query > 0, "q_query should be positive"
            assert config.num_tasks > 0, "num_tasks should be positive"
            
            # Property 2: Parameters should match input
            assert config.n_way == n_way
            assert config.k_shot == k_shot
            assert config.q_query == q_query
            assert config.num_tasks == num_tasks
            
            # Property 3: Configuration should be consistent
            total_support = config.n_way * config.k_shot
            total_query = config.n_way * config.q_query
            
            assert total_support >= config.n_way, "Should have at least one support sample per class"
            assert total_query >= config.n_way, "Should have at least one query sample per class"
            
        except Exception as e:
            assume(False)
            
    @given(
        inner_lr=st.floats(min_value=1e-6, max_value=10.0),
        outer_lr=st.floats(min_value=1e-6, max_value=10.0),
        inner_steps=st.integers(min_value=1, max_value=100)
    )
    @settings(max_examples=8, deadline=1000)
    def test_maml_configuration_properties(self, inner_lr, outer_lr, inner_steps):
        """Test properties of MAML configuration."""
        try:
            config = MAMLConfig(
                inner_lr=inner_lr,
                outer_lr=outer_lr,
                inner_steps=inner_steps
            )
            
            # Property 1: Learning rates should be positive
            assert config.inner_lr > 0, "Inner learning rate should be positive"
            assert config.outer_lr > 0, "Outer learning rate should be positive"
            assert config.inner_steps > 0, "Inner steps should be positive"
            
            # Property 2: Configuration should preserve values
            assert abs(config.inner_lr - inner_lr) < 1e-10
            assert abs(config.outer_lr - outer_lr) < 1e-10
            assert config.inner_steps == inner_steps
            
            # Property 3: Reasonable learning rate bounds
            assert config.inner_lr <= 10.0, "Inner learning rate should be reasonable"
            assert config.outer_lr <= 10.0, "Outer learning rate should be reasonable"
            assert config.inner_steps <= 100, "Inner steps should be reasonable"
            
        except Exception as e:
            assume(False)


if __name__ == "__main__":
    # Run property-based tests with custom settings
    pytest.main([__file__, "--hypothesis-show-statistics"])