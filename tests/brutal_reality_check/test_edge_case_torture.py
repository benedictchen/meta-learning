"""
🔥 EDGE CASE TORTURE TESTS 🔥
=============================

These tests will TORTURE our algorithms with every possible edge case, error condition,
and numerical nastiness to expose where our "breakthrough implementations" break down.

If our code is as robust as we claim, these tests should pass.
If not, we'll discover our algorithms are fragile toys.

Author: The Brutal Technical Advisor
Status: Maximum torture, assume perfection (for now)
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import warnings
from unittest.mock import patch, MagicMock
import logging
import traceback

from meta_learning.meta_learning_modules import (
    TestTimeComputeScaler, TestTimeComputeConfig,
    MAMLLearner, MAMLConfig, FirstOrderMAML,
    PrototypicalNetworks, PrototypicalConfig,
    OnlineMetaLearner, OnlineMetaConfig,
    MetaLearningDataset, TaskConfiguration,
    EvaluationMetrics, MetricsConfig,
    StatisticalAnalysis, StatsConfig
)


class TortureTestValidator:
    """
    The torturer that will find every weakness in our code.
    """
    
    @staticmethod
    def create_pathological_data(case_name: str, **kwargs) -> Dict[str, torch.Tensor]:
        """
        Create pathological data that breaks poorly written algorithms.
        """
        torch.manual_seed(42)
        
        if case_name == "all_zeros":
            return {
                'support_x': torch.zeros(kwargs.get('n_support', 10), kwargs.get('dim', 784)),
                'support_y': torch.zeros(kwargs.get('n_support', 10), dtype=torch.long),
                'query_x': torch.zeros(kwargs.get('n_query', 5), kwargs.get('dim', 784))
            }
        
        elif case_name == "all_identical":
            identical_pattern = torch.ones(kwargs.get('dim', 784)) * 0.5
            return {
                'support_x': identical_pattern.unsqueeze(0).repeat(kwargs.get('n_support', 10), 1),
                'support_y': torch.zeros(kwargs.get('n_support', 10), dtype=torch.long),
                'query_x': identical_pattern.unsqueeze(0).repeat(kwargs.get('n_query', 5), 1)
            }
        
        elif case_name == "extreme_values":
            return {
                'support_x': torch.tensor([1e10, -1e10, 1e-10, -1e-10] * (kwargs.get('dim', 784) // 4 + 1))[:kwargs.get('dim', 784)].unsqueeze(0).repeat(kwargs.get('n_support', 10), 1),
                'support_y': torch.randint(0, kwargs.get('n_classes', 2), (kwargs.get('n_support', 10),)),
                'query_x': torch.tensor([5e9, -5e9, 5e-9, -5e-9] * (kwargs.get('dim', 784) // 4 + 1))[:kwargs.get('dim', 784)].unsqueeze(0).repeat(kwargs.get('n_query', 5), 1)
            }
        
        elif case_name == "nan_values":
            data = torch.randn(kwargs.get('n_support', 10), kwargs.get('dim', 784))
            data[0, 0] = float('nan')
            return {
                'support_x': data,
                'support_y': torch.randint(0, kwargs.get('n_classes', 2), (kwargs.get('n_support', 10),)),
                'query_x': torch.randn(kwargs.get('n_query', 5), kwargs.get('dim', 784))
            }
        
        elif case_name == "inf_values":
            data = torch.randn(kwargs.get('n_support', 10), kwargs.get('dim', 784))
            data[0, :5] = float('inf')
            data[1, :5] = float('-inf')
            return {
                'support_x': data,
                'support_y': torch.randint(0, kwargs.get('n_classes', 2), (kwargs.get('n_support', 10),)),
                'query_x': torch.randn(kwargs.get('n_query', 5), kwargs.get('dim', 784))
            }
        
        elif case_name == "single_sample":
            return {
                'support_x': torch.randn(1, kwargs.get('dim', 784)),
                'support_y': torch.tensor([0]),
                'query_x': torch.randn(1, kwargs.get('dim', 784))
            }
        
        elif case_name == "mismatched_dimensions":
            return {
                'support_x': torch.randn(kwargs.get('n_support', 10), kwargs.get('dim', 784)),
                'support_y': torch.randint(0, kwargs.get('n_classes', 2), (kwargs.get('n_support', 10) + 1,)),  # Intentional mismatch
                'query_x': torch.randn(kwargs.get('n_query', 5), kwargs.get('dim', 784) + 1)  # Intentional mismatch
            }
        
        elif case_name == "empty_classes":
            support_y = torch.tensor([0, 0, 2, 2])  # Missing class 1
            return {
                'support_x': torch.randn(4, kwargs.get('dim', 784)),
                'support_y': support_y,
                'query_x': torch.randn(kwargs.get('n_query', 5), kwargs.get('dim', 784))
            }
        
        else:
            raise ValueError(f"Unknown pathological case: {case_name}")
    
    @staticmethod
    def check_graceful_failure(test_function, expected_error_types: List[type] = None):
        """
        Check that algorithms fail gracefully rather than silently producing garbage.
        """
        try:
            result = test_function()
            return {
                'failed_gracefully': False,
                'result': result,
                'error_type': None,
                'error_message': None
            }
        except Exception as e:
            expected_error = expected_error_types is None or type(e) in expected_error_types
            return {
                'failed_gracefully': True,
                'result': None,
                'error_type': type(e).__name__,
                'error_message': str(e),
                'expected_error': expected_error
            }


class TestPrototypicalNetworksEdgeCaseTorture:
    """
    TORTURE TEST: Break Prototypical Networks with edge cases.
    If it's robust, it should handle these gracefully.
    """
    
    def test_all_zero_inputs_handling(self):
        """
        Test behavior when all inputs are zero.
        Poorly implemented algorithms will produce NaN or crash.
        """
        torch.manual_seed(42)
        
        encoder = nn.Linear(784, 64)
        model = PrototypicalNetworks(encoder, {"n_way": 3})
        
        torture_data = TortureTestValidator.create_pathological_data(
            "all_zeros", n_support=9, n_query=3, dim=784
        )
        
        # TORTURE TEST: All zero inputs
        def run_with_zeros():
            with torch.no_grad():
                return model(
                    torture_data['support_x'], 
                    torch.tensor([0, 0, 0, 1, 1, 1, 2, 2, 2]),
                    torture_data['query_x']
                )
        
        result = TortureTestValidator.check_graceful_failure(
            run_with_zeros, 
            expected_error_types=[RuntimeError, ValueError]
        )
        
        # Either handle gracefully OR fail with appropriate error
        if not result['failed_gracefully']:
            # If it doesn't fail, check outputs are valid
            output = result['result']
            assert not torch.isnan(output).any(), "Zero inputs produced NaN outputs!"
            assert not torch.isinf(output).any(), "Zero inputs produced infinite outputs!"
        else:
            assert result['expected_error'], f"Unexpected error type: {result['error_type']}"
    
    def test_identical_samples_handling(self):
        """
        Test behavior when all samples are identical.
        This breaks algorithms that assume sample diversity.
        """
        torch.manual_seed(42)
        
        encoder = nn.Linear(784, 64)
        model = PrototypicalNetworks(encoder, {"n_way": 2})
        
        torture_data = TortureTestValidator.create_pathological_data(
            "all_identical", n_support=6, n_query=2, dim=784
        )
        
        def run_with_identical():
            with torch.no_grad():
                return model(
                    torture_data['support_x'],
                    torch.tensor([0, 0, 0, 1, 1, 1]),
                    torture_data['query_x']
                )
        
        result = TortureTestValidator.check_graceful_failure(run_with_identical)
        
        # Should handle identical samples without crashing
        if not result['failed_gracefully']:
            output = result['result']
            assert output.shape == (2, 2), f"Wrong output shape: {output.shape}"
            assert not torch.isnan(output).any(), "Identical inputs produced NaN!"
            assert not torch.isinf(output).any(), "Identical inputs produced infinity!"
    
    def test_extreme_value_numerical_stability(self):
        """
        Test numerical stability with extreme values.
        This exposes overflow/underflow issues.
        """
        torch.manual_seed(42)
        
        encoder = nn.Linear(784, 64)  
        model = PrototypicalNetworks(encoder, {"n_way": 2})
        
        torture_data = TortureTestValidator.create_pathological_data(
            "extreme_values", n_support=4, n_query=2, dim=784, n_classes=2
        )
        
        def run_with_extremes():
            with torch.no_grad():
                return model(
                    torture_data['support_x'],
                    torture_data['support_y'],
                    torture_data['query_x']
                )
        
        result = TortureTestValidator.check_graceful_failure(
            run_with_extremes,
            expected_error_types=[RuntimeError, ValueError, OverflowError]
        )
        
        if not result['failed_gracefully']:
            output = result['result']
            # If it handles extreme values, outputs should still be finite
            assert torch.isfinite(output).all(), "Extreme values caused non-finite outputs!"
    
    def test_nan_contamination_handling(self):
        """
        Test behavior with NaN inputs.
        NaN should either be handled or cause appropriate errors.
        """
        torch.manual_seed(42)
        
        encoder = nn.Linear(784, 64)
        model = PrototypicalNetworks(encoder, {"n_way": 2})
        
        torture_data = TortureTestValidator.create_pathological_data(
            "nan_values", n_support=4, n_query=2, dim=784, n_classes=2
        )
        
        def run_with_nans():
            with torch.no_grad():
                return model(
                    torture_data['support_x'],
                    torture_data['support_y'],
                    torture_data['query_x']
                )
        
        result = TortureTestValidator.check_graceful_failure(
            run_with_nans,
            expected_error_types=[RuntimeError, ValueError]
        )
        
        # NaN inputs should either be handled or cause controlled failure
        if not result['failed_gracefully']:
            # If it processes NaN, it should contain the contamination
            output = result['result']
            nan_in_output = torch.isnan(output).any()
            # This is acceptable - NaN handling is algorithm-dependent
        else:
            assert result['expected_error'], f"Unexpected error with NaN: {result['error_type']}"
    
    def test_single_sample_edge_case(self):
        """
        Test behavior with minimal data (single sample per class).
        This tests robustness of statistical computations.
        """
        torch.manual_seed(42)
        
        encoder = nn.Linear(784, 64)
        model = PrototypicalNetworks(encoder, {"n_way": 1})
        
        torture_data = TortureTestValidator.create_pathological_data(
            "single_sample", dim=784
        )
        
        def run_with_single_sample():
            with torch.no_grad():
                return model(
                    torture_data['support_x'],
                    torture_data['support_y'], 
                    torture_data['query_x']
                )
        
        result = TortureTestValidator.check_graceful_failure(run_with_single_sample)
        
        # Single sample should be handled (it's a valid edge case)
        if not result['failed_gracefully']:
            output = result['result']
            assert output.shape == (1, 1), f"Wrong shape for single sample: {output.shape}"
            assert not torch.isnan(output).any(), "Single sample produced NaN!"
        else:
            print(f"Single sample handling failed: {result['error_message']}")
    
    def test_mismatched_dimension_error_handling(self):
        """
        Test error handling with mismatched tensor dimensions.
        This should fail fast with clear error messages.
        """
        torch.manual_seed(42)
        
        encoder = nn.Linear(784, 64)
        model = PrototypicalNetworks(encoder, {"n_way": 2})
        
        torture_data = TortureTestValidator.create_pathological_data(
            "mismatched_dimensions", n_support=4, n_query=2, dim=784, n_classes=2
        )
        
        def run_with_mismatch():
            with torch.no_grad():
                return model(
                    torture_data['support_x'],
                    torture_data['support_y'][:4],  # Correct size
                    torture_data['query_x'][:2, :784]  # Correct size
                )
        
        result = TortureTestValidator.check_graceful_failure(
            run_with_mismatch,
            expected_error_types=[RuntimeError, ValueError, IndexError]
        )
        
        # Dimension mismatches SHOULD cause errors
        assert result['failed_gracefully'], "Dimension mismatch should cause an error!"
        assert result['expected_error'], f"Wrong error type for mismatch: {result['error_type']}"


class TestMAMLEdgeCaseTorture:
    """
    TORTURE TEST: Break MAML with edge cases.
    Meta-learning is complex - lots of places to break.
    """
    
    def test_zero_gradient_inner_update_handling(self):
        """
        Test MAML behavior when gradients are zero (perfect fit case).
        This tests numerical stability of meta-learning.
        """
        torch.manual_seed(42)
        
        # Model that's already perfect on the task
        class PerfectModel(nn.Module):
            def forward(self, x):
                # Return logits that perfectly classify the input
                batch_size = x.shape[0]
                # Assume first half of batch is class 0, second half is class 1
                perfect_logits = torch.zeros(batch_size, 2)
                perfect_logits[:batch_size//2, 0] = 100.0  # Very confident class 0
                perfect_logits[batch_size//2:, 1] = 100.0  # Very confident class 1
                return perfect_logits
        
        model = PerfectModel()
        config = MAMLConfig(inner_lr=0.1, inner_steps=1)
        maml = MAMLLearner(model, config)
        
        # Create task where model is already perfect
        support_x = torch.randn(4, 10)
        support_y = torch.tensor([0, 0, 1, 1])
        query_x = torch.randn(2, 10)
        query_y = torch.tensor([0, 1])
        
        def run_with_zero_gradients():
            return maml.adapt_to_task(support_x, support_y, query_x, query_y)
        
        result = TortureTestValidator.check_graceful_failure(run_with_zero_gradients)
        
        # Should handle zero gradients gracefully (no division by zero, etc.)
        if not result['failed_gracefully']:
            adapted_model = result['result']
            assert adapted_model is not None, "Zero gradient adaptation failed!"
        else:
            # If it fails, should be a controlled failure
            print(f"Zero gradient handling: {result['error_message']}")
    
    def test_exploding_gradient_robustness(self):
        """
        Test MAML robustness against exploding gradients.
        Large learning rates or steep loss landscapes can cause this.
        """
        torch.manual_seed(42)
        
        # Model that will produce large gradients
        class GradientExplodingModel(nn.Module):
            def __init__(self):
                super().__init__()
                # Initialize with very large weights to cause gradient explosion
                self.linear = nn.Linear(10, 2)
                with torch.no_grad():
                    self.linear.weight.fill_(100.0)
                    self.linear.bias.fill_(100.0)
            
            def forward(self, x):
                return self.linear(x)
        
        model = GradientExplodingModel()
        config = MAMLConfig(inner_lr=10.0, inner_steps=1)  # Large LR for explosion
        maml = MAMLLearner(model, config)
        
        # Create task with extreme targets to amplify gradients
        support_x = torch.randn(6, 10) * 10.0  # Large inputs
        support_y = torch.tensor([0, 0, 0, 1, 1, 1])
        query_x = torch.randn(3, 10)
        query_y = torch.tensor([0, 1, 0])
        
        def run_with_exploding_gradients():
            return maml.meta_train_step([(support_x, support_y, query_x, query_y)])
        
        result = TortureTestValidator.check_graceful_failure(
            run_with_exploding_gradients,
            expected_error_types=[RuntimeError, ValueError]
        )
        
        if not result['failed_gracefully']:
            meta_result = result['result']
            # If it handles exploding gradients, loss should be finite
            assert np.isfinite(meta_result['meta_loss']), "Meta-loss exploded to infinity!"
        else:
            # Gradient explosion should be detected and handled
            print(f"Gradient explosion handled: {result['error_message']}")
    
    def test_empty_meta_batch_handling(self):
        """
        Test MAML behavior with empty meta-batch.
        This tests input validation.
        """
        torch.manual_seed(42)
        
        model = nn.Linear(5, 2)
        config = MAMLConfig(inner_lr=0.1, inner_steps=1)
        maml = MAMLLearner(model, config)
        
        def run_with_empty_batch():
            return maml.meta_train_step([])  # Empty meta-batch
        
        result = TortureTestValidator.check_graceful_failure(
            run_with_empty_batch,
            expected_error_types=[ValueError, IndexError, RuntimeError]
        )
        
        # Empty meta-batch SHOULD cause an error
        assert result['failed_gracefully'], "Empty meta-batch should cause an error!"
        assert result['expected_error'], f"Wrong error type for empty batch: {result['error_type']}"
    
    def test_extremely_deep_inner_loop_stability(self):
        """
        Test MAML stability with very deep inner loops.
        This can cause gradient vanishing or explosion.
        """
        torch.manual_seed(42)
        
        model = nn.Linear(5, 2)
        config = MAMLConfig(inner_lr=0.01, inner_steps=100)  # Very deep inner loop
        maml = MAMLLearner(model, config)
        
        support_x = torch.randn(10, 5)
        support_y = torch.randint(0, 2, (10,))
        query_x = torch.randn(5, 5)
        query_y = torch.randint(0, 2, (5,))
        
        def run_with_deep_inner_loop():
            return maml.meta_train_step([(support_x, support_y, query_x, query_y)])
        
        result = TortureTestValidator.check_graceful_failure(
            run_with_deep_inner_loop,
            expected_error_types=[RuntimeError, ValueError]
        )
        
        if not result['failed_gracefully']:
            meta_result = result['result']
            # Deep inner loops should still produce finite results
            assert np.isfinite(meta_result['meta_loss']), "Deep inner loop caused infinite loss!"
        else:
            # Deep loops might hit computational limits
            print(f"Deep inner loop handling: {result['error_message']}")


class TestTestTimeComputeEdgeCaseTorture:
    """
    TORTURE TEST: Break Test-Time Compute Scaling with edge cases.
    This is our "breakthrough" algorithm - let's see how breakthrough it really is.
    """
    
    def test_zero_compute_budget_handling(self):
        """
        Test behavior with zero or negative compute budget.
        This tests input validation and error handling.
        """
        torch.manual_seed(42)
        
        model = nn.Linear(10, 3)
        
        # Test zero budget
        config_zero = TestTimeComputeConfig(
            max_compute_budget=0,
            min_compute_steps=1
        )
        
        def run_with_zero_budget():
            ttc_scaler = TestTimeComputeScaler(model, config_zero)
            support_x = torch.randn(6, 10)
            support_y = torch.tensor([0, 0, 1, 1, 2, 2])
            query_x = torch.randn(3, 10)
            
            with torch.no_grad():
                return ttc_scaler.scale_compute(support_x, support_y, query_x)
        
        result = TortureTestValidator.check_graceful_failure(
            run_with_zero_budget,
            expected_error_types=[ValueError, RuntimeError]
        )
        
        # Zero budget should either be handled or cause appropriate error
        if not result['failed_gracefully']:
            predictions, metrics = result['result']
            # If handled, should still produce valid outputs
            assert not torch.isnan(predictions).any(), "Zero budget produced NaN!"
        else:
            assert result['expected_error'], f"Zero budget error: {result['error_type']}"
    
    def test_impossible_confidence_threshold_handling(self):
        """
        Test behavior with impossible confidence thresholds (>1.0 or <0.0).
        This tests configuration validation.
        """
        torch.manual_seed(42)
        
        model = nn.Linear(10, 2)
        
        # Test impossible threshold
        def run_with_impossible_threshold():
            config = TestTimeComputeConfig(
                max_compute_budget=10,
                confidence_threshold=1.5  # Impossible - confidence can't exceed 1.0
            )
            ttc_scaler = TestTimeComputeScaler(model, config)
            support_x = torch.randn(4, 10)
            support_y = torch.tensor([0, 0, 1, 1])
            query_x = torch.randn(2, 10)
            
            with torch.no_grad():
                return ttc_scaler.scale_compute(support_x, support_y, query_x)
        
        result = TortureTestValidator.check_graceful_failure(
            run_with_impossible_threshold,
            expected_error_types=[ValueError, RuntimeError]
        )
        
        # Impossible thresholds should be caught during configuration
        if not result['failed_gracefully']:
            # If handled, early stopping should never trigger
            predictions, metrics = result['result']
            assert metrics.get('early_stopped', False) == False, \
                "Impossible threshold somehow triggered early stopping!"
    
    def test_model_returning_nan_handling(self):
        """
        Test TTC behavior when the base model returns NaN.
        This tests error propagation and handling.
        """
        torch.manual_seed(42)
        
        class NaNModel(nn.Module):
            def forward(self, support_x, support_y, query_x):
                # Model that returns NaN (simulates numerical issues)
                batch_size = query_x.shape[0]
                n_classes = len(torch.unique(support_y))
                output = torch.full((batch_size, n_classes), float('nan'))
                return output
        
        model = NaNModel()
        config = TestTimeComputeConfig(max_compute_budget=5, min_compute_steps=2)
        ttc_scaler = TestTimeComputeScaler(model, config)
        
        def run_with_nan_model():
            support_x = torch.randn(6, 10)
            support_y = torch.tensor([0, 0, 1, 1, 2, 2])
            query_x = torch.randn(3, 10)
            
            with torch.no_grad():
                return ttc_scaler.scale_compute(support_x, support_y, query_x)
        
        result = TortureTestValidator.check_graceful_failure(
            run_with_nan_model,
            expected_error_types=[RuntimeError, ValueError]
        )
        
        # NaN from model should be detected and handled appropriately
        if not result['failed_gracefully']:
            predictions, metrics = result['result']
            # If it handles NaN models, should contain the damage
            finite_predictions = torch.isfinite(predictions)
            print(f"NaN model handling: {finite_predictions.sum()}/{predictions.numel()} finite outputs")
        else:
            assert result['expected_error'], f"NaN model error: {result['error_type']}"
    
    def test_compute_budget_smaller_than_minimum_steps(self):
        """
        Test configuration where max_budget < min_steps.
        This is a logical impossibility that should be caught.
        """
        torch.manual_seed(42)
        
        model = nn.Linear(5, 2)
        
        def run_with_impossible_config():
            config = TestTimeComputeConfig(
                max_compute_budget=2,
                min_compute_steps=5  # Impossible: min > max
            )
            ttc_scaler = TestTimeComputeScaler(model, config)
            
            support_x = torch.randn(4, 5)
            support_y = torch.tensor([0, 0, 1, 1])
            query_x = torch.randn(2, 5)
            
            with torch.no_grad():
                return ttc_scaler.scale_compute(support_x, support_y, query_x)
        
        result = TortureTestValidator.check_graceful_failure(
            run_with_impossible_config,
            expected_error_types=[ValueError, AssertionError, RuntimeError]
        )
        
        # Impossible configuration should be rejected
        assert result['failed_gracefully'], "Impossible TTC config should be rejected!"
        assert result['expected_error'], f"Wrong error for impossible config: {result['error_type']}"


class TestDatasetEdgeCaseTorture:
    """
    TORTURE TEST: Break dataset handling with edge cases.
    Data loading is a common source of silent failures.
    """
    
    def test_empty_dataset_handling(self):
        """
        Test behavior with empty datasets.
        This should fail gracefully, not silently.
        """
        torch.manual_seed(42)
        
        def run_with_empty_dataset():
            # Empty tensors
            data = torch.empty(0, 784)
            labels = torch.empty(0, dtype=torch.long)
            
            task_config = TaskConfiguration(n_way=3, k_shot=2, q_query=5)
            dataset = MetaLearningDataset(data, labels, task_config)
            
            return dataset.sample_task()
        
        result = TortureTestValidator.check_graceful_failure(
            run_with_empty_dataset,
            expected_error_types=[ValueError, IndexError, RuntimeError]
        )
        
        # Empty dataset SHOULD cause an error
        assert result['failed_gracefully'], "Empty dataset should cause an error!"
        assert result['expected_error'], f"Wrong error for empty dataset: {result['error_type']}"
    
    def test_insufficient_samples_per_class_handling(self):
        """
        Test behavior when dataset has fewer samples per class than requested.
        This tests sampling logic robustness.
        """
        torch.manual_seed(42)
        
        def run_with_insufficient_samples():
            # Only 1 sample per class, but requesting 3-shot
            data = torch.randn(3, 784)  # 3 samples total
            labels = torch.tensor([0, 1, 2])  # 1 per class
            
            task_config = TaskConfiguration(n_way=3, k_shot=3, q_query=2)  # Need 3 per class
            dataset = MetaLearningDataset(data, labels, task_config)
            
            return dataset.sample_task()
        
        result = TortureTestValidator.check_graceful_failure(
            run_with_insufficient_samples,
            expected_error_types=[ValueError, IndexError, RuntimeError]
        )
        
        # Should either handle gracefully (with warning) or error appropriately
        if not result['failed_gracefully']:
            task = result['result']
            # If handled, check task is still valid
            assert 'support' in task and 'query' in task, "Insufficient samples broke task structure!"
        else:
            assert result['expected_error'], f"Wrong error for insufficient samples: {result['error_type']}"
    
    def test_label_range_validation(self):
        """
        Test dataset behavior with invalid label ranges.
        Labels should be validated to prevent silent failures.
        """
        torch.manual_seed(42)
        
        def run_with_invalid_labels():
            data = torch.randn(10, 784)
            labels = torch.tensor([0, 1, 2, 100, 4, 5, -1, 7, 8, 9])  # Invalid: 100, -1
            
            task_config = TaskConfiguration(n_way=3, k_shot=1, q_query=2)
            dataset = MetaLearningDataset(data, labels, task_config)
            
            return dataset.sample_task()
        
        result = TortureTestValidator.check_graceful_failure(
            run_with_invalid_labels,
            expected_error_types=[ValueError, IndexError, RuntimeError]
        )
        
        # Invalid labels should either be handled or cause appropriate errors
        if not result['failed_gracefully']:
            task = result['result']
            # If handled, labels should be valid
            support_labels = task['support']['labels']
            query_labels = task['query']['labels']
            assert support_labels.min() >= 0, "Negative labels in support set!"
            assert query_labels.min() >= 0, "Negative labels in query set!"
        else:
            print(f"Invalid labels handled: {result['error_message']}")
    
    def test_task_configuration_boundary_conditions(self):
        """
        Test task configuration with boundary conditions.
        Edge cases in configuration should be handled properly.
        """
        torch.manual_seed(42)
        
        data = torch.randn(100, 784)
        labels = torch.randint(0, 10, (100,))
        
        # Test various boundary conditions
        boundary_configs = [
            {'n_way': 1, 'k_shot': 1, 'q_query': 1},      # Minimal valid config
            {'n_way': 10, 'k_shot': 5, 'q_query': 5},     # Uses all classes
            {'n_way': 0, 'k_shot': 1, 'q_query': 1},      # Invalid: zero classes
            {'n_way': 1, 'k_shot': 0, 'q_query': 1},      # Invalid: zero shot
            {'n_way': 1, 'k_shot': 1, 'q_query': 0},      # Invalid: zero query
        ]
        
        for i, config_params in enumerate(boundary_configs):
            def run_with_boundary_config():
                task_config = TaskConfiguration(**config_params)
                dataset = MetaLearningDataset(data, labels, task_config)
                return dataset.sample_task()
            
            result = TortureTestValidator.check_graceful_failure(
                run_with_boundary_config,
                expected_error_types=[ValueError, RuntimeError]
            )
            
            if i < 2:  # First two should work
                if result['failed_gracefully']:
                    print(f"Config {config_params} failed unexpectedly: {result['error_message']}")
            else:  # Last three should fail
                assert result['failed_gracefully'], \
                    f"Invalid config {config_params} should have failed!"


class TestStatisticalEvaluationEdgeCaseTorture:
    """
    TORTURE TEST: Break statistical evaluation with edge cases.
    Statistics can be tricky with edge cases.
    """
    
    def test_identical_scores_confidence_interval(self):
        """
        Test confidence interval computation with identical scores.
        This can cause division by zero in variance calculations.
        """
        torch.manual_seed(42)
        
        config = StatsConfig(confidence_level=0.95, bootstrap_samples=100)
        stats = StatisticalAnalysis(config)
        
        # All identical scores
        identical_scores = np.array([0.8, 0.8, 0.8, 0.8, 0.8])
        
        def run_with_identical_scores():
            return stats.compute_confidence_interval(identical_scores)
        
        result = TortureTestValidator.check_graceful_failure(run_with_identical_scores)
        
        if not result['failed_gracefully']:
            ci = result['result']
            # Should handle zero variance gracefully
            assert isinstance(ci, tuple) and len(ci) == 2, "Invalid CI format!"
            assert ci[0] <= ci[1], "Invalid CI ordering!"
        else:
            print(f"Identical scores CI: {result['error_message']}")
    
    def test_single_score_statistical_analysis(self):
        """
        Test statistical analysis with single score.
        Many statistical measures are undefined with n=1.
        """
        torch.manual_seed(42)
        
        config = StatsConfig()
        stats = StatisticalAnalysis(config)
        
        def run_with_single_score():
            return stats.compute_confidence_interval(np.array([0.75]))
        
        result = TortureTestValidator.check_graceful_failure(
            run_with_single_score,
            expected_error_types=[ValueError, RuntimeError]
        )
        
        # Single score should either be handled or fail gracefully
        if not result['failed_gracefully']:
            ci = result['result'] 
            # If handled, should return something sensible
            assert ci is not None, "Single score CI shouldn't be None!"
        else:
            assert result['expected_error'], f"Single score error: {result['error_type']}"
    
    def test_extreme_confidence_levels(self):
        """
        Test confidence interval computation with extreme confidence levels.
        This tests statistical computation boundaries.
        """
        torch.manual_seed(42)
        
        scores = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
        
        extreme_levels = [0.01, 0.99, 0.999, 1.0, 0.0, -0.1, 1.1]
        
        for level in extreme_levels:
            def run_with_extreme_level():
                config = StatsConfig(confidence_level=level)
                stats = StatisticalAnalysis(config)
                return stats.compute_confidence_interval(scores)
            
            result = TortureTestValidator.check_graceful_failure(
                run_with_extreme_level,
                expected_error_types=[ValueError, RuntimeError]
            )
            
            if level < 0 or level > 1:
                # Invalid levels should fail
                assert result['failed_gracefully'], \
                    f"Invalid confidence level {level} should fail!"
            else:
                # Valid levels should work
                if result['failed_gracefully']:
                    print(f"Level {level} failed: {result['error_message']}")


class TestMemoryAndPerformanceEdgeCases:
    """
    TORTURE TEST: Test memory usage and performance edge cases.
    These can cause silent failures or crashes.
    """
    
    def test_large_tensor_handling(self):
        """
        Test behavior with very large tensors.
        This tests memory management and numerical stability.
        """
        torch.manual_seed(42)
        
        def run_with_large_tensors():
            # Create large but manageable tensors
            large_dim = 10000
            support_x = torch.randn(10, large_dim)
            support_y = torch.randint(0, 2, (10,))
            query_x = torch.randn(5, large_dim)
            
            encoder = nn.Linear(large_dim, 64)
            model = PrototypicalNetworks(encoder, {"n_way": 2})
            
            with torch.no_grad():
                return model(support_x, support_y, query_x)
        
        result = TortureTestValidator.check_graceful_failure(
            run_with_large_tensors,
            expected_error_types=[RuntimeError, MemoryError]
        )
        
        if not result['failed_gracefully']:
            output = result['result']
            assert output.shape == (5, 2), f"Large tensor output shape wrong: {output.shape}"
        else:
            # Memory constraints are acceptable
            print(f"Large tensor handling: {result['error_message']}")
    
    def test_deep_model_gradient_computation(self):
        """
        Test MAML with very deep models.
        This can cause gradient vanishing or memory issues.
        """
        torch.manual_seed(42)
        
        def run_with_deep_model():
            # Create deep model
            layers = []
            for i in range(20):  # Very deep
                layers.extend([nn.Linear(64, 64), nn.ReLU()])
            layers.append(nn.Linear(64, 2))
            
            deep_model = nn.Sequential(*layers)
            config = MAMLConfig(inner_lr=0.01, inner_steps=1)
            maml = MAMLLearner(deep_model, config)
            
            support_x = torch.randn(10, 64)
            support_y = torch.randint(0, 2, (10,))
            query_x = torch.randn(5, 64)
            query_y = torch.randint(0, 2, (5,))
            
            return maml.meta_train_step([(support_x, support_y, query_x, query_y)])
        
        result = TortureTestValidator.check_graceful_failure(
            run_with_deep_model,
            expected_error_types=[RuntimeError, MemoryError]
        )
        
        if not result['failed_gracefully']:
            meta_result = result['result']
            assert 'meta_loss' in meta_result, "Deep model meta-training failed!"
            assert np.isfinite(meta_result['meta_loss']), "Deep model produced infinite loss!"
        else:
            # Deep model limitations are acceptable
            print(f"Deep model handling: {result['error_message']}")


if __name__ == "__main__":
    # Run with high verbosity to see all torture test results
    pytest.main([__file__, "-v", "--tb=short", "--capture=no", 
                "--maxfail=10", "-x"])  # Stop on first 10 failures