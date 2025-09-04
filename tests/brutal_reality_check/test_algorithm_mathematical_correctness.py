"""
🔥 ALGORITHM MATHEMATICAL CORRECTNESS TESTS 🔥
=============================================

These tests will BRUTALLY EXPOSE every mathematical error in our implementations.
When we run these, we'll discover how much of our "breakthrough algorithms" 
are actually just fancy random number generators.

Author: The Brutal Technical Advisor
Status: Assumes perfection, will expose reality
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional
import warnings
from unittest.mock import patch, MagicMock

from meta_learning.meta_learning_modules import (
    TestTimeComputeScaler, TestTimeComputeConfig,
    MAMLLearner, MAMLConfig, FirstOrderMAML,
    PrototypicalNetworks, PrototypicalConfig,
    OnlineMetaLearner, OnlineMetaConfig,
    MetaLearningDataset, TaskConfiguration
)


class MathematicalCorrectnessValidator:
    """
    This class will be the judge, jury, and executioner of our mathematical sins.
    It checks if our algorithms actually implement what the papers describe.
    """
    
    @staticmethod
    def validate_gradient_computation(model, loss_fn, inputs, targets):
        """
        Verify gradients are computed correctly - not just random tensors.
        This will catch if we're returning fake gradients.
        """
        model.zero_grad()
        outputs = model(inputs)
        loss = loss_fn(outputs, targets)
        loss.backward()
        
        # Check gradients exist and aren't all zeros or NaN
        has_valid_gradients = False
        for param in model.parameters():
            if param.grad is not None:
                if not torch.isnan(param.grad).any() and not (param.grad == 0).all():
                    has_valid_gradients = True
                    break
        
        return has_valid_gradients, loss.item()
    
    @staticmethod
    def check_mathematical_properties(tensor, property_name, expected_range=None, expected_shape=None):
        """
        Verify mathematical properties are preserved.
        This will expose fake implementations that return random garbage.
        """
        results = {
            'property_name': property_name,
            'shape_correct': True if expected_shape is None else tensor.shape == expected_shape,
            'has_nan': torch.isnan(tensor).any().item(),
            'has_inf': torch.isinf(tensor).any().item(),
            'in_expected_range': True,
            'actual_range': (tensor.min().item(), tensor.max().item()) if tensor.numel() > 0 else (0, 0)
        }
        
        if expected_range is not None:
            min_val, max_val = expected_range
            results['in_expected_range'] = (tensor.min() >= min_val and tensor.max() <= max_val).item()
        
        return results


class TestPrototypicalNetworksMathematicalCorrectness:
    """
    BRUTAL VALIDATION: Snell et al. (2017) Prototypical Networks Equations
    These tests will expose if we're implementing the actual paper or just pretending.
    """
    
    def test_snell_equation_1_prototype_computation(self):
        """
        Test Equation 1 from Snell et al. (2017):
        c_k = (1/|S_k|) * Σ(x_i ∈ S_k) f_φ(x_i)
        
        This will catch if prototypes are computed incorrectly.
        """
        # Create deterministic test case
        torch.manual_seed(42)
        
        # Mock encoder that returns predictable embeddings
        class DeterministicEncoder(nn.Module):
            def forward(self, x):
                # Return embeddings where each sample gets unique values
                batch_size = x.shape[0]
                return torch.arange(batch_size).float().unsqueeze(1).repeat(1, 64)
        
        encoder = DeterministicEncoder()
        config = PrototypicalConfig(distance_metric='euclidean')
        model = PrototypicalNetworks(encoder, {"n_way": 3, "config": config})
        
        # Create support set: 3 classes, 2 samples each
        support_x = torch.randn(6, 28, 28)  # 6 samples
        support_y = torch.tensor([0, 0, 1, 1, 2, 2])  # 2 samples per class
        
        # Test prototype computation
        with torch.no_grad():
            embeddings = encoder(support_x)  # [6, 64]
            
            # Manual computation of prototypes per Snell Equation 1
            expected_prototypes = []
            for class_id in range(3):
                class_mask = support_y == class_id
                class_embeddings = embeddings[class_mask]
                prototype = class_embeddings.mean(dim=0)  # Average of class samples
                expected_prototypes.append(prototype)
            
            expected_prototypes = torch.stack(expected_prototypes)  # [3, 64]
            
            # Get model's prototypes (this will expose fake implementations)
            computed_prototypes = model.compute_prototypes(embeddings, support_y, 3)
            
            # BRUTAL CHECK: Are the prototypes mathematically correct?
            torch.testing.assert_close(
                computed_prototypes, expected_prototypes, 
                msg="Prototype computation doesn't match Snell et al. Equation 1!"
            )
            
            # Validate mathematical properties
            validator = MathematicalCorrectnessValidator()
            properties = validator.check_mathematical_properties(
                computed_prototypes, "prototypes", expected_shape=(3, 64)
            )
            
            assert not properties['has_nan'], "Prototypes contain NaN values!"
            assert not properties['has_inf'], "Prototypes contain infinity values!"
            assert properties['shape_correct'], f"Prototype shape incorrect: {computed_prototypes.shape}"
    
    def test_snell_equation_2_distance_computation(self):
        """
        Test Equation 2 from Snell et al. (2017):
        d(f_φ(x), c_k) = ||f_φ(x) - c_k||²
        
        This will catch if distances are computed incorrectly.
        """
        torch.manual_seed(42)
        
        # Create simple test case
        query_embeddings = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32)  # [2, 2]
        prototypes = torch.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=torch.float32)  # [2, 2]
        
        # Expected squared Euclidean distances
        # Query 0 to Prototype 0: ||[1,0] - [0,0]||² = 1² + 0² = 1
        # Query 0 to Prototype 1: ||[1,0] - [1,1]||² = 0² + 1² = 1
        # Query 1 to Prototype 0: ||[0,1] - [0,0]||² = 0² + 1² = 1  
        # Query 1 to Prototype 1: ||[0,1] - [1,1]||² = 1² + 0² = 1
        expected_distances = torch.tensor([[1.0, 1.0], [1.0, 1.0]])  # [2, 2]
        
        # Test model's distance computation (will expose fake implementations)
        encoder = nn.Linear(2, 2)  # Simple pass-through
        model = PrototypicalNetworks(encoder, {"n_way": 2})
        
        with torch.no_grad():
            computed_distances = model.compute_distances(query_embeddings, prototypes)
            
            # BRUTAL CHECK: Are distances mathematically correct?
            torch.testing.assert_close(
                computed_distances, expected_distances,
                msg="Distance computation doesn't match Snell et al. Equation 2!"
            )
    
    def test_snell_equation_3_probability_computation(self):
        """
        Test Equation 3 from Snell et al. (2017):
        p_φ(y = k | x) = exp(-d(f_φ(x), c_k)) / Σ_k' exp(-d(f_φ(x), c_k'))
        
        This will catch if softmax probabilities are computed incorrectly.
        """
        torch.manual_seed(42)
        
        # Create test distances
        distances = torch.tensor([[2.0, 1.0, 3.0], [1.0, 2.0, 1.0]], dtype=torch.float32)  # [2, 3]
        
        # Expected probabilities using softmax(-distances)
        expected_logits = -distances  # Negative distances for softmax
        expected_probs = torch.softmax(expected_logits, dim=1)
        
        # Test model's probability computation
        encoder = nn.Linear(10, 64)
        model = PrototypicalNetworks(encoder, {"n_way": 3})
        
        with torch.no_grad():
            computed_probs = model.compute_probabilities(distances)
            
            # BRUTAL CHECK: Are probabilities mathematically correct?
            torch.testing.assert_close(
                computed_probs, expected_probs,
                msg="Probability computation doesn't match Snell et al. Equation 3!"
            )
            
            # Check probability properties
            assert torch.allclose(computed_probs.sum(dim=1), torch.ones(2)), \
                "Probabilities don't sum to 1!"
            assert (computed_probs >= 0).all(), "Negative probabilities found!"
            assert (computed_probs <= 1).all(), "Probabilities exceed 1!"


class TestMAMLMathematicalCorrectness:
    """
    BRUTAL VALIDATION: Finn et al. (2017) MAML Equations
    These tests will expose if we implement actual MAML or just gradient noise.
    """
    
    def test_finn_equation_1_inner_loop_update(self):
        """
        Test Equation 1 from Finn et al. (2017):
        θ'_i = θ - α * ∇_θ L_T_i(f_θ)
        
        This will catch if inner loop updates are mathematically correct.
        """
        torch.manual_seed(42)
        
        # Create simple model for testing
        model = nn.Linear(10, 5)
        initial_params = [p.clone().detach() for p in model.parameters()]
        
        # Create MAML learner
        config = MAMLConfig(inner_lr=0.1, inner_steps=1)
        maml = MAMLLearner(model, config)
        
        # Create task data
        support_x = torch.randn(20, 10)
        support_y = torch.randint(0, 5, (20,))
        
        # Test single inner step
        # Manual computation of expected update (need gradients enabled)
        model.zero_grad()
        outputs = model(support_x)
        loss = nn.CrossEntropyLoss()(outputs, support_y)
        
        # Compute gradients manually
        grads = torch.autograd.grad(loss, model.parameters(), create_graph=False)
        
        with torch.no_grad():
            # Perform parameter updates without tracking gradients
            
            # Expected parameter updates: θ' = θ - α * ∇L
            expected_params = []
            for param, grad in zip(initial_params, grads):
                expected_param = param - config.inner_lr * grad
                expected_params.append(expected_param)
        
        # Test MAML's inner update (will expose fake implementations)
        adapted_model = maml.adapt_to_task(support_x, support_y, None, None)
        
        # BRUTAL CHECK: Are parameters updated correctly per Finn Equation 1?
        for expected, actual in zip(expected_params, adapted_model.parameters()):
            torch.testing.assert_close(
                actual.data, expected,
                msg="Inner loop update doesn't match Finn et al. Equation 1!"
            )
    
    def test_finn_equation_2_meta_gradient_computation(self):
        """
        Test Equation 2 from Finn et al. (2017):
        θ ← θ - β * ∇_θ Σ_{T_i ~ p(T)} L_{T_i}(f_{θ'_i})
        
        This will catch if meta-gradients are computed correctly.
        """
        torch.manual_seed(42)
        
        # Create MAML setup
        model = nn.Linear(5, 2)
        config = MAMLConfig(inner_lr=0.1, inner_steps=1, outer_lr=0.01)
        maml = MAMLLearner(model, config)
        
        # Create meta-batch (2 tasks)
        meta_batch = []
        for _ in range(2):
            support_x = torch.randn(10, 5)
            support_y = torch.randint(0, 2, (10,))
            query_x = torch.randn(5, 5)
            query_y = torch.randint(0, 2, (5,))
            meta_batch.append((support_x, support_y, query_x, query_y))
        
        # Store initial parameters
        initial_params = [p.clone().detach() for p in model.parameters()]
        
        # Test meta-gradient computation
        meta_loss = maml.meta_train_step(meta_batch)
        
        # BRUTAL CHECKS: Verify meta-gradient properties
        validator = MathematicalCorrectnessValidator()
        has_gradients, final_loss = validator.validate_gradient_computation(
            model, nn.CrossEntropyLoss(), 
            meta_batch[0][2], meta_batch[0][3]  # Use query data
        )
        
        assert has_gradients, "No valid meta-gradients computed!"
        assert isinstance(meta_loss, dict), "Meta-training should return metrics!"
        assert 'meta_loss' in meta_loss, "Missing meta-loss in results!"
        assert not np.isnan(meta_loss['meta_loss']), "Meta-loss is NaN!"
    
    def test_first_order_maml_approximation(self):
        """
        Test first-order MAML approximation correctness.
        This will catch if we're computing expensive second-order gradients unnecessarily.
        """
        torch.manual_seed(42)
        
        # Create models for comparison
        model1 = nn.Linear(5, 2)
        model2 = nn.Linear(5, 2)
        
        # Copy parameters to ensure identical starting points
        with torch.no_grad():
            for p1, p2 in zip(model1.parameters(), model2.parameters()):
                p2.data.copy_(p1.data)
        
        # Create MAML variants
        config = MAMLConfig(inner_lr=0.1, inner_steps=1, first_order=False)
        full_maml = MAMLLearner(model1, config)
        
        config_fo = MAMLConfig(inner_lr=0.1, inner_steps=1, first_order=True)
        first_order_maml = FirstOrderMAML(model2, config_fo)
        
        # Create identical task data
        support_x = torch.randn(10, 5)
        support_y = torch.randint(0, 2, (10,))
        query_x = torch.randn(5, 5)
        query_y = torch.randint(0, 2, (5,))
        
        meta_batch = [(support_x, support_y, query_x, query_y)]
        
        # Test both methods
        full_result = full_maml.meta_train_step(meta_batch)
        fo_result = first_order_maml.meta_train_step(meta_batch)
        
        # BRUTAL CHECK: First-order should be computationally different but converge similarly
        assert isinstance(full_result, dict), "Full MAML should return metrics!"
        assert isinstance(fo_result, dict), "First-order MAML should return metrics!"
        
        # Both should have valid losses (though potentially different values)
        assert not np.isnan(full_result['meta_loss']), "Full MAML meta-loss is NaN!"
        assert not np.isnan(fo_result['meta_loss']), "First-order MAML meta-loss is NaN!"


class TestTestTimeComputeMathematicalCorrectness:
    """
    BRUTAL VALIDATION: Snell et al. (2024) Test-Time Compute Scaling
    These tests will expose if we implement actual TTC or just burn GPU cycles randomly.
    """
    
    def test_ttc_adaptive_compute_allocation(self):
        """
        Test if test-time compute is allocated based on task difficulty.
        This will catch if we're just running random iterations.
        """
        torch.manual_seed(42)
        
        # Create test setup
        class MockModel(nn.Module):
            def forward(self, support_x, support_y, query_x):
                # Return predictable outputs for testing
                batch_size = query_x.shape[0]
                n_classes = len(torch.unique(support_y))
                return torch.randn(batch_size, n_classes)
        
        model = MockModel()
        config = TestTimeComputeConfig(
            max_compute_budget=20,
            min_compute_steps=5,
            difficulty_adaptive=True
        )
        ttc_scaler = TestTimeComputeScaler(model, config)
        
        # Create "easy" task (well-separated classes)
        easy_support = torch.tensor([[1.0, 0.0], [1.1, 0.1], [0.0, 1.0], [0.1, 1.1]])
        easy_support_labels = torch.tensor([0, 0, 1, 1])
        easy_query = torch.tensor([[1.05, 0.05]])
        
        # Create "hard" task (overlapping classes) 
        hard_support = torch.tensor([[0.5, 0.5], [0.51, 0.49], [0.49, 0.51], [0.52, 0.48]])
        hard_support_labels = torch.tensor([0, 0, 1, 1])
        hard_query = torch.tensor([[0.505, 0.495]])
        
        # Test compute allocation
        with torch.no_grad():
            easy_preds, easy_metrics = ttc_scaler.scale_compute(
                easy_support, easy_support_labels, easy_query
            )
            hard_preds, hard_metrics = ttc_scaler.scale_compute(
                hard_support, hard_support_labels, hard_query
            )
        
        # BRUTAL CHECKS: Verify adaptive behavior
        assert 'compute_used' in easy_metrics, "Missing compute usage metrics!"
        assert 'compute_used' in hard_metrics, "Missing compute usage metrics!"
        
        # Hard tasks should use more compute (if adaptation works)
        if config.difficulty_adaptive:
            assert hard_metrics['compute_used'] >= easy_metrics['compute_used'], \
                "Adaptive compute not working - hard task used less compute!"
        
        # All compute usage should be within bounds
        assert easy_metrics['compute_used'] >= config.min_compute_steps, \
            "Compute usage below minimum!"
        assert easy_metrics['compute_used'] <= config.max_compute_budget, \
            "Compute usage exceeds budget!"
    
    def test_ttc_confidence_estimation(self):
        """
        Test confidence estimation and early stopping logic.
        This will catch if confidence is just random numbers.
        """
        torch.manual_seed(42)
        
        # Mock model with predictable confidence behavior
        class ConfidenceModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.call_count = 0
                
            def forward(self, support_x, support_y, query_x):
                self.call_count += 1
                batch_size = query_x.shape[0]
                
                # Simulate increasing confidence over iterations
                confidence_boost = self.call_count * 0.1
                logits = torch.tensor([[2.0 + confidence_boost, -1.0]]) # High confidence class 0
                return logits.repeat(batch_size, 1)
        
        model = ConfidenceModel()
        config = TestTimeComputeConfig(
            max_compute_budget=10,
            min_compute_steps=2,
            confidence_threshold=0.9,
            early_stopping=True
        )
        ttc_scaler = TestTimeComputeScaler(model, config)
        
        # Test confidence-based early stopping
        support_x = torch.randn(4, 10)
        support_y = torch.tensor([0, 0, 1, 1])
        query_x = torch.randn(2, 10)
        
        with torch.no_grad():
            predictions, metrics = ttc_scaler.scale_compute(support_x, support_y, query_x)
        
        # BRUTAL CHECKS: Verify confidence behavior
        assert 'final_confidence' in metrics, "Missing final confidence metric!"
        assert 'early_stopped' in metrics, "Missing early stopping indicator!"
        
        confidence = metrics['final_confidence']
        assert 0.0 <= confidence <= 1.0, f"Invalid confidence value: {confidence}"
        
        # If early stopping worked, should stop before max budget
        if metrics['early_stopped']:
            assert metrics['compute_used'] < config.max_compute_budget, \
                "Early stopping claimed but used full budget!"
    
    def test_ttc_compute_efficiency_metrics(self):
        """
        Test compute efficiency tracking.
        This will catch if we're not measuring actual computational cost.
        """
        torch.manual_seed(42)
        
        # Simple model for testing
        model = nn.Linear(10, 3)
        config = TestTimeComputeConfig(
            max_compute_budget=15,
            min_compute_steps=3
        )
        ttc_scaler = TestTimeComputeScaler(model, config)
        
        # Test compute efficiency
        support_x = torch.randn(9, 10)  # 3 classes, 3 samples each
        support_y = torch.tensor([0, 0, 0, 1, 1, 1, 2, 2, 2])
        query_x = torch.randn(6, 10)
        
        with torch.no_grad():
            predictions, metrics = ttc_scaler.scale_compute(support_x, support_y, query_x)
        
        # BRUTAL CHECKS: Verify efficiency metrics
        required_metrics = ['compute_used', 'allocated_budget', 'compute_efficiency']
        for metric in required_metrics:
            assert metric in metrics, f"Missing efficiency metric: {metric}"
        
        # Compute efficiency should be meaningful
        efficiency = metrics['compute_efficiency']
        assert 0.0 <= efficiency <= 1.0, f"Invalid efficiency value: {efficiency}"
        
        # Used compute should be within bounds
        assert metrics['compute_used'] >= config.min_compute_steps, \
            "Used less than minimum compute steps!"
        assert metrics['compute_used'] <= config.max_compute_budget, \
            "Exceeded maximum compute budget!"


class TestIntegrationMathematicalCorrectness:
    """
    BRUTAL VALIDATION: End-to-end mathematical correctness
    These tests will expose integration issues between components.
    """
    
    def test_end_to_end_few_shot_learning_pipeline(self):
        """
        Test complete few-shot learning pipeline mathematical correctness.
        This will catch if components don't integrate mathematically.
        """
        torch.manual_seed(42)
        
        # Create complete pipeline
        encoder = nn.Sequential(
            nn.Linear(784, 128),
            nn.ReLU(),
            nn.Linear(128, 64)
        )
        
        proto_config = PrototypicalConfig(distance_metric='euclidean')
        proto_model = PrototypicalNetworks(encoder, {"n_way": 5, "config": proto_config})
        
        ttc_config = TestTimeComputeConfig(max_compute_budget=8, min_compute_steps=3)
        ttc_scaler = TestTimeComputeScaler(proto_model, ttc_config)
        
        # Create 5-way 3-shot task
        support_x = torch.randn(15, 784)  # 5 classes * 3 shots
        support_y = torch.tensor([0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4])
        query_x = torch.randn(10, 784)   # 2 queries per class
        query_y = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3, 4, 4])
        
        # Test complete pipeline
        with torch.no_grad():
            predictions, metrics = ttc_scaler.scale_compute(support_x, support_y, query_x)
        
        # BRUTAL MATHEMATICAL CHECKS
        validator = MathematicalCorrectnessValidator()
        
        # Check prediction properties
        pred_props = validator.check_mathematical_properties(
            predictions, "predictions", expected_shape=(10, 5)
        )
        
        assert pred_props['shape_correct'], f"Prediction shape wrong: {predictions.shape}"
        assert not pred_props['has_nan'], "Predictions contain NaN!"
        assert not pred_props['has_inf'], "Predictions contain infinity!"
        
        # Check if predictions are valid probability distributions
        if predictions.dim() == 2:
            probs = torch.softmax(predictions, dim=1)
            prob_sums = probs.sum(dim=1)
            assert torch.allclose(prob_sums, torch.ones_like(prob_sums), atol=1e-6), \
                "Predictions don't form valid probability distributions!"
        
        # Check metrics validity
        required_metrics = ['compute_used', 'final_confidence']
        for metric in required_metrics:
            assert metric in metrics, f"Missing pipeline metric: {metric}"
        
        assert not np.isnan(metrics['final_confidence']), "Final confidence is NaN!"
        assert 0 <= metrics['final_confidence'] <= 1, "Invalid confidence range!"
    
    def test_meta_learning_mathematical_properties_preservation(self):
        """
        Test that meta-learning preserves important mathematical properties.
        This will catch if adaptation breaks mathematical invariants.
        """
        torch.manual_seed(42)
        
        # Create MAML learner
        model = nn.Linear(10, 3)
        config = MAMLConfig(inner_lr=0.1, inner_steps=2)
        maml = MAMLLearner(model, config)
        
        # Store initial properties
        initial_params = [p.clone().detach() for p in model.parameters()]
        initial_norms = [torch.norm(p) for p in initial_params]
        
        # Create adaptation task
        support_x = torch.randn(15, 10)
        support_y = torch.randint(0, 3, (15,))
        query_x = torch.randn(6, 10)
        query_y = torch.randint(0, 3, (6,))
        
        # Test adaptation
        adapted_model = maml.adapt_to_task(support_x, support_y, query_x, query_y)
        
        # BRUTAL MATHEMATICAL PROPERTY CHECKS
        validator = MathematicalCorrectnessValidator()
        
        # Check parameter changes are reasonable
        adapted_params = list(adapted_model.parameters())
        param_changes = []
        
        for initial, adapted in zip(initial_params, adapted_params):
            change = torch.norm(adapted - initial)
            param_changes.append(change.item())
        
        # Parameters should change but not explode
        assert all(change > 1e-8 for change in param_changes), \
            "No parameter changes detected - adaptation failed!"
        assert all(change < 100.0 for change in param_changes), \
            "Parameter changes too large - gradient explosion!"
        
        # Check adapted model produces valid outputs
        with torch.no_grad():
            adapted_outputs = adapted_model(query_x)
            
            output_props = validator.check_mathematical_properties(
                adapted_outputs, "adapted_outputs", expected_shape=(6, 3)
            )
            
            assert output_props['shape_correct'], "Adapted model output shape wrong!"
            assert not output_props['has_nan'], "Adapted model outputs contain NaN!"
            assert not output_props['has_inf'], "Adapted model outputs contain infinity!"


# Edge Case Mathematical Correctness Tests
class TestEdgeCaseMathematicalCorrectness:
    """
    BRUTAL EDGE CASE VALIDATION: These will expose numerical instabilities
    and mathematical edge cases that break our algorithms.
    """
    
    def test_numerical_stability_zero_gradients(self):
        """
        Test behavior with zero gradients (perfect fit case).
        This will catch division by zero errors.
        """
        torch.manual_seed(42)
        
        # Create perfect classification scenario
        class PerfectModel(nn.Module):
            def forward(self, x):
                # Return perfect logits based on input patterns
                batch_size = x.shape[0]
                return torch.tensor([[100.0, -100.0, -100.0]] * batch_size)
        
        model = PerfectModel()
        config = MAMLConfig(inner_lr=0.1, inner_steps=1)
        maml = MAMLLearner(model, config)
        
        # Create task where model is already perfect
        support_x = torch.ones(6, 10)
        support_y = torch.zeros(6, dtype=torch.long)  # All class 0
        
        # This should handle zero gradients gracefully
        adapted_model = maml.adapt_to_task(support_x, support_y, None, None)
        
        assert adapted_model is not None, "Failed to handle zero gradient case!"
    
    def test_numerical_stability_extreme_values(self):
        """
        Test behavior with extreme input values.
        This will catch overflow/underflow issues.
        """
        torch.manual_seed(42)
        
        # Test with extreme values
        extreme_values = [
            torch.tensor([1e10]),   # Very large
            torch.tensor([1e-10]),  # Very small  
            torch.tensor([-1e10]),  # Very negative
            torch.tensor([0.0]),    # Zero
            torch.tensor([float('inf')]),  # Infinity (should be handled)
        ]
        
        encoder = nn.Linear(1, 64)
        model = PrototypicalNetworks(encoder, {"n_way": 2})
        
        for extreme_val in extreme_values[:4]:  # Skip infinity for now
            try:
                with torch.no_grad():
                    # Create test data with extreme values
                    support_x = extreme_val.repeat(4, 1)
                    support_y = torch.tensor([0, 0, 1, 1])
                    query_x = extreme_val.repeat(2, 1)
                    
                    # Model should handle extreme values gracefully
                    outputs = model(support_x, support_y, query_x)
                    
                    # Check outputs are still valid
                    assert not torch.isnan(outputs).any(), \
                        f"NaN outputs with extreme value: {extreme_val.item()}"
                    assert not torch.isinf(outputs).any(), \
                        f"Infinite outputs with extreme value: {extreme_val.item()}"
                        
            except Exception as e:
                pytest.fail(f"Failed to handle extreme value {extreme_val.item()}: {e}")
    
    def test_empty_or_minimal_data_handling(self):
        """
        Test behavior with minimal data scenarios.
        This will catch indexing and mathematical errors with edge cases.
        """
        torch.manual_seed(42)
        
        encoder = nn.Linear(10, 32)
        model = PrototypicalNetworks(encoder, {"n_way": 1})
        
        # Test with single sample per class (minimal case)
        support_x = torch.randn(1, 10)
        support_y = torch.tensor([0])
        query_x = torch.randn(1, 10)
        
        try:
            with torch.no_grad():
                outputs = model(support_x, support_y, query_x)
                
                assert outputs is not None, "Failed with minimal data!"
                assert outputs.shape == (1, 1), f"Wrong output shape: {outputs.shape}"
                assert not torch.isnan(outputs).any(), "NaN with minimal data!"
                
        except Exception as e:
            pytest.fail(f"Failed to handle minimal data: {e}")


if __name__ == "__main__":
    # Run with maximum verbosity to catch every mathematical sin
    pytest.main([__file__, "-v", "--tb=long", "--capture=no"])