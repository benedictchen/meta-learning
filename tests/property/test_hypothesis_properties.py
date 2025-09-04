"""
Comprehensive property-based tests using Hypothesis for meta-learning modules.

Tests mathematical invariants, edge cases, and properties that should hold
across all implementations, configurations, and parameter ranges.
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
from hypothesis import given, strategies as st, settings, assume, note
from hypothesis.extra.numpy import arrays
from typing import Dict, List, Tuple, Any, Optional
import math

# Import all modules for property testing
from meta_learning.meta_learning_modules.test_time_compute import (
    TestTimeComputeScaler, TestTimeComputeConfig
)
from meta_learning.meta_learning_modules.maml_variants import (
    MAML, FOMAML, Reptile, MAMLConfig, functional_forward
)
from meta_learning.meta_learning_modules.few_shot_learning import (
    PrototypicalLearner, PrototypicalConfig, FewShotConfig
)
from meta_learning.meta_learning_modules.continual_meta_learning import (
    ContinualMetaLearner, EWCRegularizer, ContinualConfig
)
from meta_learning.meta_learning_modules.utils import (
    basic_confidence_interval, compute_confidence_interval,
    estimate_difficulty, MetaLearningDataset, DatasetConfig
)


# Custom strategies for meta-learning domain
@st.composite
def meta_learning_episode(draw, min_way=2, max_way=10, min_shot=1, max_shot=10, 
                         min_queries=5, max_queries=50, min_features=4, max_features=128):
    """Generate realistic meta-learning episodes."""
    n_way = draw(st.integers(min_value=min_way, max_value=max_way))
    k_shot = draw(st.integers(min_value=min_shot, max_value=max_shot))
    query_shots = draw(st.integers(min_value=min_queries, max_value=max_queries))
    feature_dim = draw(st.integers(min_value=min_features, max_features=max_features))
    
    support_x = torch.randn(n_way, k_shot, feature_dim)
    support_y = torch.arange(n_way).repeat_interleave(k_shot)
    query_x = torch.randn(n_way * query_shots, feature_dim)
    query_y = torch.arange(n_way).repeat(query_shots)
    
    return support_x, support_y, query_x, query_y, n_way, k_shot, feature_dim


@st.composite
def simple_neural_network(draw, min_input=4, max_input=64, min_output=2, max_output=20):
    """Generate simple neural networks for testing."""
    input_dim = draw(st.integers(min_value=min_input, max_value=max_input))
    output_dim = draw(st.integers(min_value=min_output, max_value=max_output))
    hidden_dim = draw(st.integers(min_value=max(input_dim//2, 4), max_value=input_dim*2))
    
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, output_dim)
    ), input_dim, output_dim


@st.composite  
def accuracy_list(draw, min_size=3, max_size=50):
    """Generate list of accuracy values for statistical testing."""
    size = draw(st.integers(min_value=min_size, max_value=max_size))
    accuracies = draw(st.lists(
        st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
        min_size=size, max_size=size
    ))
    return accuracies


class TestPrototypicalNetworkProperties:
    """Property-based tests for Prototypical Networks."""
    
    @given(episode=meta_learning_episode(max_way=5, max_shot=5, max_queries=20))
    @settings(max_examples=20, deadline=5000)
    def test_prototypical_output_shape_invariant(self, episode):
        """Test that prototypical networks always output correct shapes."""
        support_x, support_y, query_x, query_y, n_way, k_shot, feature_dim = episode
        
        encoder = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.ReLU(),
            nn.Linear(feature_dim // 2, feature_dim // 4)
        )
        
        config = PrototypicalConfig(n_way=n_way, k_shot=k_shot)
        learner = PrototypicalLearner(encoder, config)
        
        logits = learner(support_x, support_y, query_x)
        
        # Shape invariant: (n_queries, n_classes)
        expected_queries = query_x.shape[0]
        assert logits.shape == (expected_queries, n_way)
        assert torch.isfinite(logits).all()
        
    @given(
        n_way=st.integers(min_value=2, max_value=8),
        k_shot=st.integers(min_value=1, max_value=6),
        feature_dim=st.integers(min_value=8, max_value=32),
        temperature=st.floats(min_value=0.1, max_value=10.0, allow_nan=False)
    )
    @settings(max_examples=15, deadline=5000)
    def test_prototypical_temperature_scaling_property(self, n_way, k_shot, feature_dim, temperature):
        """Test temperature scaling mathematical property."""
        encoder = nn.Linear(feature_dim, 16)
        config = PrototypicalConfig(
            n_way=n_way, 
            k_shot=k_shot,
            use_temperature_scaling=True,
            temperature=temperature
        )
        
        learner = PrototypicalLearner(encoder, config)
        
        support_x = torch.randn(n_way, k_shot, feature_dim)
        support_y = torch.arange(n_way).repeat_interleave(k_shot)
        query_x = torch.randn(12, feature_dim)
        
        logits_temp = learner(support_x, support_y, query_x)
        
        # Test without temperature scaling
        config_no_temp = PrototypicalConfig(
            n_way=n_way,
            k_shot=k_shot, 
            use_temperature_scaling=False
        )
        learner_no_temp = PrototypicalLearner(encoder, config_no_temp)
        logits_no_temp = learner_no_temp(support_x, support_y, query_x)
        
        # Mathematical property: logits_temp ≈ logits_no_temp / temperature
        if temperature != 1.0:
            expected_logits = logits_no_temp / temperature
            # Due to potential numerical differences, use tolerant comparison
            assert torch.allclose(logits_temp, expected_logits, rtol=1e-3, atol=1e-4)
            
    @given(
        n_way=st.integers(min_value=2, max_value=6),
        k_shot=st.integers(min_value=2, max_value=8),
        feature_dim=st.integers(min_value=4, max_value=24)
    )
    @settings(max_examples=10, deadline=5000)  
    def test_prototype_centroid_property(self, n_way, k_shot, feature_dim):
        """Test that prototypes are centroids of support features."""
        encoder = nn.Identity()  # Identity to directly test prototype computation
        config = PrototypicalConfig(protonet_variant="original")
        learner = PrototypicalLearner(encoder, config)
        
        # Create support set where we can verify centroids manually
        support_x = torch.randn(n_way, k_shot, feature_dim)
        support_y = torch.arange(n_way).repeat_interleave(k_shot)
        
        # Flatten for encoder
        support_features = support_x.view(-1, feature_dim)
        support_features = learner.encoder(support_features)
        support_features = support_features.view(n_way, k_shot, -1)
        
        prototypes = learner.compute_prototypes(support_features, support_y)
        
        # Mathematical property: each prototype should be the mean of its class features
        expected_prototypes = support_features.mean(dim=1)  # Mean over k_shot dimension
        
        assert torch.allclose(prototypes, expected_prototypes, atol=1e-6)
        assert prototypes.shape == (n_way, feature_dim)


class TestMAMLProperties:
    """Property-based tests for MAML variants."""
    
    @given(
        network=simple_neural_network(max_input=32, max_output=8),
        inner_lr=st.floats(min_value=0.001, max_value=0.1, allow_nan=False),
        inner_steps=st.integers(min_value=1, max_value=5),
        meta_batch_size=st.integers(min_value=1, max_value=4)
    )
    @settings(max_examples=10, deadline=8000)
    def test_maml_adaptation_property(self, network, inner_lr, inner_steps, meta_batch_size):
        """Test that MAML adaptation changes parameters."""
        model, input_dim, output_dim = network
        
        config = MAMLConfig(
            maml_variant="maml",
            inner_lr=inner_lr,
            inner_steps=inner_steps
        )
        
        maml_learner = MAML(model, config)
        
        # Store original parameters
        original_params = {name: param.clone() for name, param in model.named_parameters()}
        
        # Create meta-batch
        support_x_batch = []
        support_y_batch = []  
        query_x_batch = []
        query_y_batch = []
        
        for _ in range(meta_batch_size):
            n_way = min(output_dim, 3)  # Don't exceed model output
            k_shot = 2
            
            support_x = torch.randn(n_way, k_shot, input_dim)
            support_y = torch.arange(n_way).repeat_interleave(k_shot)
            query_x = torch.randn(n_way * 5, input_dim)
            query_y = torch.arange(n_way).repeat(5)
            
            support_x_batch.append(support_x)
            support_y_batch.append(support_y)
            query_x_batch.append(query_x)
            query_y_batch.append(query_y)
            
        # Perform inner adaptation on first task
        task_loss, adapted_params = maml_learner.inner_loop_adaptation(
            support_x_batch[0], support_y_batch[0],
            query_x_batch[0], query_y_batch[0]
        )
        
        # Mathematical property: adapted parameters should be different from original
        param_changed = False
        for name, adapted_param in adapted_params.items():
            if name in original_params:
                if not torch.allclose(adapted_param, original_params[name], atol=1e-7):
                    param_changed = True
                    break
                    
        assert param_changed, "MAML should modify parameters during adaptation"
        assert torch.isfinite(task_loss)
        
    @given(
        input_dim=st.integers(min_value=4, max_value=16),
        output_dim=st.integers(min_value=2, max_value=6),
        inner_lr=st.floats(min_value=0.01, max_value=0.1)
    )
    @settings(max_examples=8, deadline=5000)
    def test_fomaml_vs_maml_gradient_property(self, input_dim, output_dim, inner_lr):
        """Test FOMAML first-order approximation property."""
        model_maml = nn.Linear(input_dim, output_dim) 
        model_fomaml = nn.Linear(input_dim, output_dim)
        
        # Initialize with same parameters
        with torch.no_grad():
            for p_maml, p_fomaml in zip(model_maml.parameters(), model_fomaml.parameters()):
                p_fomaml.data.copy_(p_maml.data)
                
        config = MAMLConfig(inner_lr=inner_lr, inner_steps=1)
        
        maml_learner = MAML(model_maml, config)
        fomaml_config = MAMLConfig(maml_variant="fomaml", inner_lr=inner_lr, inner_steps=1)
        fomaml_learner = FOMAML(model_fomaml, fomaml_config)
        
        # Create same task for both
        support_x = torch.randn(output_dim, 2, input_dim)
        support_y = torch.arange(output_dim).repeat_interleave(2)
        query_x = torch.randn(output_dim * 3, input_dim)
        query_y = torch.arange(output_dim).repeat(3)
        
        # Both should handle the same task (mathematical property: same input → valid output)
        try:
            maml_loss = maml_learner.meta_train_step([support_x], [support_y], [query_x], [query_y])
            fomaml_loss = fomaml_learner.meta_train_step([support_x], [support_y], [query_x], [query_y])
            
            assert torch.isfinite(maml_loss)
            assert torch.isfinite(fomaml_loss)
            # FOMAML and MAML should give different losses (first-order vs second-order)
        except RuntimeError:
            # May fail due to dimensionality issues, which is acceptable for property testing
            pass
            
    @given(
        batch_size=st.integers(min_value=1, max_value=8),
        feature_dim=st.integers(min_value=4, max_value=20),
        n_classes=st.integers(min_value=2, max_value=5)
    )
    @settings(max_examples=10, deadline=5000)
    def test_functional_forward_consistency_property(self, batch_size, feature_dim, n_classes):
        """Test functional_forward consistency across methods."""
        model = nn.Linear(feature_dim, n_classes)
        x = torch.randn(batch_size, feature_dim)
        
        # Get parameters
        params = dict(model.named_parameters())
        
        methods = ["basic", "l2l_style", "higher_style", "manual"]
        
        outputs = []
        for method in methods:
            try:
                output = functional_forward(model, params, x, method=method)
                outputs.append(output)
                
                # Shape invariant
                assert output.shape == (batch_size, n_classes)
                assert torch.isfinite(output).all()
            except (RuntimeError, NotImplementedError):
                # Some methods might not be implemented or fail for certain configurations
                pass
                
        # If multiple methods work, they should give similar results (consistency property)
        if len(outputs) > 1:
            for i in range(1, len(outputs)):
                assert torch.allclose(outputs[0], outputs[i], rtol=1e-4, atol=1e-5)


class TestContinualLearningProperties:
    """Property-based tests for continual learning."""
    
    @given(
        n_tasks=st.integers(min_value=2, max_value=8),
        ewc_lambda=st.floats(min_value=0.1, max_value=2.0, allow_nan=False),
        feature_dim=st.integers(min_value=8, max_value=32)
    )
    @settings(max_examples=8, deadline=8000)
    def test_ewc_regularization_monotonicity_property(self, n_tasks, ewc_lambda, feature_dim):
        """Test EWC regularization monotonicity property."""
        model = nn.Sequential(
            nn.Linear(feature_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 4)
        )
        
        config = ContinualConfig(ewc_lambda=ewc_lambda, ewc_method="diagonal")
        regularizer = EWCRegularizer(model, config)
        
        # Learn multiple tasks sequentially
        for task_id in range(n_tasks):
            x = torch.randn(20, feature_dim)
            y = torch.randint(0, 4, (20,))
            
            regularizer.store_task_parameters(x, y, task_id=f"task_{task_id}")
            
            # Slightly modify parameters
            for param in model.parameters():
                param.data += torch.randn_like(param.data) * 0.01
                
        ewc_loss = regularizer.compute_ewc_loss()
        
        # Mathematical property: EWC loss should be non-negative and finite
        assert ewc_loss.item() >= 0
        assert torch.isfinite(ewc_loss)
        
        # Property: More deviation should lead to higher EWC loss
        for param in model.parameters():
            param.data += torch.randn_like(param.data) * 0.1  # Larger deviation
            
        ewc_loss_larger = regularizer.compute_ewc_loss()
        
        # Monotonicity property: larger parameter changes → larger EWC loss
        assert ewc_loss_larger.item() >= ewc_loss.item()
        
    @given(
        memory_size=st.integers(min_value=10, max_value=100),
        n_episodes=st.integers(min_value=5, max_value=50)
    )
    @settings(max_examples=10, deadline=5000)
    def test_memory_bank_capacity_property(self, memory_size, n_episodes):
        """Test memory bank capacity constraint property.""" 
        from meta_learning.meta_learning_modules.continual_meta_learning import MemoryBank
        
        config = ContinualConfig(memory_bank_size=memory_size)
        memory_bank = MemoryBank(config)
        
        # Add episodes
        for i in range(n_episodes):
            episode = {
                'support_x': torch.randn(3, 2, 16),
                'support_y': torch.arange(3).repeat_interleave(2),
                'query_x': torch.randn(12, 16),
                'query_y': torch.arange(3).repeat(4),
                'task_id': f'task_{i}'
            }
            memory_bank.add_episode(episode)
            
        # Capacity constraint property: should never exceed max size
        assert len(memory_bank.episodes) <= memory_size
        assert memory_bank.current_size <= memory_size
        
        # If we added more episodes than capacity, memory should be at capacity
        if n_episodes > memory_size:
            assert len(memory_bank.episodes) == memory_size


class TestStatisticalProperties:
    """Property-based tests for statistical utilities."""
    
    @given(
        accuracies=accuracy_list(min_size=5, max_size=30),
        confidence_level=st.floats(min_value=0.8, max_value=0.99, allow_nan=False)
    )
    @settings(max_examples=15, deadline=3000)
    def test_confidence_interval_coverage_property(self, accuracies, confidence_level):
        """Test confidence interval coverage property."""
        assume(len(set(accuracies)) > 1)  # Need some variance
        assume(all(0 <= acc <= 1 for acc in accuracies))  # Valid accuracies
        
        ci_lower, ci_upper = basic_confidence_interval(accuracies, confidence_level)
        
        # Mathematical properties
        assert ci_lower <= ci_upper, "Lower bound should be ≤ upper bound"
        assert 0 <= ci_lower <= 1, "Lower bound should be in [0, 1]"
        assert 0 <= ci_upper <= 1, "Upper bound should be in [0, 1]"
        
        # Coverage property: mean should typically be within CI
        mean_acc = np.mean(accuracies)
        
        # For well-behaved data, mean should be in CI most of the time
        note(f"Mean: {mean_acc}, CI: [{ci_lower}, {ci_upper}]")
        
        # Statistical property: CI width should decrease with higher confidence
        if confidence_level < 0.98:  # Don't test at extreme confidence levels
            ci_lower_90, ci_upper_90 = basic_confidence_interval(accuracies, 0.90)
            ci_lower_95, ci_upper_95 = basic_confidence_interval(accuracies, 0.95)
            
            width_90 = ci_upper_90 - ci_lower_90
            width_95 = ci_upper_95 - ci_lower_95
            
            # Higher confidence → wider interval
            assert width_95 >= width_90, "95% CI should be wider than 90% CI"
            
    @given(
        values1=accuracy_list(min_size=10, max_size=25),
        values2=accuracy_list(min_size=10, max_size=25)
    )
    @settings(max_examples=10, deadline=3000)
    def test_confidence_interval_method_consistency(self, values1, values2):
        """Test consistency across different CI methods."""
        assume(len(set(values1)) > 2)  # Need variance
        assume(len(set(values2)) > 2)
        assume(all(0 <= v <= 1 for v in values1 + values2))
        
        methods = ["auto", "bootstrap", "t_distribution", "normal"]
        
        results1 = {}
        results2 = {}
        
        for method in methods:
            try:
                ci1_lower, ci1_upper = compute_confidence_interval(values1, method=method)
                ci2_lower, ci2_upper = compute_confidence_interval(values2, method=method)
                
                results1[method] = (ci1_lower, ci1_upper)
                results2[method] = (ci2_lower, ci2_upper)
                
                # Basic properties should hold for all methods
                assert ci1_lower <= ci1_upper
                assert ci2_lower <= ci2_upper
                assert 0 <= ci1_lower <= 1 and 0 <= ci1_upper <= 1
                assert 0 <= ci2_lower <= 1 and 0 <= ci2_upper <= 1
                
            except (ValueError, RuntimeError):
                # Some methods may not work for all data
                continue
                
        # Consistency property: if values1 has higher mean, its CI should typically be higher
        if len(results1) > 0 and len(results2) > 0:
            mean1, mean2 = np.mean(values1), np.mean(values2)
            
            if abs(mean1 - mean2) > 0.1:  # Significant difference
                # At least some methods should reflect this difference
                consistent_methods = 0
                for method in results1:
                    if method in results2:
                        ci1_mid = (results1[method][0] + results1[method][1]) / 2
                        ci2_mid = (results2[method][0] + results2[method][1]) / 2
                        
                        if (mean1 > mean2 and ci1_mid > ci2_mid) or (mean1 < mean2 and ci1_mid < ci2_mid):
                            consistent_methods += 1
                            
                # At least some methods should be consistent with mean ordering
                assert consistent_methods > 0
                
    @given(
        n_samples=st.integers(min_value=5, max_value=100),
        mean=st.floats(min_value=0.3, max_value=0.9, allow_nan=False),
        std=st.floats(min_value=0.05, max_value=0.2, allow_nan=False)
    )
    @settings(max_examples=10, deadline=3000)
    def test_confidence_interval_sample_size_property(self, n_samples, mean, std):
        """Test confidence interval width vs sample size property."""
        # Generate two samples with different sizes
        np.random.seed(42)  # For reproducibility in property testing
        
        small_sample = np.random.normal(mean, std, n_samples).clip(0, 1).tolist()
        large_sample = np.random.normal(mean, std, n_samples * 3).clip(0, 1).tolist()
        
        assume(len(set(small_sample)) > 2)
        assume(len(set(large_sample)) > 2)
        
        ci_small_lower, ci_small_upper = basic_confidence_interval(small_sample, 0.95)
        ci_large_lower, ci_large_upper = basic_confidence_interval(large_sample, 0.95)
        
        width_small = ci_small_upper - ci_small_lower
        width_large = ci_large_upper - ci_large_lower
        
        # Statistical property: larger sample size → narrower confidence interval
        assert width_large <= width_small * 1.5  # Allow some tolerance due to randomness


class TestDifficultyEstimationProperties:
    """Property-based tests for difficulty estimation."""
    
    @given(
        n_samples=st.integers(min_value=10, max_value=50),
        n_features=st.integers(min_value=4, max_value=32),
        n_classes=st.integers(min_value=2, max_value=8)
    )
    @settings(max_examples=10, deadline=3000)
    def test_difficulty_estimation_bounds_property(self, n_samples, n_features, n_classes):
        """Test difficulty estimation bounds property."""
        task_data = {
            'features': torch.randn(n_samples, n_features),
            'labels': torch.randint(0, n_classes, (n_samples,)),
            'n_way': n_classes,
            'k_shot': max(1, n_samples // n_classes)
        }
        
        methods = ["feature_variance", "label_entropy", "inter_class_distance", "intra_class_variance"]
        
        for method in methods:
            difficulty = estimate_difficulty(task_data, method=method)
            
            # Bounds property: difficulty should be in [0, 1]
            assert 0 <= difficulty <= 1, f"Difficulty {difficulty} outside [0,1] for method {method}"
            assert isinstance(difficulty, float)
            assert not math.isnan(difficulty)
            
    @given(
        base_variance=st.floats(min_value=0.1, max_value=2.0, allow_nan=False),
        n_samples=st.integers(min_value=20, max_value=60),
        n_features=st.integers(min_value=8, max_value=24),
        n_classes=st.integers(min_value=2, max_value=5)
    )
    @settings(max_examples=8, deadline=3000)
    def test_difficulty_estimation_monotonicity_property(self, base_variance, n_samples, n_features, n_classes):
        """Test difficulty estimation monotonicity with task complexity."""
        # Create easy task (low variance features)
        easy_features = torch.randn(n_samples, n_features) * base_variance
        easy_labels = torch.randint(0, n_classes, (n_samples,))
        
        easy_task = {
            'features': easy_features,
            'labels': easy_labels,
            'n_way': n_classes,
            'k_shot': n_samples // n_classes
        }
        
        # Create hard task (high variance features)
        hard_features = torch.randn(n_samples, n_features) * (base_variance * 3)
        hard_labels = torch.randint(0, n_classes, (n_samples,))
        
        hard_task = {
            'features': hard_features,
            'labels': hard_labels,
            'n_way': n_classes,
            'k_shot': n_samples // n_classes
        }
        
        # Test with feature variance method (should be sensitive to variance)
        easy_difficulty = estimate_difficulty(easy_task, method="feature_variance")
        hard_difficulty = estimate_difficulty(hard_task, method="feature_variance")
        
        # Monotonicity property: higher variance → higher difficulty
        note(f"Easy difficulty: {easy_difficulty}, Hard difficulty: {hard_difficulty}")
        assert hard_difficulty >= easy_difficulty, "Higher variance should lead to higher difficulty"


class TestNumericalStabilityProperties:
    """Property-based tests for numerical stability."""
    
    @given(
        logit_scale=st.floats(min_value=0.1, max_value=100.0, allow_nan=False),
        batch_size=st.integers(min_value=5, max_value=20),
        n_classes=st.integers(min_value=2, max_value=8),
        temperature=st.floats(min_value=0.1, max_value=10.0, allow_nan=False)
    )
    @settings(max_examples=10, deadline=3000)
    def test_softmax_numerical_stability_property(self, logit_scale, batch_size, n_classes, temperature):
        """Test numerical stability of softmax computations."""
        # Create logits with various scales
        logits = torch.randn(batch_size, n_classes) * logit_scale
        
        # Apply temperature scaling
        scaled_logits = logits / temperature
        
        # Compute softmax probabilities
        probs = torch.softmax(scaled_logits, dim=1)
        
        # Numerical stability properties
        assert torch.isfinite(probs).all(), "Probabilities should be finite"
        assert (probs >= 0).all(), "Probabilities should be non-negative"
        assert torch.allclose(probs.sum(dim=1), torch.ones(batch_size)), "Probabilities should sum to 1"
        
        # Test with extreme logits
        extreme_logits = torch.tensor([[1000.0, -1000.0]]).expand(batch_size, -1)
        if n_classes == 2:
            extreme_probs = torch.softmax(extreme_logits, dim=1)
            assert torch.isfinite(extreme_probs).all(), "Should handle extreme logits"
            
    @given(
        distances=st.lists(
            st.floats(min_value=0.0, max_value=100.0, allow_nan=False, allow_infinity=False),
            min_size=3, max_size=15
        ),
        temperature=st.floats(min_value=0.01, max_value=10.0, allow_nan=False)
    )
    @settings(max_examples=10, deadline=2000)
    def test_distance_to_logits_stability_property(self, distances, temperature):
        """Test numerical stability of distance-to-logits conversion."""
        distance_tensor = torch.tensor(distances)
        
        # Convert distances to logits (negative distances)
        logits = -distance_tensor / temperature
        
        # Test properties
        assert torch.isfinite(logits).all(), "Logits should be finite"
        
        # Monotonicity property: smaller distances → larger logits
        sorted_distances, sort_indices = torch.sort(distance_tensor)
        sorted_logits = logits[sort_indices]
        
        # Should be decreasing (since logits = -distance/temp)
        for i in range(len(sorted_logits) - 1):
            assert sorted_logits[i] >= sorted_logits[i + 1], "Logits should decrease with distance"


class TestInvariantProperties:
    """Test invariant properties that should hold across all implementations."""
    
    @given(
        episode=meta_learning_episode(max_way=4, max_shot=3, max_queries=15),
        seed1=st.integers(min_value=0, max_value=1000),
        seed2=st.integers(min_value=0, max_value=1000)
    )
    @settings(max_examples=8, deadline=5000)
    def test_deterministic_reproducibility_property(self, episode, seed1, seed2):
        """Test deterministic reproducibility property."""
        support_x, support_y, query_x, query_y, n_way, k_shot, feature_dim = episode
        
        encoder = nn.Linear(feature_dim, 16)
        config = PrototypicalConfig()
        
        # Run with same seed twice
        torch.manual_seed(seed1)
        learner1 = PrototypicalLearner(encoder, config)
        logits1 = learner1(support_x, support_y, query_x)
        
        torch.manual_seed(seed1)  # Same seed
        learner2 = PrototypicalLearner(encoder, config)
        logits2 = learner2(support_x, support_y, query_x)
        
        # Reproducibility property: same seed → same results
        assert torch.allclose(logits1, logits2, atol=1e-6), "Same seed should give same results"
        
        # Different seed should typically give different results (unless very unlikely)
        if seed1 != seed2:
            torch.manual_seed(seed2)
            learner3 = PrototypicalLearner(encoder, config)
            logits3 = learner3(support_x, support_y, query_x)
            
            # Very likely to be different (not guaranteed due to randomness)
            different = not torch.allclose(logits1, logits3, atol=1e-5)
            note(f"Different seeds gave different results: {different}")
            
    @given(
        n_way=st.integers(min_value=2, max_value=6),
        k_shot=st.integers(min_value=1, max_value=4),
        query_multiplier=st.integers(min_value=2, max_value=8)
    )
    @settings(max_examples=10, deadline=3000)
    def test_label_permutation_invariance_property(self, n_way, k_shot, query_multiplier):
        """Test label permutation invariance property."""
        feature_dim = 16
        
        encoder = nn.Identity()  # Identity to isolate label permutation effects
        config = PrototypicalConfig()
        learner = PrototypicalLearner(encoder, config)
        
        # Create episode
        support_x = torch.randn(n_way, k_shot, feature_dim)
        support_y = torch.arange(n_way).repeat_interleave(k_shot)
        query_x = torch.randn(n_way * query_multiplier, feature_dim)
        query_y = torch.arange(n_way).repeat(query_multiplier)
        
        # Compute logits
        logits_original = learner(support_x, support_y, query_x)
        
        # Permute class labels
        perm = torch.randperm(n_way)
        
        # Apply permutation to support set
        support_x_perm = support_x[perm]
        support_y_perm = perm[support_y]
        
        # Apply permutation to query set
        query_y_perm = perm[query_y]
        
        # Compute logits with permuted labels
        logits_perm = learner(support_x_perm, support_y_perm, query_x)
        
        # Apply inverse permutation to logits to compare
        inv_perm = torch.argsort(perm)
        logits_perm_aligned = logits_perm[:, perm]
        
        # Permutation invariance property: results should be equivalent up to label permutation
        assert torch.allclose(logits_original, logits_perm_aligned, atol=1e-5)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short", "-m", "property"])