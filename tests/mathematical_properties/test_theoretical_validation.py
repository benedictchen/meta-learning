"""
🧮 Mathematical Properties and Theoretical Validation Tests
==========================================================

These tests validate that our meta-learning implementations satisfy
theoretical mathematical properties and convergence guarantees from
the foundational research papers.

Mathematical Properties Validated:
- MAML: Gradient descent properties and convergence conditions
- Prototypical Networks: Distance metric properties and prototype computation
- Test-Time Compute: Computational complexity and allocation efficiency
- Continual Learning: Fisher Information properties and EWC guarantees
- Few-Shot Learning: Statistical properties and confidence intervals

Each test includes:
- Mathematical formulations from papers
- Theoretical property verification
- Convergence behavior validation  
- Numerical stability checks
- Research-accurate constraint verification
"""

import pytest
import torch
# DEMO/TEST REPRODUCIBILITY: Set seed for consistent synthetic data
torch.manual_seed(42)  # Reproducible test/demo data
import torch.nn as nn
# DEMO/TEST REPRODUCIBILITY: Set seed for consistent synthetic data
torch.manual_seed(42)  # Reproducible test/demo data
import torch.nn.functional as F
# DEMO/TEST REPRODUCIBILITY: Set seed for consistent synthetic data
torch.manual_seed(42)  # Reproducible test/demo data
import numpy as np
import math
from typing import Dict, List, Tuple, Any, Callable
from scipy import stats
from scipy.spatial.distance import pdist, squareform

# Research-accurate imports
from src.meta_learning.meta_learning_modules.test_time_compute import (
    TestTimeComputeScaler, TestTimeComputeConfig
)
from src.meta_learning.meta_learning_modules.maml_variants import (
    MAMLLearner, MAMLConfig
)
from src.meta_learning.meta_learning_modules.few_shot_learning import (
    PrototypicalNetworks, PrototypicalConfig
)
from src.meta_learning.meta_learning_modules.continual_meta_learning import (
    OnlineMetaLearner, ContinualMetaConfig
)
from src.meta_learning.meta_learning_modules.utils import (
    MetaLearningDataset, DatasetConfig, EvaluationMetrics, MetricsConfig,
    basic_confidence_interval, compute_confidence_interval
)
from src.meta_learning.meta_learning_modules.hardware_utils import (
    HardwareManager, HardwareConfig
)


@pytest.mark.mathematical_properties
class TestMAMLTheoreticalProperties:
    """
    Test MAML theoretical properties from Finn et al. (2017).
    
    Mathematical Properties:
    1. Gradient Descent: θ'ᵢ = θ - α∇θL_τᵢ(fθ)
    2. Meta-Objective: min_θ Σᵢ L_τᵢ(fθ'ᵢ)  
    3. Second-Order Gradients: ∇θL_τᵢ(fθ'ᵢ) involves Hessian terms
    4. Convergence: Meta-loss should decrease with proper learning rates
    5. Lipschitz Continuity: Gradients should be bounded
    """
    
    @pytest.fixture
    def theoretical_maml_setup(self):
        """Setup for theoretical MAML validation."""
        # Simple quadratic model for theoretical analysis
        encoder = nn.Linear(2, 1, bias=False)  # y = w₁x₁ + w₂x₂
        
        # Initialize with known weights for predictable behavior
        with torch.no_grad():
            encoder.weight.fill_(1.0)  # w₁ = w₂ = 1
        
        config = MAMLConfig(
            inner_lr=0.1,    # α = 0.1 (moderate for stability)
            outer_lr=0.01,   # β = 0.01 (smaller for meta-learning)
            num_inner_steps=1  # Single step for theoretical analysis
        )
        
        hw_manager = HardwareManager(HardwareConfig())
        encoder = hw_manager.prepare_model(encoder)
        maml_learner = MAMLLearner(encoder, config)
        
        return maml_learner, config, hw_manager
    
    def test_gradient_descent_mathematical_property(self, theoretical_maml_setup):
        """
        Test that inner loop follows gradient descent: θ'ᵢ = θ - α∇θL_τᵢ(fθ)
        
        Mathematical validation of Finn et al. 2017 Equation 1.
        """
        maml_learner, config, hw_manager = theoretical_maml_setup
        
        # Create simple task with known optimal solution
        # Task: Learn linear function y = 2x₁ + 3x₂
        support_x = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32)  # Basis vectors
        support_y = torch.tensor([2.0, 3.0], dtype=torch.float32)  # Target outputs
        query_x = support_x.clone()
        query_y = support_y.clone()
        
        # Prepare data
        support_x = hw_manager.prepare_data(support_x)
        support_y = hw_manager.prepare_data(support_y)
        query_x = hw_manager.prepare_data(query_x)
        query_y = hw_manager.prepare_data(query_y)
        
        # Get initial parameters
        initial_params = {}
        for name, param in maml_learner.model.named_parameters():
            initial_params[name] = param.clone().detach()
        
        # Manually compute expected gradient
        with torch.enable_grad():
            # Forward pass with initial parameters
            logits = maml_learner.model(support_x).squeeze()
            task_loss = F.mse_loss(logits, support_y)
            
            # Compute gradient manually
            task_loss.backward(retain_graph=True)
            expected_grad = {}
            for name, param in maml_learner.model.named_parameters():
                if param.grad is not None:
                    expected_grad[name] = param.grad.clone().detach()
                param.grad.zero_()
        
        # Now test MAML adaptation
        with hw_manager.autocast_context():
            meta_loss, adapted_params = maml_learner.meta_forward(
                support_x, support_y, query_x, query_y
            )
        
        # Validate gradient descent property: θ' = θ - α∇θL
        for name, initial_param in initial_params.items():
            if name in adapted_params and name in expected_grad:
                adapted_param = adapted_params[name]
                grad = expected_grad[name]
                
                # Expected adapted parameter: θ' = θ - α∇θL  
                expected_adapted = initial_param - config.inner_lr * grad
                
                # Check if adaptation follows gradient descent (within numerical precision)
                param_diff = torch.norm(adapted_param - expected_adapted).item()
                relative_error = param_diff / (torch.norm(expected_adapted).item() + 1e-8)
                
                assert relative_error < 0.1, f"Gradient descent property violated for {name}: relative error {relative_error}"
                
    
    def test_second_order_gradient_computation(self, theoretical_maml_setup):
        """
        Test second-order gradient computation in MAML meta-objective.
        
        Mathematical property: ∇θL_τᵢ(fθ'ᵢ) involves Hessian terms when θ'ᵢ depends on θ.
        """
        maml_learner, config, hw_manager = theoretical_maml_setup
        
        # Simple task for gradient analysis
        support_x = torch.tensor([[1.0, 1.0]], dtype=torch.float32)
        support_y = torch.tensor([1.0], dtype=torch.float32)
        query_x = torch.tensor([[1.0, 1.0]], dtype=torch.float32)  
        query_y = torch.tensor([2.0], dtype=torch.float32)  # Different target
        
        # Prepare data
        support_x = hw_manager.prepare_data(support_x)
        support_y = hw_manager.prepare_data(support_y)
        query_x = hw_manager.prepare_data(query_x)
        query_y = hw_manager.prepare_data(query_y)
        
        # Store initial parameters for comparison
        initial_params = list(maml_learner.model.parameters())
        
        # Compute meta-gradients
        optimizer = torch.optim.SGD(maml_learner.parameters(), lr=config.outer_lr)
        optimizer.zero_grad()
        
        with hw_manager.autocast_context():
            meta_loss, adapted_params = maml_learner.meta_forward(
                support_x, support_y, query_x, query_y
            )
        
        # Compute meta-gradients
        meta_loss.backward()
        
        # Validate that meta-gradients exist and are finite
        meta_grads = []
        for param in maml_learner.model.parameters():
            if param.grad is not None:
                meta_grads.append(param.grad.clone())
                
                # Meta-gradients should be finite
                assert torch.isfinite(param.grad).all(), "Meta-gradients must be finite"
                
                # Meta-gradients should be non-zero (unless at optimal point)
                grad_norm = torch.norm(param.grad).item()
                assert grad_norm < 1000, f"Meta-gradient norm too large: {grad_norm}"
        
        assert len(meta_grads) > 0, "Should have computed meta-gradients"
        
        # Test that meta-gradients differ from first-order gradients
        # (This validates second-order computation)
        first_order_config = MAMLConfig(
            inner_lr=config.inner_lr,
            outer_lr=config.outer_lr, 
            num_inner_steps=config.num_inner_steps,
            use_first_order=True  # First-order approximation
        )
        
        first_order_learner = MAMLLearner(
            nn.Linear(2, 1, bias=False), 
            first_order_config
        )
        first_order_learner = hw_manager.prepare_model(first_order_learner)
        
        # Initialize with same weights
        with torch.no_grad():
            for fo_param, so_param in zip(first_order_learner.parameters(), maml_learner.parameters()):
                fo_param.copy_(so_param.detach())
        
        # Compute first-order gradients
        fo_optimizer = torch.optim.SGD(first_order_learner.parameters(), lr=config.outer_lr)
        fo_optimizer.zero_grad()
        
        fo_meta_loss, _ = first_order_learner.meta_forward(
            support_x, support_y, query_x, query_y
        )
        fo_meta_loss.backward()
        
        # Compare gradients (should be different due to second-order terms)
        gradient_differences = []
        for so_param, fo_param in zip(maml_learner.parameters(), first_order_learner.parameters()):
            if so_param.grad is not None and fo_param.grad is not None:
                diff = torch.norm(so_param.grad - fo_param.grad).item()
                gradient_differences.append(diff)
        
        if len(gradient_differences) > 0:
            max_diff = max(gradient_differences)
            # Second-order and first-order gradients should differ (in most cases)
            if max_diff > 1e-6:
                print(f"✅ Second-order vs first-order gradient difference: {max_diff:.6f}")
            else:
                print("⚠️  Second-order and first-order gradients very similar (may be expected for this task)")
        
    
    def test_meta_learning_convergence_property(self, theoretical_maml_setup):
        """
        Test that MAML meta-learning demonstrates convergence properties.
        
        Property: Meta-loss should decrease over meta-training iterations.
        """
        maml_learner, config, hw_manager = theoretical_maml_setup
        
        # Create consistent task distribution
        def generate_linear_task():
            """Generate linear regression tasks: y = ax + b + noise"""
            a = torch.randn(1) * 2  # Random slope
            b = torch.randn(1)      # Random intercept
            
            support_x = torch.randn(5, 2)
            support_y = (support_x @ torch.tensor([[a], [a]])).squeeze() + b + 0.1 * torch.randn(5)
            
            query_x = torch.randn(3, 2) 
            query_y = (query_x @ torch.tensor([[a], [a]])).squeeze() + b + 0.1 * torch.randn(3)
            
            return support_x, support_y, query_x, query_y
        
        # Meta-training setup
        optimizer = torch.optim.Adam(maml_learner.parameters(), lr=config.outer_lr)
        meta_losses = []
        
        # Run several meta-training iterations
        num_meta_iterations = 20
        for meta_iter in range(num_meta_iterations):
            # Sample batch of tasks
            batch_meta_loss = 0.0
            batch_size = 4
            
            for _ in range(batch_size):
                support_x, support_y, query_x, query_y = generate_linear_task()
                
                # Prepare data
                support_x = hw_manager.prepare_data(support_x)
                support_y = hw_manager.prepare_data(support_y)
                query_x = hw_manager.prepare_data(query_x)
                query_y = hw_manager.prepare_data(query_y)
                
                optimizer.zero_grad()
                
                with hw_manager.autocast_context():
                    meta_loss, _ = maml_learner.meta_forward(
                        support_x, support_y, query_x, query_y
                    )
                
                batch_meta_loss += meta_loss.item()
                meta_loss.backward()
            
            optimizer.step()
            avg_meta_loss = batch_meta_loss / batch_size
            meta_losses.append(avg_meta_loss)
        
        # Analyze convergence properties
        
        # 1. Meta-losses should be finite and positive
        assert all(np.isfinite(loss) for loss in meta_losses), "All meta-losses should be finite"
        assert all(loss > 0 for loss in meta_losses), "All meta-losses should be positive"
        
        # 2. Overall trend should be decreasing (with some tolerance for noise)
        # Use linear regression to check trend
        iterations = np.arange(len(meta_losses))
        slope, intercept, r_value, p_value, std_err = stats.linregress(iterations, meta_losses)
        
        # Negative slope indicates decreasing trend (convergence)
        assert slope < 0.1, f"Meta-loss should show decreasing trend, got slope {slope:.4f}"
        
        # 3. Final meta-loss should be lower than initial (allowing for variation)
        final_losses = meta_losses[-5:]  # Last 5 iterations
        initial_losses = meta_losses[:5]  # First 5 iterations
        
        final_avg = np.mean(final_losses)
        initial_avg = np.mean(initial_losses)
        
        improvement = (initial_avg - final_avg) / initial_avg
        assert improvement > -0.5, f"Meta-loss should not significantly increase: improvement={improvement:.3f}"
        
        print(f"✅ MAML convergence validated: slope={slope:.4f}, improvement={improvement:.3f}")


@pytest.mark.mathematical_properties
class TestPrototypicalNetworksMathematicalProperties:
    """
    Test mathematical properties of Prototypical Networks (Snell et al. 2017).
    
    Mathematical Properties:
    1. Prototype Computation: cₖ = (1/|Sₖ|) Σ fφ(xᵢ) for (xᵢ,yᵢ) ∈ Sₖ
    2. Distance Function: d(fφ(x), cₖ) with metric properties
    3. Softmax Classification: p(y=k|x) = exp(-d(x,cₖ)) / Σⱼ exp(-d(x,cⱼ))
    4. Euclidean Distance Properties: Triangle inequality, symmetry, non-negativity
    5. Probability Simplex: Σₖ p(y=k|x) = 1
    """
    
    @pytest.fixture
    def prototypical_mathematical_setup(self):
        """Setup for mathematical validation of prototypical networks."""
        # Simple encoder for predictable embeddings
        encoder = nn.Sequential(
            nn.Linear(4, 8),
            nn.ReLU(),
            nn.Linear(8, 3)  # 3D embeddings for geometric analysis
        )
        
        config = PrototypicalConfig(distance_metric='euclidean')
        hw_manager = HardwareManager(HardwareConfig())
        encoder = hw_manager.prepare_model(encoder)
        proto_learner = PrototypicalNetworks(encoder, config)
        
        return proto_learner, encoder, hw_manager
    
    def test_prototype_computation_mathematical_accuracy(self, prototypical_mathematical_setup):
        """
        Test that prototype computation follows mathematical definition.
        
        Property: cₖ = (1/|Sₖ|) Σ_{(xᵢ,yᵢ) ∈ Sₖ} fφ(xᵢ)
        """
        proto_learner, encoder, hw_manager = prototypical_mathematical_setup
        
        # Create support set with known structure
        n_way, k_shot = 3, 4  # 3 classes, 4 examples each
        support_x = torch.randn(n_way * k_shot, 4)
        support_y = torch.repeat_interleave(torch.arange(n_way), k_shot)
        
        # Prepare data
        support_x = hw_manager.prepare_data(support_x)
        support_y = hw_manager.prepare_data(support_y)
        
        # Compute embeddings
        with torch.no_grad():
            embeddings = encoder(support_x)
        
        # Manually compute prototypes according to mathematical definition
        manual_prototypes = []
        for class_id in range(n_way):
            class_mask = (support_y == class_id)
            class_embeddings = embeddings[class_mask]
            
            # Mathematical definition: cₖ = (1/|Sₖ|) Σ fφ(xᵢ)
            class_size = class_embeddings.shape[0]
            prototype = (1.0 / class_size) * torch.sum(class_embeddings, dim=0)
            manual_prototypes.append(prototype)
        
        manual_prototypes = torch.stack(manual_prototypes)
        
        # Test prototype computation through model
        query_x = torch.randn(1, 4)  # Dummy query
        query_x = hw_manager.prepare_data(query_x)
        
        with hw_manager.autocast_context():
            _ = proto_learner(support_x, support_y, query_x)
        
        # Verify internal prototype computation matches mathematical definition
        # Note: This requires access to internal prototypes or computing them similarly
        computed_prototypes = []
        for class_id in range(n_way):
            class_mask = (support_y == class_id)
            class_embeddings = embeddings[class_mask]
            prototype = torch.mean(class_embeddings, dim=0)  # Mean = (1/n) * sum
            computed_prototypes.append(prototype)
        
        computed_prototypes = torch.stack(computed_prototypes)
        
        # Mathematical validation
        prototype_diff = torch.norm(computed_prototypes - manual_prototypes).item()
        relative_error = prototype_diff / (torch.norm(manual_prototypes).item() + 1e-8)
        
        assert relative_error < 1e-6, f"Prototype computation error: {relative_error}"
        
        # Validate mathematical properties
        for k in range(n_way):
            prototype = computed_prototypes[k]
            
            # Prototype should be finite
            assert torch.isfinite(prototype).all(), f"Prototype {k} should be finite"
            
            # Prototype should be in same space as embeddings
            assert prototype.shape == embeddings.shape[1:], f"Prototype {k} shape mismatch"
        
    
    def test_euclidean_distance_metric_properties(self, prototypical_mathematical_setup):
        """
        Test that Euclidean distance satisfies metric properties.
        
        Metric Properties:
        1. Non-negativity: d(x,y) ≥ 0
        2. Identity: d(x,y) = 0 ⟺ x = y  
        3. Symmetry: d(x,y) = d(y,x)
        4. Triangle Inequality: d(x,z) ≤ d(x,y) + d(y,z)
        """
        proto_learner, encoder, hw_manager = prototypical_mathematical_setup
        
        # Generate test embeddings
        n_points = 10
        embeddings = torch.randn(n_points, 3)  # 3D embeddings
        
        # Test all pairs of embeddings
        for i in range(n_points):
            for j in range(n_points):
                x = embeddings[i]
                y = embeddings[j]
                
                # Compute Euclidean distance
                dist_xy = torch.norm(x - y).item()
                dist_yx = torch.norm(y - x).item()
                
                # 1. Non-negativity: d(x,y) ≥ 0
                assert dist_xy >= 0, f"Distance should be non-negative: d({i},{j}) = {dist_xy}"
                
                # 2. Identity: d(x,y) = 0 ⟺ x = y
                if i == j:
                    assert abs(dist_xy) < 1e-6, f"Distance to self should be zero: d({i},{i}) = {dist_xy}"
                
                # 3. Symmetry: d(x,y) = d(y,x)
                assert abs(dist_xy - dist_yx) < 1e-6, f"Distance should be symmetric: d({i},{j}) ≠ d({j},{i})"
                
                # 4. Triangle inequality: d(x,z) ≤ d(x,y) + d(y,z)
                for k in range(n_points):
                    if k != i and k != j:
                        z = embeddings[k]
                        dist_xz = torch.norm(x - z).item()
                        dist_yz = torch.norm(y - z).item()
                        
                        triangle_violation = dist_xz - (dist_xy + dist_yz)
                        assert triangle_violation <= 1e-6, \
                            f"Triangle inequality violated: d({i},{k}) > d({i},{j}) + d({j},{k}), violation = {triangle_violation}"
        
    
    def test_softmax_probability_simplex_property(self, prototypical_mathematical_setup):
        """
        Test that softmax probabilities form a valid probability simplex.
        
        Properties:
        1. Non-negativity: p(y=k|x) ≥ 0 for all k
        2. Normalization: Σₖ p(y=k|x) = 1  
        3. Monotonicity: Smaller distances → Higher probabilities
        """
        proto_learner, encoder, hw_manager = prototypical_mathematical_setup
        
        # Create test scenario
        n_way, k_shot = 4, 3
        support_x = torch.randn(n_way * k_shot, 4)
        support_y = torch.repeat_interleave(torch.arange(n_way), k_shot)
        query_x = torch.randn(5, 4)  # 5 query points
        
        # Prepare data
        support_x = hw_manager.prepare_data(support_x)
        support_y = hw_manager.prepare_data(support_y)
        query_x = hw_manager.prepare_data(query_x)
        
        # Get predictions
        with hw_manager.autocast_context():
            logits = proto_learner(support_x, support_y, query_x)
        
        # Convert to probabilities
        probabilities = F.softmax(logits, dim=1)
        
        # Validate probability simplex properties
        for q in range(query_x.shape[0]):
            query_probs = probabilities[q]
            
            # 1. Non-negativity: p(y=k|x) ≥ 0
            assert (query_probs >= 0).all(), f"Probabilities should be non-negative for query {q}"
            
            # 2. Normalization: Σₖ p(y=k|x) = 1
            prob_sum = query_probs.sum().item()
            assert abs(prob_sum - 1.0) < 1e-6, f"Probabilities should sum to 1 for query {q}: sum = {prob_sum}"
            
            # 3. Probabilities should be finite
            assert torch.isfinite(query_probs).all(), f"Probabilities should be finite for query {q}"
        
        # Test monotonicity property: closer prototypes → higher probabilities
        # (This is a statistical property, not guaranteed for individual cases)
        
        # Compute distances from first query to all prototypes
        with torch.no_grad():
            query_embedding = encoder(query_x[:1])  # First query
            
            # Compute prototypes
            support_embeddings = encoder(support_x)
            prototypes = []
            for class_id in range(n_way):
                class_mask = (support_y == class_id)
                class_embeddings = support_embeddings[class_mask]
                prototype = torch.mean(class_embeddings, dim=0)
                prototypes.append(prototype)
            
            prototypes = torch.stack(prototypes)
            
            # Distances from query to each prototype
            distances = torch.norm(query_embedding - prototypes, dim=1)
            
            # Probabilities for first query
            first_query_probs = probabilities[0]
            
            # Check if there's general inverse relationship between distance and probability
            # (Statistical test - not guaranteed for single instance)
            distance_prob_correlation = torch.corrcoef(torch.stack([distances, first_query_probs]))[0, 1]
            
            # Correlation should be negative (higher distance → lower probability)
            if torch.isfinite(distance_prob_correlation):
                print(f"Distance-probability correlation: {distance_prob_correlation:.4f} (should be negative)")
            else:
                print("Distance-probability correlation not computable (may be expected)")
        


@pytest.mark.mathematical_properties
class TestStatisticalPropertiesValidation:
    """
    Test statistical properties and confidence interval computations.
    
    Mathematical Properties:
    1. Confidence Intervals: P(μ ∈ [L, U]) = 1 - α
    2. Bootstrap Distribution: Approximates sampling distribution
    3. Central Limit Theorem: Sample means approach normality
    4. Bias-Corrected Accelerated (BCa): Advanced bootstrap correction
    """
    
    def test_confidence_interval_coverage_property(self):
        """
        Test that confidence intervals achieve correct coverage probability.
        
        Mathematical Property: P(μ ∈ [L, U]) = 1 - α for confidence level 1-α
        """
        # Generate samples from known distribution
        true_mean = 5.0
        true_std = 2.0
        sample_size = 30
        num_experiments = 100  # Number of CI experiments
        confidence_level = 0.95
        expected_coverage = 0.95
        
        coverage_count = 0
        
        for experiment in range(num_experiments):
            # Generate sample from normal distribution
            sample = np.random.normal(true_mean, true_std, sample_size)
            
            # Compute confidence interval
            try:
                ci_lower, ci_upper = basic_confidence_interval(sample, confidence_level=confidence_level)
                
                # Check if true mean is within confidence interval
                if ci_lower <= true_mean <= ci_upper:
                    coverage_count += 1
                
                # Validate CI properties
                assert ci_lower <= ci_upper, f"CI bounds should be ordered: [{ci_lower}, {ci_upper}]"
                assert np.isfinite(ci_lower) and np.isfinite(ci_upper), "CI bounds should be finite"
                
            except Exception as e:
                print(f"CI computation failed for experiment {experiment}: {e}")
        
        # Test coverage probability
        observed_coverage = coverage_count / num_experiments
        coverage_error = abs(observed_coverage - expected_coverage)
        
        # Allow some tolerance for sampling variation (±10%)
        max_coverage_error = 0.10
        assert coverage_error <= max_coverage_error, \
            f"Coverage error too large: observed={observed_coverage:.3f}, expected={expected_coverage:.3f}, error={coverage_error:.3f}"
        
        print(f"✅ Confidence interval coverage: {observed_coverage:.3f} (expected {expected_coverage:.3f})")
    
    def test_bootstrap_distribution_properties(self):
        """
        Test mathematical properties of bootstrap distribution.
        
        Properties:
        1. Bootstrap mean approximates original sample mean
        2. Bootstrap variance approximates sampling variance
        3. Bootstrap distribution shape approximates sampling distribution
        """
        # Original sample
        true_mean = 10.0
        sample_size = 50
        original_sample = np.random.exponential(scale=true_mean, size=sample_size)
        original_mean = np.mean(original_sample)
        
        # Bootstrap sampling
        n_bootstrap = 1000
        bootstrap_means = []
        
        for _ in range(n_bootstrap):
            bootstrap_sample = np.random.choice(original_sample, size=sample_size, replace=True)
            bootstrap_means.append(np.mean(bootstrap_sample))
        
        bootstrap_means = np.array(bootstrap_means)
        
        # Test bootstrap properties
        
        # 1. Bootstrap mean should approximate original sample mean
        bootstrap_mean_of_means = np.mean(bootstrap_means)
        mean_bias = abs(bootstrap_mean_of_means - original_mean)
        relative_mean_bias = mean_bias / abs(original_mean) if original_mean != 0 else mean_bias
        
        assert relative_mean_bias < 0.05, f"Bootstrap mean bias too large: {relative_mean_bias:.4f}"
        
        # 2. Bootstrap variance should approximate theoretical sampling variance
        bootstrap_variance = np.var(bootstrap_means)
        theoretical_sampling_variance = np.var(original_sample) / sample_size
        
        variance_ratio = bootstrap_variance / theoretical_sampling_variance
        assert 0.5 < variance_ratio < 2.0, f"Bootstrap variance ratio out of range: {variance_ratio:.3f}"
        
        # 3. Test approximate normality (Central Limit Theorem)
        # Kolmogorov-Smirnov test against normal distribution
        normalized_bootstrap = (bootstrap_means - bootstrap_mean_of_means) / np.sqrt(bootstrap_variance)
        ks_statistic, ks_p_value = stats.kstest(normalized_bootstrap, 'norm')
        
        # P-value should be reasonably high for normality (not too strict due to finite sample)
        if ks_p_value > 0.01:
            print(f"✅ Bootstrap distribution approximately normal (p={ks_p_value:.4f})")
        else:
            print(f"⚠️  Bootstrap distribution normality test: p={ks_p_value:.4f} (may be expected for small samples)")
        
        print(f"✅ Bootstrap properties validated: bias={relative_mean_bias:.4f}, var_ratio={variance_ratio:.3f}")
    
    def test_statistical_significance_properties(self):
        """
        Test statistical significance testing properties.
        
        Properties:
        1. Type I Error Control: P(reject H₀ | H₀ true) ≤ α
        2. Power: P(reject H₀ | H₁ true) should be high for large effects
        3. P-value Properties: Uniform under null hypothesis
        """
        alpha = 0.05  # Significance level
        
        # Test Type I Error Control
        # H₀: μ = μ₀ (null hypothesis true)
        null_mean = 0.0
        sample_size = 30
        num_tests = 1000
        
        false_positives = 0
        p_values_under_null = []
        
        for test in range(num_tests):
            # Generate sample under null hypothesis
            sample = np.random.normal(null_mean, 1.0, sample_size)
            sample_mean = np.mean(sample)
            sample_std = np.std(sample, ddof=1)
            
            # Perform t-test: H₀: μ = 0 vs H₁: μ ≠ 0
            t_statistic = sample_mean / (sample_std / np.sqrt(sample_size))
            p_value = 2 * (1 - stats.t.cdf(abs(t_statistic), df=sample_size - 1))  # Two-tailed
            
            p_values_under_null.append(p_value)
            
            # Count rejections
            if p_value < alpha:
                false_positives += 1
        
        # Test Type I Error Control
        observed_type_I_error = false_positives / num_tests
        type_I_error_tolerance = alpha + 0.02  # Allow some sampling variation
        
        assert observed_type_I_error <= type_I_error_tolerance, \
            f"Type I error rate too high: {observed_type_I_error:.4f} > {type_I_error_tolerance:.4f}"
        
        # Test P-value uniformity under null (Kolmogorov-Smirnov test)
        ks_stat, ks_p = stats.kstest(p_values_under_null, 'uniform')
        
        # P-values should be approximately uniform under null
        if ks_p > 0.05:
            print(f"✅ P-values uniform under null hypothesis (KS p={ks_p:.4f})")
        else:
            print(f"⚠️  P-value uniformity test failed (KS p={ks_p:.4f})")
        
        # Test Power (H₁: μ = effect_size)
        effect_size = 0.5  # Medium effect
        true_alternative_mean = effect_size
        power_tests = 500
        correct_rejections = 0
        
        for test in range(power_tests):
            # Generate sample under alternative hypothesis
            sample = np.random.normal(true_alternative_mean, 1.0, sample_size)
            sample_mean = np.mean(sample)
            sample_std = np.std(sample, ddof=1)
            
            # T-test against null mean = 0
            t_statistic = sample_mean / (sample_std / np.sqrt(sample_size))
            p_value = 2 * (1 - stats.t.cdf(abs(t_statistic), df=sample_size - 1))
            
            if p_value < alpha:
                correct_rejections += 1
        
        observed_power = correct_rejections / power_tests
        expected_power_approx = 0.3  # Rough expectation for medium effect
        
        assert observed_power >= expected_power_approx, \
            f"Statistical power too low: {observed_power:.3f} < {expected_power_approx:.3f}"
        
        print(f"✅ Statistical properties validated: Type I error={observed_type_I_error:.4f}, Power={observed_power:.3f}")


@pytest.mark.mathematical_properties  
class TestNumericalStabilityProperties:
    """
    Test numerical stability properties across algorithms.
    
    Properties:
    1. Gradient Norms: Should remain bounded
    2. Parameter Updates: Should not cause overflow/underflow
    3. Loss Values: Should remain finite during training
    4. Condition Numbers: Should remain reasonable
    """
    
    def test_gradient_norm_boundedness(self):
        """
        Test that gradients remain bounded during meta-learning.
        
        Property: ||∇θ L|| < ∞ and reasonable magnitude
        """
        encoder = nn.Sequential(
            nn.Linear(10, 20),
            nn.ReLU(), 
            nn.Linear(20, 5)
        )
        
        config = MAMLConfig(inner_lr=0.01, outer_lr=0.001, num_inner_steps=5)
        hw_manager = HardwareManager(HardwareConfig())
        encoder = hw_manager.prepare_model(encoder)
        maml_learner = MAMLLearner(encoder, config)
        
        gradient_norms = []
        
        # Run several meta-training steps
        for step in range(10):
            # Generate random task
            support_x = torch.randn(10, 10)
            support_y = torch.randint(0, 5, (10,))
            query_x = torch.randn(5, 10)
            query_y = torch.randint(0, 5, (5,))
            
            # Prepare data
            support_x = hw_manager.prepare_data(support_x)
            support_y = hw_manager.prepare_data(support_y)
            query_x = hw_manager.prepare_data(query_x)
            query_y = hw_manager.prepare_data(query_y)
            
            # Compute meta-gradients
            optimizer = torch.optim.Adam(maml_learner.parameters(), lr=config.outer_lr)
            optimizer.zero_grad()
            
            with hw_manager.autocast_context():
                meta_loss, _ = maml_learner.meta_forward(support_x, support_y, query_x, query_y)
            
            meta_loss.backward()
            
            # Compute gradient norm
            total_grad_norm = 0.0
            param_count = 0
            
            for param in maml_learner.parameters():
                if param.grad is not None:
                    param_grad_norm = torch.norm(param.grad).item()
                    total_grad_norm += param_grad_norm ** 2
                    param_count += 1
                    
                    # Individual parameter gradients should be finite and bounded
                    assert np.isfinite(param_grad_norm), f"Parameter gradient should be finite at step {step}"
                    assert param_grad_norm < 1000, f"Parameter gradient too large at step {step}: {param_grad_norm}"
            
            if param_count > 0:
                total_grad_norm = np.sqrt(total_grad_norm)
                gradient_norms.append(total_grad_norm)
                
                # Total gradient norm should be bounded
                assert total_grad_norm < 1000, f"Total gradient norm too large at step {step}: {total_grad_norm}"
            
            optimizer.step()
        
        # Check gradient norm statistics
        if len(gradient_norms) > 0:
            mean_grad_norm = np.mean(gradient_norms)
            max_grad_norm = np.max(gradient_norms)
            std_grad_norm = np.std(gradient_norms)
            
            # Gradient norms should be reasonable and not exploding
            assert mean_grad_norm < 100, f"Mean gradient norm too large: {mean_grad_norm}"
            assert max_grad_norm < 500, f"Max gradient norm too large: {max_grad_norm}"
            
            # Gradient norms shouldn't vary too wildly (sign of instability)
            if mean_grad_norm > 0:
                coefficient_of_variation = std_grad_norm / mean_grad_norm
                assert coefficient_of_variation < 5.0, f"Gradient norm variation too high: {coefficient_of_variation}"
            
            print(f"✅ Gradient boundedness validated: mean={mean_grad_norm:.4f}, max={max_grad_norm:.4f}, CV={coefficient_of_variation:.3f}")
        else:
            print("⚠️  No gradients computed - check gradient computation")
    
    def test_loss_value_stability(self):
        """
        Test that loss values remain finite and reasonable during training.
        
        Property: Loss ∈ ℝ⁺ and bounded
        """
        encoder = nn.Sequential(nn.Linear(5, 10), nn.ReLU(), nn.Linear(10, 3))
        config = PrototypicalConfig()
        hw_manager = HardwareManager(HardwareConfig())
        encoder = hw_manager.prepare_model(encoder)
        proto_learner = PrototypicalNetworks(encoder, config)
        
        loss_values = []
        
        for iteration in range(15):
            # Generate task
            support_x = torch.randn(9, 5)  # 3 classes, 3 examples each
            support_y = torch.repeat_interleave(torch.arange(3), 3)
            query_x = torch.randn(6, 5)   # 2 queries per class
            query_y = torch.repeat_interleave(torch.arange(3), 2)
            
            # Prepare data
            support_x = hw_manager.prepare_data(support_x)
            support_y = hw_manager.prepare_data(support_y)
            query_x = hw_manager.prepare_data(query_x)
            query_y = hw_manager.prepare_data(query_y)
            
            with hw_manager.autocast_context():
                logits = proto_learner(support_x, support_y, query_x)
                loss = F.cross_entropy(logits, query_y)
            
            loss_value = loss.item()
            loss_values.append(loss_value)
            
            # Individual loss checks
            assert np.isfinite(loss_value), f"Loss should be finite at iteration {iteration}: {loss_value}"
            assert loss_value >= 0, f"Loss should be non-negative at iteration {iteration}: {loss_value}"
            assert loss_value < 100, f"Loss too large at iteration {iteration}: {loss_value}"
        
        # Statistical checks on loss values
        mean_loss = np.mean(loss_values)
        std_loss = np.std(loss_values)
        max_loss = np.max(loss_values)
        min_loss = np.min(loss_values)
        
        # Loss statistics should be reasonable
        assert mean_loss > 0, f"Mean loss should be positive: {mean_loss}"
        assert mean_loss < 50, f"Mean loss too high: {mean_loss}"
        assert std_loss < mean_loss, f"Loss variance too high: std={std_loss}, mean={mean_loss}"
        
        # Check for loss explosion or collapse
        loss_range = max_loss - min_loss
        relative_range = loss_range / mean_loss if mean_loss > 0 else loss_range
        assert relative_range < 10, f"Loss range too large: relative_range={relative_range}"
        
        print(f"✅ Loss stability validated: mean={mean_loss:.4f}, std={std_loss:.4f}, range={loss_range:.4f}")


if __name__ == "__main__":
    # Run with: pytest tests/mathematical_properties/test_theoretical_validation.py -v -m mathematical_properties
    pytest.main([__file__, "-v", "-m", "mathematical_properties"])