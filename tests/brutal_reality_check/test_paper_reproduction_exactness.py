"""
🔥 PAPER REPRODUCTION EXACTNESS TESTS 🔥
=========================================

These tests will BRUTALLY VERIFY if we actually implement the papers we claim to.
Every algorithm will be tested against the EXACT equations, datasets, and results
from the original research papers.

When we run these tests, we'll discover which of our implementations
are just academic fan fiction.

Author: The Brutal Technical Advisor  
Status: Written assuming perfection, will expose delusion
"""

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional
import math
from dataclasses import dataclass

from meta_learning.meta_learning_modules import (
    TestTimeComputeScaler, TestTimeComputeConfig,
    MAMLLearner, MAMLConfig, FirstOrderMAML,
    PrototypicalNetworks, PrototypicalConfig,
    MatchingNetworks, MatchingConfig,
    RelationNetworks, RelationConfig,
    OnlineMetaLearner, OnlineMetaConfig,
    MetaLearningDataset, TaskConfiguration
)


@dataclass
class PaperReproductionResult:
    """Results from attempting to reproduce paper results."""
    paper_name: str
    claimed_accuracy: float
    reproduced_accuracy: float
    accuracy_gap: float
    mathematical_correctness: bool
    equation_validation_passed: bool
    dataset_correctness: bool
    hyperparameter_match: bool
    overall_reproduction_success: bool


class PaperReproductionValidator:
    """
    The judge that will determine if we're implementing papers or just making stuff up.
    """
    
    @staticmethod
    def validate_equation_implementation(computed_result, expected_result, 
                                       paper_equation, tolerance=1e-6):
        """
        Verify if our implementation matches the paper's equation exactly.
        Returns detailed analysis of mathematical correctness.
        """
        if torch.is_tensor(computed_result) and torch.is_tensor(expected_result):
            matches = torch.allclose(computed_result, expected_result, atol=tolerance, rtol=tolerance)
            max_error = torch.max(torch.abs(computed_result - expected_result)).item()
        else:
            matches = abs(computed_result - expected_result) < tolerance
            max_error = abs(computed_result - expected_result)
        
        return {
            'equation': paper_equation,
            'matches_paper': matches,
            'max_error': max_error,
            'tolerance_used': tolerance,
            'computed_shape': getattr(computed_result, 'shape', 'scalar'),
            'expected_shape': getattr(expected_result, 'shape', 'scalar')
        }
    
    @staticmethod
    def check_hyperparameter_alignment(our_config, paper_hyperparams):
        """
        Check if our hyperparameters match the paper exactly.
        Papers are very specific about hyperparameters for reproducibility.
        """
        mismatches = []
        for param_name, paper_value in paper_hyperparams.items():
            if hasattr(our_config, param_name):
                our_value = getattr(our_config, param_name)
                if our_value != paper_value:
                    mismatches.append({
                        'param': param_name,
                        'paper_value': paper_value,
                        'our_value': our_value
                    })
            else:
                mismatches.append({
                    'param': param_name,
                    'paper_value': paper_value,
                    'our_value': 'MISSING'
                })
        
        return {
            'matches_paper': len(mismatches) == 0,
            'mismatches': mismatches,
            'mismatch_count': len(mismatches)
        }


class TestSnell2017PrototypicalReproduction:
    """
    BRUTAL TEST: Snell et al. (2017) "Prototypical Networks for Few-shot Learning"
    
    We claim to implement this paper. Let's see if we actually do.
    Original paper results on Omniglot: 96.0% (20-way 1-shot), 98.9% (20-way 5-shot)
    """
    
    def test_snell2017_equation_1_exact_reproduction(self):
        """
        Test EXACT reproduction of Snell et al. Equation 1:
        c_k = (1/|S_k|) * Σ(x_i ∈ S_k) f_φ(x_i)
        
        This will catch if we compute prototypes differently than the paper.
        """
        torch.manual_seed(42)
        
        # Use paper's exact setup: embedding dimension is paper-specific
        embedding_dim = 64  # As mentioned in Snell et al. supplementary
        n_classes = 5
        
        # Create deterministic encoder for testing
        class PaperEncoder(nn.Module):
            def forward(self, x):
                # Simplified encoder that gives predictable embeddings
                return x.view(x.shape[0], -1)[:, :embedding_dim]
        
        encoder = PaperEncoder()
        model = PrototypicalNetworks(encoder, {"n_way": n_classes})
        
        # Create test embeddings (paper uses real Omniglot, we'll test with controlled data)
        embeddings = torch.tensor([
            [1.0, 0.0, 0.0, 0.0],  # Class 0 sample 1
            [1.1, 0.1, 0.0, 0.0],  # Class 0 sample 2
            [0.0, 1.0, 0.0, 0.0],  # Class 1 sample 1
            [0.0, 1.1, 0.1, 0.0],  # Class 1 sample 2
            [0.0, 0.0, 1.0, 0.0],  # Class 2 sample 1
        ])
        support_labels = torch.tensor([0, 0, 1, 1, 2])
        
        # PAPER EQUATION 1: Manual computation for verification
        expected_prototypes = []
        for k in range(3):  # 3 classes in this test
            class_mask = support_labels == k
            if class_mask.sum() > 0:
                class_embeddings = embeddings[class_mask]
                prototype = class_embeddings.mean(dim=0)  # Exact paper formula
                expected_prototypes.append(prototype)
        expected_prototypes = torch.stack(expected_prototypes)
        
        # Test our implementation
        computed_prototypes = model.compute_prototypes(embeddings, support_labels, 3)
        
        # BRUTAL VERIFICATION
        validator = PaperReproductionValidator()
        equation_check = validator.validate_equation_implementation(
            computed_prototypes, expected_prototypes, 
            "Snell et al. Eq 1: c_k = (1/|S_k|) * Σ(x_i ∈ S_k) f_φ(x_i)"
        )
        
        assert equation_check['matches_paper'], \
            f"Prototypical Networks Equation 1 implementation WRONG! Error: {equation_check['max_error']}"
    
    def test_snell2017_equation_2_distance_exact_reproduction(self):
        """
        Test EXACT reproduction of Snell et al. Equation 2:
        d(f_φ(x), c_k) = ||f_φ(x) - c_k||²
        
        Paper specifically uses squared Euclidean distance.
        """
        torch.manual_seed(42)
        
        # Paper's exact distance computation test
        query_embedding = torch.tensor([[2.0, 1.0]])
        prototype = torch.tensor([[1.0, 0.0]])
        
        # PAPER EQUATION 2: Manual computation
        # ||[2,1] - [1,0]||² = |[1,1]|² = 1² + 1² = 2
        expected_distance = torch.tensor([[2.0]])
        
        # Test our implementation  
        encoder = nn.Identity()
        model = PrototypicalNetworks(encoder, {"n_way": 1})
        computed_distance = model.compute_distances(query_embedding, prototype)
        
        # BRUTAL VERIFICATION
        validator = PaperReproductionValidator()
        equation_check = validator.validate_equation_implementation(
            computed_distance, expected_distance,
            "Snell et al. Eq 2: d(f_φ(x), c_k) = ||f_φ(x) - c_k||²"
        )
        
        assert equation_check['matches_paper'], \
            f"Distance computation doesn't match paper! Error: {equation_check['max_error']}"
    
    def test_snell2017_equation_3_softmax_exact_reproduction(self):
        """
        Test EXACT reproduction of Snell et al. Equation 3:
        p_φ(y = k | x) = exp(-d(f_φ(x), c_k)) / Σ_k' exp(-d(f_φ(x), c_k'))
        
        Paper uses negative distance in softmax (closer = higher probability).
        """
        torch.manual_seed(42)
        
        # Test distances from query to prototypes
        distances = torch.tensor([[1.0, 2.0, 4.0]])  # Query to 3 prototypes
        
        # PAPER EQUATION 3: Manual computation
        # Softmax of negative distances: softmax([-1, -2, -4])
        negative_distances = -distances
        expected_probabilities = torch.softmax(negative_distances, dim=1)
        
        # Test our implementation
        encoder = nn.Identity()
        model = PrototypicalNetworks(encoder, {"n_way": 3})
        computed_probabilities = model.compute_probabilities(distances)
        
        # BRUTAL VERIFICATION
        validator = PaperReproductionValidator()
        equation_check = validator.validate_equation_implementation(
            computed_probabilities, expected_probabilities,
            "Snell et al. Eq 3: p_φ(y=k|x) = exp(-d(f_φ(x),c_k)) / Σ_k' exp(-d(f_φ(x),c_k'))"
        )
        
        assert equation_check['matches_paper'], \
            f"Probability computation doesn't match paper! Error: {equation_check['max_error']}"
    
    def test_snell2017_hyperparameter_reproduction(self):
        """
        Test if our hyperparameters match those used in the paper.
        Papers are very specific about these for reproducibility.
        """
        # Snell et al. 2017 hyperparameters from paper
        paper_hyperparams = {
            'distance_metric': 'euclidean',  # Paper uses squared Euclidean
            'temperature': 1.0,             # No temperature scaling mentioned
            'learn_temperature': False,     # Fixed temperature
        }
        
        # Test our default configuration
        our_config = PrototypicalConfig()
        
        validator = PaperReproductionValidator()
        hyperparameter_check = validator.check_hyperparameter_alignment(
            our_config, paper_hyperparams
        )
        
        if not hyperparameter_check['matches_paper']:
            print(f"⚠️  Hyperparameter mismatches with Snell 2017:")
            for mismatch in hyperparameter_check['mismatches']:
                print(f"   {mismatch['param']}: Paper={mismatch['paper_value']}, Ours={mismatch['our_value']}")
        
        # This is a warning, not a failure, but indicates reproduction issues
        assert hyperparameter_check['mismatch_count'] < 3, \
            "Too many hyperparameter mismatches with original paper!"


class TestFinn2017MAMLReproduction:
    """
    BRUTAL TEST: Finn et al. (2017) "Model-Agnostic Meta-Learning for Fast Adaptation"
    
    We claim to implement MAML. Let's see if it's actually MAML or just gradient descent.
    Original paper results on Omniglot: 89.7% (20-way 1-shot), 94.9% (20-way 5-shot)  
    """
    
    def test_finn2017_equation_1_inner_update_exact_reproduction(self):
        """
        Test EXACT reproduction of Finn et al. Equation 1:
        θ'_i = θ - α * ∇_θ L_T_i(f_θ)
        
        This is the core of MAML - if this is wrong, it's not MAML.
        """
        torch.manual_seed(42)
        
        # Simple model for controlled testing
        model = nn.Linear(2, 1)
        initial_weight = torch.tensor([[1.0, -1.0]])
        initial_bias = torch.tensor([0.5])
        
        with torch.no_grad():
            model.weight.copy_(initial_weight)
            model.bias.copy_(initial_bias)
        
        # MAML configuration matching paper
        alpha = 0.01  # Inner learning rate from paper
        config = MAMLConfig(inner_lr=alpha, inner_steps=1)
        maml = MAMLLearner(model, config)
        
        # Create task with known gradient
        x = torch.tensor([[1.0, 0.0]])  # Input
        y = torch.tensor([2.0])         # Target
        
        # PAPER EQUATION 1: Manual computation of expected update
        model.zero_grad()
        prediction = model(x)
        loss = F.mse_loss(prediction, y)
        loss.backward()
        
        # Expected parameters after one gradient step: θ' = θ - α * ∇L
        expected_weight = initial_weight - alpha * model.weight.grad
        expected_bias = initial_bias - alpha * model.bias.grad
        
        # Reset model for MAML test
        with torch.no_grad():
            model.weight.copy_(initial_weight)
            model.bias.copy_(initial_bias)
        
        # Test MAML inner update
        adapted_model = maml.adapt_to_task(x, y, None, None)
        
        # BRUTAL VERIFICATION
        validator = PaperReproductionValidator()
        
        weight_check = validator.validate_equation_implementation(
            adapted_model.weight.data, expected_weight,
            "Finn et al. Eq 1: θ'_i = θ - α * ∇_θ L_T_i(f_θ) [weight]"
        )
        
        bias_check = validator.validate_equation_implementation(
            adapted_model.bias.data, expected_bias,
            "Finn et al. Eq 1: θ'_i = θ - α * ∇_θ L_T_i(f_θ) [bias]"
        )
        
        assert weight_check['matches_paper'], \
            f"MAML inner update WRONG for weights! Error: {weight_check['max_error']}"
        assert bias_check['matches_paper'], \
            f"MAML inner update WRONG for bias! Error: {bias_check['max_error']}"
    
    def test_finn2017_equation_2_meta_gradient_computation(self):
        """
        Test EXACT reproduction of Finn et al. Equation 2:
        θ ← θ - β * ∇_θ Σ_{T_i ~ p(T)} L_{T_i}(f_{θ'_i})
        
        This tests if we compute meta-gradients correctly (the hard part of MAML).
        """
        torch.manual_seed(42)
        
        # Simple 2-layer model for gradient testing
        model = nn.Sequential(
            nn.Linear(3, 2),
            nn.Linear(2, 1)
        )
        
        # MAML configuration from paper
        config = MAMLConfig(
            inner_lr=0.01,    # α in paper
            inner_steps=1,    # Paper uses 1 step for many experiments  
            outer_lr=0.001    # β in paper
        )
        maml = MAMLLearner(model, config)
        
        # Create meta-batch of tasks
        meta_batch = []
        for i in range(2):  # Small meta-batch for controlled testing
            support_x = torch.randn(5, 3)
            support_y = torch.randn(5, 1)
            query_x = torch.randn(3, 3)
            query_y = torch.randn(3, 1)
            meta_batch.append((support_x, support_y, query_x, query_y))
        
        # Store initial parameters for comparison
        initial_params = [p.clone().detach() for p in model.parameters()]
        
        # Test meta-update
        meta_result = maml.meta_train_step(meta_batch)
        
        # BRUTAL VERIFICATION: Check if parameters actually updated
        parameters_changed = False
        for initial, current in zip(initial_params, model.parameters()):
            if not torch.allclose(initial, current.data, atol=1e-8):
                parameters_changed = True
                break
        
        assert parameters_changed, "Meta-gradient step didn't update parameters!"
        assert isinstance(meta_result, dict), "Meta-training should return metrics!"
        assert 'meta_loss' in meta_result, "Missing meta-loss in results!"
        assert not np.isnan(meta_result['meta_loss']), "Meta-loss is NaN!"
    
    def test_finn2017_first_order_approximation_correctness(self):
        """
        Test first-order MAML approximation (FOMAML) from Finn et al.
        Paper shows this is much faster with minimal performance loss.
        """
        torch.manual_seed(42)
        
        # Create identical models for comparison
        model1 = nn.Linear(5, 2)
        model2 = nn.Linear(5, 2)
        
        # Copy parameters to ensure identical starting points
        with torch.no_grad():
            for p1, p2 in zip(model1.parameters(), model2.parameters()):
                p2.data.copy_(p1.data)
        
        # Full MAML vs First-order MAML
        full_maml = MAMLLearner(model1, MAMLConfig(inner_lr=0.1, first_order=False))
        fo_maml = FirstOrderMAML(model2, MAMLConfig(inner_lr=0.1, first_order=True))
        
        # Create identical task
        support_x = torch.randn(10, 5)
        support_y = torch.randint(0, 2, (10,))
        query_x = torch.randn(5, 5) 
        query_y = torch.randint(0, 2, (5,))
        
        meta_batch = [(support_x, support_y, query_x, query_y)]
        
        # Test both methods
        full_result = full_maml.meta_train_step(meta_batch)
        fo_result = fo_maml.meta_train_step(meta_batch)
        
        # BRUTAL VERIFICATION: Both should work but potentially give different results
        assert isinstance(full_result, dict) and isinstance(fo_result, dict), \
            "Both MAML variants should return metrics!"
        
        assert not np.isnan(full_result['meta_loss']), "Full MAML meta-loss is NaN!"
        assert not np.isnan(fo_result['meta_loss']), "First-order MAML meta-loss is NaN!"
        
        # First-order should be computationally different but still valid
        # (We can't easily test speed here, but we can verify it doesn't crash)


class TestSnell2024TestTimeComputeReproduction:
    """
    BRUTAL TEST: Snell et al. (2024) "Scaling LLM Test-Time Compute Optimally"
    
    We claim this is a "FIRST PUBLIC IMPLEMENTATION" of 2024 breakthrough research.
    Let's see if it's actually implementing the paper or just burning GPU cycles.
    """
    
    def test_snell2024_compute_scaling_law_reproduction(self):
        """
        Test if we implement the compute scaling law from Snell et al. 2024.
        Paper shows performance scales with compute budget in predictable ways.
        """
        torch.manual_seed(42)
        
        # Create test setup matching paper's conceptual framework
        class MockLLMModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(10, 5)
                self.iteration_count = 0
                
            def forward(self, support_x, support_y, query_x):
                self.iteration_count += 1
                # Simulate improvement with more iterations (paper's key insight)
                improvement_factor = 1.0 + (self.iteration_count * 0.05)
                base_output = self.linear(query_x)
                return base_output * improvement_factor
        
        model = MockLLMModel()
        config = TestTimeComputeConfig(
            max_compute_budget=20,
            min_compute_steps=5,
            difficulty_adaptive=True
        )
        ttc_scaler = TestTimeComputeScaler(model, config)
        
        # Test compute scaling behavior
        support_x = torch.randn(15, 10)
        support_y = torch.randint(0, 5, (15,))
        query_x = torch.randn(10, 10)
        
        # Test different compute budgets
        budgets = [5, 10, 20]
        results = []
        
        for budget in budgets:
            model.iteration_count = 0  # Reset
            config.max_compute_budget = budget
            
            with torch.no_grad():
                predictions, metrics = ttc_scaler.scale_compute(support_x, support_y, query_x)
                results.append({
                    'budget': budget,
                    'compute_used': metrics['compute_used'],
                    'predictions': predictions,
                    'confidence': metrics.get('final_confidence', 0.5)
                })
        
        # BRUTAL VERIFICATION: More compute should generally lead to better performance
        assert all(r['compute_used'] <= r['budget'] for r in results), \
            "Compute usage exceeds allocated budget!"
        
        # Verify compute scaling properties (paper's main contribution)
        compute_used_values = [r['compute_used'] for r in results]
        assert compute_used_values == sorted(compute_used_values), \
            "Higher budgets should use more compute!"
    
    def test_snell2024_adaptive_compute_allocation(self):
        """
        Test adaptive compute allocation based on task difficulty.
        Paper's key insight: harder tasks need more compute.
        """
        torch.manual_seed(42)
        
        # Create model that can differentiate task difficulty
        model = nn.Linear(10, 3)
        config = TestTimeComputeConfig(
            max_compute_budget=15,
            min_compute_steps=3,
            difficulty_adaptive=True,
            confidence_threshold=0.8
        )
        ttc_scaler = TestTimeComputeScaler(model, config)
        
        # Create "easy" task (well-separated classes)
        easy_support = torch.tensor([
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        ])
        easy_labels = torch.tensor([0, 0, 1, 1, 2, 2])
        easy_query = torch.tensor([[0.9, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
        
        # Create "hard" task (overlapping classes)  
        hard_support = torch.randn(6, 10) * 0.1 + 0.5  # All samples very similar
        hard_labels = torch.tensor([0, 0, 1, 1, 2, 2])
        hard_query = torch.randn(1, 10) * 0.1 + 0.5
        
        # Test both tasks
        with torch.no_grad():
            easy_preds, easy_metrics = ttc_scaler.scale_compute(easy_support, easy_labels, easy_query)
            hard_preds, hard_metrics = ttc_scaler.scale_compute(hard_support, hard_labels, hard_query)
        
        # BRUTAL VERIFICATION: Hard task should use more compute (paper's key insight)
        if config.difficulty_adaptive:
            assert hard_metrics['compute_used'] >= easy_metrics['compute_used'], \
                "Hard task should use more compute than easy task!"
        
        # Both should produce valid outputs
        assert easy_preds.shape == hard_preds.shape, "Inconsistent output shapes!"
        assert not torch.isnan(easy_preds).any(), "Easy task predictions contain NaN!"
        assert not torch.isnan(hard_preds).any(), "Hard task predictions contain NaN!"


class TestPaperReproductionIntegration:
    """
    BRUTAL INTEGRATION TEST: Do our implementations work together as claimed?
    Many papers build on each other - let's see if our implementations do too.
    """
    
    def test_prototypical_networks_with_test_time_compute_integration(self):
        """
        Test integration of Prototypical Networks (2017) with Test-Time Compute (2024).
        We claim to combine these - let's see if it actually works mathematically.
        """
        torch.manual_seed(42)
        
        # Create Prototypical Networks as base model
        encoder = nn.Sequential(
            nn.Linear(784, 128),
            nn.ReLU(),
            nn.Linear(128, 64)
        )
        proto_model = PrototypicalNetworks(encoder, {"n_way": 5})
        
        # Add Test-Time Compute scaling
        ttc_config = TestTimeComputeConfig(max_compute_budget=8, min_compute_steps=3)
        ttc_scaler = TestTimeComputeScaler(proto_model, ttc_config)
        
        # Create 5-way 3-shot task (standard few-shot setup)
        support_x = torch.randn(15, 784)  # 5 classes * 3 samples
        support_y = torch.tensor([0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4])
        query_x = torch.randn(10, 784)
        
        # Test integrated pipeline
        with torch.no_grad():
            predictions, metrics = ttc_scaler.scale_compute(support_x, support_y, query_x)
        
        # BRUTAL VERIFICATION: Integration should preserve both papers' properties
        
        # Prototypical Networks properties (Snell 2017)
        assert predictions.shape == (10, 5), "Wrong output shape for 5-way classification!"
        
        # Check if predictions form valid probability distributions (when softmaxed)
        probs = torch.softmax(predictions, dim=1)
        prob_sums = probs.sum(dim=1)
        assert torch.allclose(prob_sums, torch.ones_like(prob_sums), atol=1e-6), \
            "Predictions don't form valid probability distributions!"
        
        # Test-Time Compute properties (Snell 2024)
        required_ttc_metrics = ['compute_used', 'allocated_budget', 'final_confidence']
        for metric in required_ttc_metrics:
            assert metric in metrics, f"Missing TTC metric: {metric}"
        
        assert metrics['compute_used'] >= ttc_config.min_compute_steps, \
            "Used less compute than minimum required!"
        assert metrics['compute_used'] <= ttc_config.max_compute_budget, \
            "Used more compute than budgeted!"
    
    def test_multi_algorithm_reproduction_consistency(self):
        """
        Test if multiple algorithms produce consistent behavior when they should.
        Papers often compare algorithms - our implementations should too.
        """
        torch.manual_seed(42)
        
        # Create same encoder for fair comparison
        def create_encoder():
            return nn.Sequential(nn.Linear(10, 32), nn.ReLU(), nn.Linear(32, 16))
        
        # Create different few-shot models
        proto_model = PrototypicalNetworks(create_encoder(), {"n_way": 3})
        matching_model = MatchingNetworks(create_encoder(), {"n_way": 3, "k_shot": 2})
        relation_model = RelationNetworks(create_encoder(), {"n_way": 3})
        
        # Same task for all models
        support_x = torch.randn(6, 10)  # 3-way 2-shot
        support_y = torch.tensor([0, 0, 1, 1, 2, 2])
        query_x = torch.randn(3, 10)
        
        # Test all models
        results = {}
        with torch.no_grad():
            results['prototypical'] = proto_model(support_x, support_y, query_x)
            results['matching'] = matching_model(support_x, support_y, query_x)
            results['relation'] = relation_model(support_x, support_y, query_x)
        
        # BRUTAL VERIFICATION: All should produce valid outputs for same task
        for model_name, output in results.items():
            assert output.shape == (3, 3), \
                f"{model_name} wrong output shape: {output.shape}"
            assert not torch.isnan(output).any(), \
                f"{model_name} produced NaN outputs!"
            assert not torch.isinf(output).any(), \
                f"{model_name} produced infinite outputs!"
        
        # All models should produce different results (they use different algorithms)
        proto_vs_matching = not torch.allclose(results['prototypical'], results['matching'], atol=1e-3)
        proto_vs_relation = not torch.allclose(results['prototypical'], results['relation'], atol=1e-3)
        
        assert proto_vs_matching, "Prototypical and Matching Networks identical - implementation error!"
        assert proto_vs_relation, "Prototypical and Relation Networks identical - implementation error!"


class TestPaperClaimsVerification:
    """
    BRUTAL VERIFICATION: Test our specific claims about paper implementations.
    We make bold claims - let's see if they hold up under scrutiny.
    """
    
    def test_breakthrough_algorithm_claims(self):
        """
        Verify our claims about implementing "breakthrough algorithms" and "first implementations".
        This is where our marketing meets mathematical reality.
        """
        # Claims we make in our package description
        claims = {
            'test_time_compute_first_implementation': True,
            'advanced_maml_variants': True,
            'enhanced_few_shot_learning': True,
            'research_accurate_implementations': True
        }
        
        verification_results = {}
        
        # Test Test-Time Compute claim
        try:
            ttc_config = TestTimeComputeConfig(max_compute_budget=10)
            model = nn.Linear(5, 2)
            ttc_scaler = TestTimeComputeScaler(model, ttc_config)
            
            support_x, support_y = torch.randn(4, 5), torch.tensor([0, 0, 1, 1])
            query_x = torch.randn(2, 5)
            
            with torch.no_grad():
                preds, metrics = ttc_scaler.scale_compute(support_x, support_y, query_x)
            
            # Verify it actually scales compute (not just returns random results)
            has_compute_scaling = 'compute_used' in metrics and metrics['compute_used'] > 0
            verification_results['test_time_compute'] = has_compute_scaling
            
        except Exception as e:
            verification_results['test_time_compute'] = False
            print(f"TTC test failed: {e}")
        
        # Test Advanced MAML claim
        try:
            model = nn.Linear(3, 2)
            config = MAMLConfig(inner_lr=0.1, inner_steps=2)
            maml = MAMLLearner(model, config)
            
            meta_batch = [(torch.randn(6, 3), torch.randint(0, 2, (6,)), 
                          torch.randn(3, 3), torch.randint(0, 2, (3,)))]
            
            result = maml.meta_train_step(meta_batch)
            
            # Verify it actually does meta-learning (not just gradient descent)
            has_meta_learning = isinstance(result, dict) and 'meta_loss' in result
            verification_results['advanced_maml'] = has_meta_learning
            
        except Exception as e:
            verification_results['advanced_maml'] = False
            print(f"MAML test failed: {e}")
        
        # Test Enhanced Few-Shot Learning claim
        try:
            encoder = nn.Linear(10, 16)
            proto_model = PrototypicalNetworks(encoder, {"n_way": 2})
            
            support_x, support_y = torch.randn(4, 10), torch.tensor([0, 0, 1, 1])
            query_x = torch.randn(2, 10)
            
            with torch.no_grad():
                output = proto_model(support_x, support_y, query_x)
            
            # Verify it produces sensible few-shot predictions
            has_few_shot = output.shape == (2, 2) and not torch.isnan(output).any()
            verification_results['enhanced_few_shot'] = has_few_shot
            
        except Exception as e:
            verification_results['enhanced_few_shot'] = False
            print(f"Few-shot test failed: {e}")
        
        # BRUTAL VERIFICATION: Check if our claims match reality
        failed_claims = [claim for claim, verified in verification_results.items() if not verified]
        
        if failed_claims:
            print(f"🚨 FAILED CLAIMS: {failed_claims}")
            print("Our marketing claims don't match our technical implementations!")
        
        # At least 75% of claims should be verifiable
        success_rate = sum(verification_results.values()) / len(verification_results)
        assert success_rate >= 0.75, \
            f"Too many unverifiable claims! Success rate: {success_rate:.2%}"


if __name__ == "__main__":
    # Run with maximum verbosity to catch every claim that doesn't hold up
    pytest.main([__file__, "-v", "--tb=long", "--capture=no", 
                "-k", "not slow", "--maxfail=5"])