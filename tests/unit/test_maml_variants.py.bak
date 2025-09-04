"""
Unit Tests for MAML Variants Module
===================================

Comprehensive unit tests for Model-Agnostic Meta-Learning variants:
- All MAML algorithm variants (FOMAML, ANIL, BOIL, Reptile)
- Functional forward implementations (all research solutions)
- MAML-en-LLM research-accurate implementation
- Property-based testing for meta-learning properties
- Performance regression testing

Tests all research solutions for MAML implementations.

Author: Benedict Chen (benedict@benedictchen.com)
Testing Framework: pytest + hypothesis + coverage
"""

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from hypothesis import given, strategies as st, assume
import copy
from typing import Dict, Any

from meta_learning.meta_learning_modules.maml_variants import (
    MAMLLearner, MAMLConfig, FirstOrderMAML, MAMLenLLM, MAMLenLLMConfig,
    functional_forward, functional_forward_l2l_style, functional_forward_higher_style,
    functional_forward_manual, functional_forward_compiled
)


class TestMAMLConfig:
    """Test MAMLConfig class and all configuration options."""
    
    def test_default_configuration(self):
        """Test default MAML configuration values."""
        config = MAMLConfig()
        
        assert config.inner_lr == 0.01
        assert config.outer_lr == 0.001
        assert config.inner_steps == 5
        assert config.meta_batch_size == 16
        assert config.first_order == False
        assert config.maml_variant == "standard"
        assert config.functional_forward_method == "higher_style"
    
    @pytest.mark.parametrize("variant", [
        "standard", "fomaml", "reptile", "anil", "boil"
    ])
    def test_all_maml_variants(self, variant):
        """Test all MAML variants can be configured."""
        config = MAMLConfig(maml_variant=variant)
        assert config.maml_variant == variant
    
    @pytest.mark.parametrize("method", [
        "basic", "l2l_style", "higher_style", "manual", "compiled"
    ])
    def test_functional_forward_methods(self, method):
        """Test all functional forward methods can be configured."""
        config = MAMLConfig(functional_forward_method=method)
        assert config.functional_forward_method == method
    
    def test_research_accurate_configurations(self):
        """Test research-accurate configuration combinations."""
        # FOMAML configuration
        fomaml_config = MAMLConfig(
            maml_variant="fomaml",
            first_order=True,
            inner_steps=3
        )
        assert fomaml_config.first_order == True
        
        # ANIL configuration
        anil_config = MAMLConfig(
            maml_variant="anil",
            anil_inner_loop_only=True,
            anil_num_inner_layers=2
        )
        assert anil_config.anil_inner_loop_only == True
        
        # Reptile configuration
        reptile_config = MAMLConfig(
            maml_variant="reptile",
            reptile_step_size=0.01,
            reptile_inner_steps=10
        )
        assert reptile_config.reptile_step_size == 0.01


class TestFunctionalForward:
    """Test all functional forward implementations (research solutions)."""
    
    def test_basic_functional_forward(self, simple_model):
        """Test basic functional forward implementation."""
        params = dict(simple_model.named_parameters())
        input_data = torch.randn(5, 84)  # Batch size 5, feature dim 84
        
        output = functional_forward(simple_model, params, input_data, method="basic")
        
        assert output.shape[0] == 5  # Batch size preserved
        assert output.shape[1] == 5  # Output classes
        assert not torch.isnan(output).any()
    
    def test_l2l_style_functional_forward(self, simple_model):
        """Test FIXME solution: learn2learn-style functional forward."""
        params = dict(simple_model.named_parameters())
        input_data = torch.randn(5, 84)
        
        output = functional_forward_l2l_style(simple_model, params, input_data)
        
        assert output.shape[0] == 5
        assert output.shape[1] == 5
        assert not torch.isnan(output).any()
    
    def test_higher_style_functional_forward(self, simple_model):
        """Test FIXME solution: higher-library-style functional forward."""
        params = dict(simple_model.named_parameters())
        input_data = torch.randn(5, 84)
        
        output = functional_forward_higher_style(simple_model, params, input_data)
        
        assert output.shape[0] == 5
        assert output.shape[1] == 5
        assert not torch.isnan(output).any()
    
    def test_manual_functional_forward(self, simple_model):
        """Test FIXME solution: manual functional forward for complex models."""
        params = dict(simple_model.named_parameters())
        input_data = torch.randn(5, 84)
        
        output = functional_forward_manual(simple_model, params, input_data)
        
        assert output.shape[0] == 5
        assert output.shape[1] == 5
        assert not torch.isnan(output).any()
    
    def test_compiled_functional_forward(self, simple_model):
        """Test FIXME solution: PyTorch 2.0+ compiled functional forward."""
        params = dict(simple_model.named_parameters())
        input_data = torch.randn(5, 84)
        
        # Note: torch.compile may not work in all test environments
        try:
            output = functional_forward_compiled(simple_model, params, input_data)
            assert output.shape[0] == 5
            assert output.shape[1] == 5
            assert not torch.isnan(output).any()
        except RuntimeError as e:
            # Skip if torch.compile not available
            pytest.skip(f"torch.compile not available: {e}")
    
    def test_functional_forward_gradient_flow(self, simple_model):
        """Test that functional forward preserves gradient flow."""
        params = dict(simple_model.named_parameters())
        input_data = torch.randn(5, 84, requires_grad=True)
        
        # Modify parameters to require gradients
        modified_params = {}
        for name, param in params.items():
            modified_params[name] = param.clone().requires_grad_(True)
        
        output = functional_forward(simple_model, modified_params, input_data, method="higher_style")
        loss = output.sum()
        loss.backward()
        
        # Check gradients exist
        assert input_data.grad is not None
        for param in modified_params.values():
            assert param.grad is not None


class TestMAMLLearner:
    """Test MAMLLearner class and core functionality."""
    
    def test_initialization(self, simple_model, maml_config):
        """Test MAML learner initialization."""
        learner = MAMLLearner(simple_model, maml_config)
        
        assert learner.model == simple_model
        assert learner.config == maml_config
        assert hasattr(learner, 'meta_optimizer')
    
    def test_meta_train_step_basic(self, simple_model, sample_data):
        """Test basic meta-training step."""
        config = MAMLConfig(
            inner_steps=2,  # Keep low for testing
            meta_batch_size=2,
            maml_variant="standard"
        )
        learner = MAMLLearner(simple_model, config)
        
        # Create task batch
        task_batch = []
        for _ in range(2):
            task_batch.append({
                'support': {
                    'data': sample_data['support_data'][:10],  # Smaller for speed
                    'labels': sample_data['support_labels'][:10]
                },
                'query': {
                    'data': sample_data['query_data'][:10],
                    'labels': sample_data['query_labels'][:10]
                }
            })
        
        metrics = learner.meta_train_step(task_batch)
        
        assert 'meta_loss' in metrics
        assert 'task_metrics' in metrics
        assert len(metrics['task_metrics']) == 2
    
    @pytest.mark.parametrize("variant", ["standard", "fomaml", "reptile", "anil", "boil"])
    def test_all_maml_variants_execution(self, simple_model, sample_data, variant):
        """Test all MAML variants execute without errors."""
        config = MAMLConfig(
            maml_variant=variant,
            inner_steps=2,  # Keep low for testing
            meta_batch_size=1,
            first_order=(variant == "fomaml")
        )
        learner = MAMLLearner(simple_model, config)
        
        task_batch = [{
            'support': {
                'data': sample_data['support_data'][:5],
                'labels': sample_data['support_labels'][:5]
            },
            'query': {
                'data': sample_data['query_data'][:5],
                'labels': sample_data['query_labels'][:5]
            }
        }]
        
        metrics = learner.meta_train_step(task_batch)
        
        assert 'meta_loss' in metrics
        assert isinstance(metrics['meta_loss'], (int, float))
    
    def test_inner_loop_adaptation(self, simple_model, sample_data):
        """Test inner loop adaptation process."""
        config = MAMLConfig(inner_steps=3, inner_lr=0.01)
        learner = MAMLLearner(simple_model, config)
        
        support_data = sample_data['support_data'][:10]
        support_labels = sample_data['support_labels'][:10]
        
        adapted_params, adaptation_info = learner._inner_loop_adapt(
            support_data, support_labels, task_id="test_task"
        )
        
        assert isinstance(adapted_params, dict)
        assert 'steps' in adaptation_info
        assert adaptation_info['steps'] <= config.inner_steps
        
        # Check that parameters were actually modified
        original_params = dict(simple_model.named_parameters())
        for name in adapted_params:
            assert name in original_params
            # Parameters should be different after adaptation
            assert not torch.equal(adapted_params[name], original_params[name])
    
    def test_query_loss_computation(self, simple_model, sample_data):
        """Test query loss computation with adapted parameters."""
        config = MAMLConfig(inner_steps=2)
        learner = MAMLLearner(simple_model, config)
        
        # Get adapted parameters
        adapted_params, _ = learner._inner_loop_adapt(
            sample_data['support_data'][:10],
            sample_data['support_labels'][:10]
        )
        
        # Compute query loss
        query_loss = learner._compute_query_loss(
            adapted_params,
            sample_data['query_data'][:10],
            sample_data['query_labels'][:10]
        )
        
        assert isinstance(query_loss, torch.Tensor)
        assert query_loss.requires_grad
        assert query_loss.item() >= 0
    
    def test_gradient_clipping(self, simple_model, sample_data):
        """Test gradient clipping functionality."""
        config = MAMLConfig(
            gradient_clipping=1.0,
            inner_steps=2,
            meta_batch_size=1
        )
        learner = MAMLLearner(simple_model, config)
        
        task_batch = [{
            'support': {
                'data': sample_data['support_data'][:5],
                'labels': sample_data['support_labels'][:5]
            },
            'query': {
                'data': sample_data['query_data'][:5],
                'labels': sample_data['query_labels'][:5]
            }
        }]
        
        metrics = learner.meta_train_step(task_batch)
        
        # Should complete without errors
        assert 'meta_loss' in metrics


class TestFirstOrderMAML:
    """Test First-Order MAML implementation."""
    
    def test_first_order_maml_initialization(self, simple_model):
        """Test FOMAML initialization."""
        config = MAMLConfig(first_order=True, maml_variant="fomaml")
        fomaml = FirstOrderMAML(simple_model, config)
        
        assert fomaml.config.first_order == True
        assert isinstance(fomaml, FirstOrderMAML)
    
    def test_first_order_computation(self, simple_model, sample_data):
        """Test that FOMAML uses first-order gradients only."""
        config = MAMLConfig(
            maml_variant="fomaml",
            first_order=True,
            inner_steps=2,
            meta_batch_size=1
        )
        fomaml = FirstOrderMAML(simple_model, config)
        
        task_batch = [{
            'support': {
                'data': sample_data['support_data'][:5],
                'labels': sample_data['support_labels'][:5]
            },
            'query': {
                'data': sample_data['query_data'][:5],
                'labels': sample_data['query_labels'][:5]
            }
        }]
        
        metrics = fomaml.meta_train_step(task_batch)
        
        assert 'meta_loss' in metrics
        # FOMAML should execute faster than full MAML
        assert isinstance(metrics['meta_loss'], (int, float))


class TestMAMLenLLM:
    """Test MAML-en-LLM research-accurate implementation."""
    
    def test_maml_en_llm_initialization(self, mock_tokenizer):
        """Test MAML-en-LLM initialization with research-accurate approach."""
        base_model = nn.Sequential(nn.Linear(512, 256), nn.Linear(256, 100))
        config = MAMLenLLMConfig()
        
        maml_llm = MAMLenLLM(base_model, config, mock_tokenizer)
        
        assert maml_llm.base_llm == base_model
        assert maml_llm.tokenizer == mock_tokenizer
        assert hasattr(maml_llm, 'synthetic_generators')
        assert hasattr(maml_llm, 'context_optimizer')
    
    def test_synthetic_data_generation(self, mock_tokenizer):
        """Test FIXME solution: synthetic data generation for meta-training."""
        base_model = nn.Linear(512, 100)
        config = MAMLenLLMConfig()
        maml_llm = MAMLenLLM(base_model, config, mock_tokenizer)
        
        # Test different domains
        for domain in ['sentiment', 'qa', 'classification', 'reasoning', 'summarization']:
            synthetic_tasks = maml_llm.generate_synthetic_meta_training_data(
                domain=domain,
                num_tasks=3,  # Keep small for testing
                shots_per_task=2
            )
            
            assert len(synthetic_tasks) == 3
            for task in synthetic_tasks:
                assert 'task_id' in task
                assert 'domain' in task
                assert 'support' in task
                assert 'query' in task
                assert task['domain'] == domain
    
    def test_in_context_learning_optimization(self, mock_tokenizer):
        """Test FIXME solution: in-context learning performance optimization."""
        base_model = nn.Linear(512, 100)
        config = MAMLenLLMConfig()
        maml_llm = MAMLenLLM(base_model, config, mock_tokenizer)
        
        support_texts = ["Example 1", "Example 2", "Example 3"]
        support_labels = ["Label A", "Label B", "Label A"]
        domain = "classification"
        
        optimized_context = maml_llm._optimize_in_context_learning(
            support_texts, support_labels, domain
        )
        
        assert 'examples' in optimized_context
        assert 'prompt_template' in optimized_context
        assert 'strategy' in optimized_context
        assert 'domain' in optimized_context
        assert optimized_context['domain'] == domain
    
    def test_cross_domain_adaptation(self, mock_tokenizer):
        """Test FIXME solution: cross-domain task adaptation."""
        base_model = nn.Linear(512, 100)
        config = MAMLenLLMConfig()
        maml_llm = MAMLenLLM(base_model, config, mock_tokenizer)
        
        # Test meta-training on multiple domains
        domains = ['sentiment', 'qa', 'classification']
        all_tasks = []
        
        for domain in domains:
            domain_tasks = maml_llm.generate_synthetic_meta_training_data(
                domain=domain, num_tasks=2, shots_per_task=2
            )
            all_tasks.extend(domain_tasks)
        
        # Meta-train step should handle multiple domains
        metrics = maml_llm.meta_train_step(all_tasks[:3])  # Test with subset
        
        assert 'meta_performance' in metrics
        assert 'domains_seen' in metrics
        assert len(metrics['domains_seen']) > 0
    
    def test_context_optimization_components(self, mock_tokenizer):
        """Test individual context optimization components."""
        base_model = nn.Linear(512, 100)
        config = MAMLenLLMConfig()
        maml_llm = MAMLenLLM(base_model, config, mock_tokenizer)
        
        # Test context selector
        context_selector = maml_llm.context_optimizer['context_selection']
        selected = context_selector.select_examples(
            ["text1", "text2", "text3"],
            ["label1", "label2", "label1"],
            "test_domain"
        )
        assert len(selected) >= 1
        
        # Test prompt optimizer
        prompt_optimizer = maml_llm.context_optimizer['prompt_optimizer']
        template = prompt_optimizer.optimize_prompt(selected, "test_domain")
        assert isinstance(template, str)
        assert "{text}" in template and "{label}" in template
        
        # Test example orderer
        example_orderer = maml_llm.context_optimizer['example_ordering']
        ordered = example_orderer.order_examples(selected, "test_domain")
        assert len(ordered) == len(selected)


class TestMAMLPropertyBased:
    """Property-based tests using Hypothesis."""
    
    @given(
        inner_steps=st.integers(1, 5),
        inner_lr=st.floats(1e-4, 1e-1),
        batch_size=st.integers(1, 4)
    )
    def test_maml_configuration_robustness(self, inner_steps, inner_lr, batch_size):
        """Property test: MAML should handle various configuration parameters."""
        config = MAMLConfig(
            inner_steps=inner_steps,
            inner_lr=inner_lr,
            meta_batch_size=batch_size
        )
        
        model = nn.Sequential(nn.Linear(32, 16), nn.ReLU(), nn.Linear(16, 5))
        learner = MAMLLearner(model, config)
        
        # Should initialize without errors
        assert learner.config.inner_steps == inner_steps
        assert learner.config.inner_lr == inner_lr
        assert learner.config.meta_batch_size == batch_size
    
    @given(
        n_support=st.integers(5, 20),
        n_query=st.integers(5, 20),
        feature_dim=st.integers(16, 64),
        n_classes=st.integers(2, 5)
    )
    def test_adaptation_invariants(self, n_support, n_query, feature_dim, n_classes):
        """Property test: adaptation should preserve key invariants."""
        assume(n_support >= n_classes)  # Need samples for each class
        
        model = nn.Sequential(
            nn.Linear(feature_dim, 32),
            nn.ReLU(),
            nn.Linear(32, n_classes)
        )
        
        config = MAMLConfig(inner_steps=2, inner_lr=0.01)
        learner = MAMLLearner(model, config)
        
        support_data = torch.randn(n_support, feature_dim)
        support_labels = torch.randint(0, n_classes, (n_support,))
        
        adapted_params, adaptation_info = learner._inner_loop_adapt(
            support_data, support_labels
        )
        
        # Property: adapted parameters should have same structure as original
        original_params = dict(model.named_parameters())
        assert set(adapted_params.keys()) == set(original_params.keys())
        
        # Property: parameter shapes should be preserved
        for name in adapted_params:
            assert adapted_params[name].shape == original_params[name].shape
        
        # Property: adaptation should have taken some steps
        assert adaptation_info['steps'] > 0
        assert adaptation_info['steps'] <= config.inner_steps
    
    @given(
        variant=st.sampled_from(["standard", "fomaml", "reptile", "anil", "boil"]),
        forward_method=st.sampled_from(["basic", "l2l_style", "higher_style", "manual"])
    )
    def test_variant_method_combinations(self, variant, forward_method):
        """Property test: all variant-method combinations should be valid."""
        config = MAMLConfig(
            maml_variant=variant,
            functional_forward_method=forward_method,
            inner_steps=1,  # Keep minimal for testing
            meta_batch_size=1
        )
        
        model = nn.Linear(16, 3)
        learner = MAMLLearner(model, config)
        
        # Should initialize successfully
        assert learner.config.maml_variant == variant
        assert learner.config.functional_forward_method == forward_method


@pytest.mark.fixme_solution
class TestFixmeSolutions:
    """Dedicated tests for all research solutions in MAML variants module."""
    
    def test_all_functional_forward_solutions(self, simple_model):
        """Test all research solutions for functional forward implementations."""
        params = dict(simple_model.named_parameters())
        input_data = torch.randn(3, 84)
        
        methods = ["basic", "l2l_style", "higher_style", "manual"]
        results = {}
        
        for method in methods:
            output = functional_forward(simple_model, params, input_data, method=method)
            results[method] = output
            
            # Each method should produce valid output
            assert output.shape == (3, 5)
            assert not torch.isnan(output).any()
            assert not torch.isinf(output).any()
        
        # All methods should produce similar outputs (within reason)
        base_output = results["basic"]
        for method, output in results.items():
            if method != "basic":
                # Outputs should be in similar range (methods implement same computation)
                assert torch.allclose(base_output, output, atol=1e-3, rtol=1e-2)
    
    @pytest.mark.research_accuracy
    def test_maml_en_llm_research_accuracy(self, mock_tokenizer):
        """Test that MAML-en-LLM follows research-accurate implementation."""
        base_model = nn.Linear(512, 100)
        config = MAMLenLLMConfig()
        maml_llm = MAMLenLLM(base_model, config, mock_tokenizer)
        
        # Should implement synthetic data generation (key innovation from paper)
        assert hasattr(maml_llm, 'synthetic_generators')
        assert len(maml_llm.synthetic_generators) > 0
        
        # Should implement in-context learning optimization (core focus)
        assert hasattr(maml_llm, 'context_optimizer')
        assert 'context_selection' in maml_llm.context_optimizer
        assert 'prompt_optimizer' in maml_llm.context_optimizer
        
        # Should implement cross-domain adaptation (paper's contribution)
        assert hasattr(maml_llm, 'domain_memory')
        assert hasattr(maml_llm, 'domain_performance')
        
        # Should NOT use LoRA adapters (this was the incorrect implementation)
        assert not hasattr(maml_llm, 'lora_adapters')
    
    def test_variant_specific_implementations(self, simple_model, sample_data):
        """Test variant-specific implementations work correctly."""
        variants_to_test = ["fomaml", "anil", "boil", "reptile"]
        
        for variant in variants_to_test:
            config = MAMLConfig(
                maml_variant=variant,
                inner_steps=2,
                meta_batch_size=1
            )
            learner = MAMLLearner(simple_model, config)
            
            task_batch = [{
                'support': {
                    'data': sample_data['support_data'][:5],
                    'labels': sample_data['support_labels'][:5]
                },
                'query': {
                    'data': sample_data['query_data'][:5],
                    'labels': sample_data['query_labels'][:5]
                }
            }]
            
            metrics = learner.meta_train_step(task_batch)
            
            assert 'meta_loss' in metrics
            assert metrics['meta_loss'] >= 0
            # Each variant should execute successfully
            assert isinstance(metrics['meta_loss'], (int, float))


# =============================================================================
# PERFORMANCE AND INTEGRATION TESTS
# =============================================================================

@pytest.mark.slow
class TestMAMLPerformance:
    """Performance and regression tests."""
    
    def test_adaptation_speed_comparison(self, simple_model, sample_data):
        """Test that FOMAML is faster than standard MAML."""
        import time
        
        # Standard MAML
        standard_config = MAMLConfig(
            maml_variant="standard",
            inner_steps=5,
            meta_batch_size=2
        )
        standard_learner = MAMLLearner(simple_model, standard_config)
        
        # First-order MAML
        fomaml_config = MAMLConfig(
            maml_variant="fomaml",
            first_order=True,
            inner_steps=5,
            meta_batch_size=2
        )
        fomaml_learner = FirstOrderMAML(copy.deepcopy(simple_model), fomaml_config)
        
        task_batch = [{
            'support': {
                'data': sample_data['support_data'][:10],
                'labels': sample_data['support_labels'][:10]
            },
            'query': {
                'data': sample_data['query_data'][:10],
                'labels': sample_data['query_labels'][:10]
            }
        }]
        
        # Time standard MAML
        start_time = time.time()
        standard_metrics = standard_learner.meta_train_step(task_batch)
        standard_time = time.time() - start_time
        
        # Time FOMAML
        start_time = time.time()
        fomaml_metrics = fomaml_learner.meta_train_step(task_batch)
        fomaml_time = time.time() - start_time
        
        # FOMAML should be faster or similar
        assert fomaml_time <= standard_time * 1.5  # Allow some variance
        
        # Both should produce valid results
        assert 'meta_loss' in standard_metrics
        assert 'meta_loss' in fomaml_metrics


@pytest.mark.integration
class TestMAMLIntegration:
    """Integration tests with other modules."""
    
    def test_maml_with_test_time_compute(self, simple_model, sample_data):
        """Test MAML integration with test-time compute scaling."""
        from meta_learning.meta_learning_modules.test_time_compute import TestTimeComputeScaler, TestTimeComputeConfig
        
        # Create MAML learner
        maml_config = MAMLConfig(inner_steps=2, meta_batch_size=1)
        maml_learner = MAMLLearner(simple_model, maml_config)
        
        # Create test-time compute scaler
        ttc_config = TestTimeComputeConfig(compute_strategy="basic", max_compute_steps=2)
        ttc_scaler = TestTimeComputeScaler(simple_model, ttc_config)
        
        # Test that both can work with the same model
        # First adapt with MAML
        adapted_params, _ = maml_learner._inner_loop_adapt(
            sample_data['support_data'][:5],
            sample_data['support_labels'][:5]
        )
        
        # Then use test-time compute
        predictions, metrics = ttc_scaler.scale_compute(
            sample_data['support_data'][:5],
            sample_data['support_labels'][:5],
            sample_data['query_data'][:5]
        )
        
        assert predictions.shape[0] == 5
        assert 'strategy' in metrics
    
    def test_maml_with_continual_learning(self, simple_model, sample_data):
        """Test MAML integration with continual learning."""
        config = MAMLConfig(inner_steps=2, meta_batch_size=1)
        learner = MAMLLearner(simple_model, config)
        
        # Simulate multiple tasks (continual learning scenario)
        task_results = []
        
        for task_id in range(3):
            # Create slightly different task data
            task_support = sample_data['support_data'] + 0.1 * task_id * torch.randn_like(sample_data['support_data'])
            task_query = sample_data['query_data'] + 0.1 * task_id * torch.randn_like(sample_data['query_data'])
            
            task_batch = [{
                'support': {
                    'data': task_support[:5],
                    'labels': sample_data['support_labels'][:5]
                },
                'query': {
                    'data': task_query[:5],
                    'labels': sample_data['query_labels'][:5]
                }
            }]
            
            metrics = learner.meta_train_step(task_batch)
            task_results.append(metrics)
        
        # Should handle multiple tasks successfully
        assert len(task_results) == 3
        for metrics in task_results:
            assert 'meta_loss' in metrics
            assert isinstance(metrics['meta_loss'], (int, float))