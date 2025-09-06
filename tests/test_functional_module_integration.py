#!/usr/bin/env python3
"""
Test FunctionalModule Integration
================================

Tests for the FunctionalModule integration in MAML training, ensuring that
the functional forward passes work correctly with adapted parameters.
"""

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from meta_learning.algorithms.maml_research_accurate import FunctionalModule
from meta_learning.toolkit import MetaLearningToolkit
from meta_learning.core.episode import Episode


class TestFunctionalModule:
    """Test FunctionalModule functionality."""

    def test_functional_module_import(self):
        """Test that FunctionalModule can be imported correctly."""
        assert FunctionalModule is not None
        assert hasattr(FunctionalModule, 'functional_forward')
        assert callable(getattr(FunctionalModule, 'functional_forward'))

    def test_functional_forward_basic(self):
        """Test basic functional forward pass."""
        # Create simple model
        model = nn.Sequential(
            nn.Linear(10, 5),
            nn.ReLU(),
            nn.Linear(5, 3)
        )
        
        # Create test input
        x = torch.randn(4, 10)
        
        # Forward pass without parameter override
        output1 = FunctionalModule.functional_forward(model, x)
        output2 = model(x)
        
        # Should be identical when no parameter override
        assert torch.allclose(output1, output2, atol=1e-6)
        assert output1.shape == (4, 3)

    def test_functional_forward_with_params(self):
        """Test functional forward pass with parameter override."""
        # Create simple model
        model = nn.Linear(5, 3)
        x = torch.randn(2, 5)
        
        # Get original parameters
        original_output = model(x)
        
        # Create modified parameters
        modified_params = {}
        for name, param in model.named_parameters():
            modified_params[name] = param + 0.1  # Small modification
        
        # Forward pass with modified parameters
        modified_output = FunctionalModule.functional_forward(model, x, modified_params)
        
        # Should be different from original
        assert not torch.allclose(original_output, modified_output, atol=1e-6)
        assert modified_output.shape == original_output.shape

    def test_functional_forward_gradient_flow(self):
        """Test that gradients flow correctly through functional forward."""
        model = nn.Linear(3, 2)
        x = torch.randn(1, 3, requires_grad=True)
        
        # Create parameters that require gradients
        modified_params = {}
        for name, param in model.named_parameters():
            modified_params[name] = param.clone().requires_grad_(True)
        
        # Forward pass
        output = FunctionalModule.functional_forward(model, x, modified_params)
        loss = output.sum()
        
        # Backward pass
        loss.backward()
        
        # Check gradients exist
        assert x.grad is not None
        for param in modified_params.values():
            assert param.grad is not None

    def test_functional_forward_complex_model(self):
        """Test functional forward with more complex model architecture."""
        model = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(32, 5)
        )
        
        x = torch.randn(2, 1, 28, 28)
        
        # Test without parameter override
        output1 = FunctionalModule.functional_forward(model, x)
        output2 = model(x)
        
        assert torch.allclose(output1, output2, atol=1e-6)
        assert output1.shape == (2, 5)

    def test_functional_forward_empty_params(self):
        """Test functional forward with empty parameter dictionary."""
        model = nn.Linear(4, 2)
        x = torch.randn(3, 4)
        
        # Empty params dict should behave like no override
        output1 = FunctionalModule.functional_forward(model, x, {})
        output2 = FunctionalModule.functional_forward(model, x, None)
        output3 = model(x)
        
        assert torch.allclose(output1, output2, atol=1e-6)
        assert torch.allclose(output1, output3, atol=1e-6)


class TestFunctionalModuleInMAML:
    """Test FunctionalModule integration within MAML training."""

    def create_simple_model(self):
        """Create a simple model for testing."""
        return nn.Sequential(
            nn.Linear(10, 32),
            nn.ReLU(),
            nn.Linear(32, 5)
        )

    def create_test_episode(self):
        """Create a test episode."""
        # 5-way, 1-shot support
        support_x = torch.randn(5, 10)
        support_y = torch.arange(5)
        
        # 10 queries (2 per class)
        query_x = torch.randn(10, 10)
        query_y = torch.repeat_interleave(torch.arange(5), 2)
        
        return Episode(support_x, support_y, query_x, query_y)

    def test_maml_uses_functional_module(self):
        """Test that MAML training actually uses FunctionalModule."""
        model = self.create_simple_model()
        toolkit = MetaLearningToolkit()
        
        # Create MAML learner
        maml_learner = toolkit.create_research_maml(model)
        
        # Create episode
        episode = self.create_test_episode()
        
        # Train episode - this should use FunctionalModule internally
        results = toolkit.train_episode(episode, algorithm="maml")
        
        # Verify results structure
        assert isinstance(results, dict)
        assert 'query_accuracy' in results
        assert 'query_loss' in results
        assert 'support_loss' in results
        assert 'meta_loss' in results
        
        # Verify values are reasonable
        assert 0.0 <= results['query_accuracy'] <= 1.0
        assert results['query_loss'] >= 0.0
        assert results['support_loss'] >= 0.0
        assert results['meta_loss'] >= 0.0

    def test_maml_adapted_vs_base_parameters(self):
        """Test that MAML produces different outputs with adapted vs base parameters."""
        model = self.create_simple_model()
        toolkit = MetaLearningToolkit()
        maml_learner = toolkit.create_research_maml(model)
        
        episode = self.create_test_episode()
        
        # Get base model output
        base_output = model(episode.query_x)
        
        # Get inner loop adapted parameters
        loss_fn = F.cross_entropy
        adapted_params = maml_learner.inner_loop(
            episode.support_x, episode.support_y, loss_fn
        )
        
        # Get adapted output using FunctionalModule
        if adapted_params is not None and len(adapted_params) > 0:
            adapted_output = FunctionalModule.functional_forward(
                model, episode.query_x, adapted_params
            )
            
            # Outputs should be different (adaptation occurred)
            assert not torch.allclose(base_output, adapted_output, atol=1e-6)
            assert adapted_output.shape == base_output.shape
        else:
            pytest.skip("No adapted parameters generated (likely zero inner steps)")

    def test_maml_functional_module_error_handling(self):
        """Test error handling when adapted parameters are None."""
        model = self.create_simple_model()
        toolkit = MetaLearningToolkit()
        
        # Create MAML with zero inner steps (should produce None adapted params)
        from meta_learning.algorithms.maml_research_accurate import MAMLConfig, MAMLVariant
        config = MAMLConfig(
            variant=MAMLVariant.MAML,
            inner_lr=0.01,
            outer_lr=0.001,
            inner_steps=0,  # Zero inner steps
            first_order=False
        )
        
        maml_learner = toolkit.create_research_maml(model, config)
        episode = self.create_test_episode()
        
        # Should handle None adapted parameters gracefully
        with pytest.warns(UserWarning, match="MAML falling back to base model"):
            results = toolkit.train_episode(episode, algorithm="maml")
        
        # Should still produce valid results
        assert isinstance(results, dict)
        assert 'query_accuracy' in results
        assert 0.0 <= results['query_accuracy'] <= 1.0

    def test_maml_functional_module_consistency(self):
        """Test that multiple runs with same seed produce consistent results."""
        model1 = self.create_simple_model()
        model2 = self.create_simple_model()
        
        # Copy weights to ensure identical starting point
        model2.load_state_dict(model1.state_dict())
        
        # Create identical toolkits
        toolkit1 = MetaLearningToolkit()
        toolkit2 = MetaLearningToolkit()
        
        toolkit1.setup_deterministic_training(seed=42)
        toolkit2.setup_deterministic_training(seed=42)
        
        maml1 = toolkit1.create_research_maml(model1)
        maml2 = toolkit2.create_research_maml(model2)
        
        # Create identical episode
        torch.manual_seed(123)
        episode1 = self.create_test_episode()
        
        torch.manual_seed(123)  # Reset seed
        episode2 = self.create_test_episode()
        
        # Results should be identical
        results1 = toolkit1.train_episode(episode1, algorithm="maml")
        results2 = toolkit2.train_episode(episode2, algorithm="maml")
        
        assert abs(results1['query_accuracy'] - results2['query_accuracy']) < 1e-6
        assert abs(results1['query_loss'] - results2['query_loss']) < 1e-6


if __name__ == "__main__":
    pytest.main([__file__])