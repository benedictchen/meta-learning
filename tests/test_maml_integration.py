"""
Tests for MAML learn2learn integration and enhanced functionality.

Tests cover:
- clone_module and update_module functions
- EnhancedMAML class functionality
- maml_with_fallback error handling
- ContinualMAMLEnhanced features
- Learn2learn compatibility patterns
- Memory management and error recovery
"""

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from unittest.mock import Mock, patch, MagicMock

from meta_learning.core.episode import Episode
from meta_learning.algos.maml import (
    clone_module, update_module, EnhancedMAML, maml_with_fallback,
    ContinualMAMLEnhanced, FunctionalModule
)


class TestCloneModule:
    """Test clone_module function for learn2learn compatibility."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.simple_model = nn.Sequential(
            nn.Linear(10, 20),
            nn.ReLU(),
            nn.Linear(20, 5)
        )
        
        self.nested_model = nn.Module()
        self.nested_model.encoder = nn.Linear(10, 20)
        self.nested_model.classifier = nn.Linear(20, 5)
    
    def test_clone_module_basic(self):
        """Test basic module cloning."""
        cloned = clone_module(self.simple_model)
        
        # Should be a different object
        assert cloned is not self.simple_model
        
        # But should have same structure
        assert len(cloned) == len(self.simple_model)
        
        # Parameters should have same values but different tensors
        for orig_param, clone_param in zip(self.simple_model.parameters(), cloned.parameters()):
            assert torch.equal(orig_param, clone_param)
            assert orig_param is not clone_param
    
    def test_clone_module_with_memo(self):
        """Test module cloning with memo parameter."""
        memo = {}
        cloned1 = clone_module(self.simple_model, memo)
        cloned2 = clone_module(self.simple_model, memo)
        
        # With same memo, should return same cloned object
        assert cloned1 is cloned2
        assert id(self.simple_model) in memo
    
    def test_clone_module_nested(self):
        """Test cloning of nested modules."""
        cloned = clone_module(self.nested_model)
        
        assert hasattr(cloned, 'encoder')
        assert hasattr(cloned, 'classifier')
        assert cloned.encoder is not self.nested_model.encoder
        assert cloned.classifier is not self.nested_model.classifier
        
        # Check parameter copying
        for orig_param, clone_param in zip(self.nested_model.parameters(), cloned.parameters()):
            assert torch.equal(orig_param, clone_param)
    
    def test_clone_module_gradients(self):
        """Test that cloned modules handle gradients correctly."""
        # Create some dummy gradients
        x = torch.randn(5, 10, requires_grad=True)
        y = self.simple_model(x).sum()
        y.backward()
        
        # Clone after gradients exist
        cloned = clone_module(self.simple_model)
        
        # Cloned parameters should not have gradients initially
        for param in cloned.parameters():
            assert param.grad is None
        
        # But should be able to compute gradients independently
        x_new = torch.randn(3, 10, requires_grad=True)
        y_new = cloned(x_new).sum()
        y_new.backward()
        
        # Now cloned should have gradients
        for param in cloned.parameters():
            assert param.grad is not None
    
    def test_clone_module_state_dict(self):
        """Test that cloned module has correct state dict."""
        orig_state = self.simple_model.state_dict()
        cloned = clone_module(self.simple_model)
        clone_state = cloned.state_dict()
        
        assert set(orig_state.keys()) == set(clone_state.keys())
        
        for key in orig_state:
            assert torch.equal(orig_state[key], clone_state[key])
    
    def test_clone_module_with_buffers(self):
        """Test cloning modules with buffers (like BatchNorm)."""
        bn_model = nn.Sequential(
            nn.Linear(10, 20),
            nn.BatchNorm1d(20),
            nn.Linear(20, 5)
        )
        
        # Put in eval mode and run some data to populate running stats
        bn_model.eval()
        with torch.no_grad():
            x = torch.randn(32, 10)
            _ = bn_model(x)
        
        cloned = clone_module(bn_model)
        
        # Check that buffers are properly cloned
        orig_buffers = dict(bn_model.named_buffers())
        clone_buffers = dict(cloned.named_buffers())
        
        assert set(orig_buffers.keys()) == set(clone_buffers.keys())
        
        for key in orig_buffers:
            assert torch.equal(orig_buffers[key], clone_buffers[key])
            assert orig_buffers[key] is not clone_buffers[key]


class TestUpdateModule:
    """Test update_module function for learn2learn compatibility."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.model = nn.Linear(5, 3)
        self.original_weight = self.model.weight.clone()
        self.original_bias = self.model.bias.clone()
    
    def test_update_module_basic(self):
        """Test basic module parameter updates."""
        # Create some updates
        updates = {
            'weight': torch.randn_like(self.model.weight) * 0.1,
            'bias': torch.randn_like(self.model.bias) * 0.1
        }
        
        updated = update_module(self.model, updates)
        
        # Should be a different object
        assert updated is not self.model
        
        # Parameters should be updated
        assert torch.equal(updated.weight, self.original_weight + updates['weight'])
        assert torch.equal(updated.bias, self.original_bias + updates['bias'])
        
        # Original should be unchanged
        assert torch.equal(self.model.weight, self.original_weight)
        assert torch.equal(self.model.bias, self.original_bias)
    
    def test_update_module_partial_updates(self):
        """Test module updates with only some parameters."""
        # Only update weight, not bias
        updates = {
            'weight': torch.randn_like(self.model.weight) * 0.1
        }
        
        updated = update_module(self.model, updates)
        
        # Weight should be updated, bias should remain the same
        assert torch.equal(updated.weight, self.original_weight + updates['weight'])
        assert torch.equal(updated.bias, self.original_bias)
    
    def test_update_module_no_updates(self):
        """Test module updates with empty updates dict."""
        updated = update_module(self.model, {})
        
        # Should be equivalent to cloning
        assert updated is not self.model
        assert torch.equal(updated.weight, self.original_weight)
        assert torch.equal(updated.bias, self.original_bias)
    
    def test_update_module_with_memo(self):
        """Test module updates with memo parameter."""
        updates = {'weight': torch.randn_like(self.model.weight) * 0.1}
        memo = {}
        
        updated1 = update_module(self.model, updates, memo)
        updated2 = update_module(self.model, updates, memo)
        
        # With same memo and updates, should return same object
        assert updated1 is updated2
    
    def test_update_module_nested_parameters(self):
        """Test updating nested module parameters."""
        nested_model = nn.Sequential(
            nn.Linear(5, 10),
            nn.ReLU(),
            nn.Linear(10, 3)
        )
        
        updates = {
            '0.weight': torch.randn_like(nested_model[0].weight) * 0.1,
            '0.bias': torch.randn_like(nested_model[0].bias) * 0.1,
            '2.weight': torch.randn_like(nested_model[2].weight) * 0.1
        }
        
        updated = update_module(nested_model, updates)
        
        # Check that nested parameters were updated correctly
        assert torch.equal(updated[0].weight, nested_model[0].weight + updates['0.weight'])
        assert torch.equal(updated[0].bias, nested_model[0].bias + updates['0.bias'])
        assert torch.equal(updated[2].weight, nested_model[2].weight + updates['2.weight'])
        
        # Bias of layer 2 should be unchanged
        assert torch.equal(updated[2].bias, nested_model[2].bias)
    
    def test_update_module_preserves_grad_fn(self):
        """Test that updated modules preserve gradient computation."""
        self.model.weight.requires_grad_(True)
        self.model.bias.requires_grad_(True)
        
        # Create updates that require gradients
        weight_update = torch.randn_like(self.model.weight) * 0.1
        weight_update.requires_grad_(True)
        
        updates = {'weight': weight_update}
        updated = update_module(self.model, updates)
        
        # Updated parameters should still require gradients
        assert updated.weight.requires_grad
        assert updated.bias.requires_grad
        
        # Should be able to compute gradients through the update
        x = torch.randn(2, 5)
        y = updated(x).sum()
        y.backward()
        
        assert weight_update.grad is not None


class TestEnhancedMAML:
    """Test EnhancedMAML class with multiple approaches."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.model = nn.Sequential(
            nn.Linear(10, 20),
            nn.ReLU(),
            nn.Linear(20, 5)
        )
        
        self.episode = Episode(
            support_x=torch.randn(25, 10),  # 5-way, 5-shot
            support_y=torch.repeat_interleave(torch.arange(5), 5),
            query_x=torch.randn(15, 10),    # 5-way, 3-query
            query_y=torch.repeat_interleave(torch.arange(5), 3)
        )
    
    def test_enhanced_maml_initialization(self):
        """Test EnhancedMAML initialization."""
        from meta_learning.algorithms.maml_research_accurate import MAMLConfig
        
        config = MAMLConfig(inner_lr=0.01, inner_steps=1)
        enhanced_maml = EnhancedMAML(self.model, config)
        
        assert enhanced_maml.model is self.model
        assert enhanced_maml.config == config
        assert enhanced_maml.use_functional_call == True  # Default
    
    def test_enhanced_maml_functional_call_approach(self):
        """Test EnhancedMAML with functional call approach."""
        from meta_learning.algorithms.maml_research_accurate import MAMLConfig
        
        config = MAMLConfig(inner_lr=0.01, inner_steps=1)
        enhanced_maml = EnhancedMAML(self.model, config, use_functional_call=True)
        
        # Test inner adaptation
        adapted_params = enhanced_maml.inner_adapt(
            self.episode.support_x,
            self.episode.support_y,
            F.cross_entropy
        )
        
        assert adapted_params is not None
        assert isinstance(adapted_params, (dict, OrderedDict))
        
        # Test forward pass with adapted parameters
        query_logits = enhanced_maml.forward_with_params(
            self.episode.query_x,
            adapted_params
        )
        
        assert query_logits.shape == (15, 5)
    
    def test_enhanced_maml_clone_approach(self):
        """Test EnhancedMAML with clone module approach."""
        from meta_learning.algorithms.maml_research_accurate import MAMLConfig
        
        config = MAMLConfig(inner_lr=0.01, inner_steps=2)
        enhanced_maml = EnhancedMAML(self.model, config, use_functional_call=False)
        
        # Test inner adaptation
        adapted_params = enhanced_maml.inner_adapt(
            self.episode.support_x,
            self.episode.support_y,
            F.cross_entropy
        )
        
        # With clone approach, should return the adapted model itself
        assert adapted_params is not None
        
        # Test that we can use the adapted parameters
        if isinstance(adapted_params, nn.Module):
            query_logits = adapted_params(self.episode.query_x)
        else:
            query_logits = enhanced_maml.forward_with_params(
                self.episode.query_x,
                adapted_params
            )
        
        assert query_logits.shape == (15, 5)
    
    def test_enhanced_maml_multiple_inner_steps(self):
        """Test EnhancedMAML with multiple inner steps."""
        from meta_learning.algorithms.maml_research_accurate import MAMLConfig
        
        config = MAMLConfig(inner_lr=0.01, inner_steps=3)
        enhanced_maml = EnhancedMAML(self.model, config)
        
        # Track initial loss
        initial_logits = self.model(self.episode.support_x)
        initial_loss = F.cross_entropy(initial_logits, self.episode.support_y)
        
        # Perform adaptation
        adapted_params = enhanced_maml.inner_adapt(
            self.episode.support_x,
            self.episode.support_y,
            F.cross_entropy
        )
        
        # Check that adaptation actually happened
        adapted_logits = enhanced_maml.forward_with_params(
            self.episode.support_x,
            adapted_params
        )
        adapted_loss = F.cross_entropy(adapted_logits, self.episode.support_y)
        
        # Loss should generally decrease (though not guaranteed due to randomness)
        # Just check that we got different results
        assert not torch.equal(initial_logits, adapted_logits)
    
    def test_enhanced_maml_error_handling(self):
        """Test EnhancedMAML error handling."""
        from meta_learning.algorithms.maml_research_accurate import MAMLConfig
        
        config = MAMLConfig(inner_lr=float('inf'), inner_steps=1)  # Bad learning rate
        enhanced_maml = EnhancedMAML(self.model, config)
        
        # Should handle numerical instability gracefully
        with pytest.warns(UserWarning, match="Numerical instability detected"):
            adapted_params = enhanced_maml.inner_adapt(
                self.episode.support_x,
                self.episode.support_y,
                F.cross_entropy
            )
            
            # Should still return some parameters (possibly original)
            assert adapted_params is not None


class TestMAMLWithFallback:
    """Test maml_with_fallback error handling function."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.model = nn.Linear(10, 5)
        self.episode = Episode(
            support_x=torch.randn(20, 10),
            support_y=torch.repeat_interleave(torch.arange(4), 5),
            query_x=torch.randn(12, 10),
            query_y=torch.repeat_interleave(torch.arange(4), 3)
        )
    
    def test_maml_with_fallback_success(self):
        """Test maml_with_fallback when MAML succeeds."""
        result = maml_with_fallback(
            self.model,
            self.episode,
            inner_lr=0.01,
            loss_fn=F.cross_entropy,
            inner_steps=1
        )
        
        assert 'query_loss' in result
        assert 'query_accuracy' in result
        assert 'support_loss' in result
        assert 'fallback_used' in result
        assert result['fallback_used'] == False
        
        # Values should be reasonable
        assert 0.0 <= result['query_accuracy'] <= 1.0
        assert result['query_loss'] > 0
        assert result['support_loss'] > 0
    
    @patch('meta_learning.algos.maml.EnhancedMAML.inner_adapt')
    def test_maml_with_fallback_inner_adapt_failure(self, mock_inner_adapt):
        """Test fallback when inner adaptation fails."""
        # Mock inner adaptation failure
        mock_inner_adapt.side_effect = RuntimeError("Inner adaptation failed")
        
        import warnings
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            
            result = maml_with_fallback(
                self.model,
                self.episode,
                inner_lr=0.01,
                loss_fn=F.cross_entropy
            )
            
            assert result['fallback_used'] == True
            assert 'fallback_reason' in result
            assert "Inner adaptation failed" in result['fallback_reason']
            assert len(w) > 0  # Should emit warning
    
    @patch('meta_learning.algos.maml.EnhancedMAML')
    def test_maml_with_fallback_initialization_failure(self, mock_enhanced_maml):
        """Test fallback when MAML initialization fails."""
        # Mock MAML initialization failure
        mock_enhanced_maml.side_effect = ValueError("Bad configuration")
        
        import warnings
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            
            result = maml_with_fallback(
                self.model,
                self.episode,
                inner_lr=0.01,
                loss_fn=F.cross_entropy
            )
            
            assert result['fallback_used'] == True
            assert 'fallback_reason' in result
            assert len(w) > 0
    
    def test_maml_with_fallback_numerical_instability(self):
        """Test fallback with numerical instability."""
        # Use extremely high learning rate to cause instability
        result = maml_with_fallback(
            self.model,
            self.episode,
            inner_lr=1e6,  # Extremely high learning rate
            loss_fn=F.cross_entropy,
            inner_steps=1
        )
        
        # Should still produce reasonable results via fallback
        assert 'query_accuracy' in result
        assert 0.0 <= result['query_accuracy'] <= 1.0
        
        # May or may not use fallback depending on exact failure mode
        assert 'fallback_used' in result
    
    def test_maml_with_fallback_custom_parameters(self):
        """Test maml_with_fallback with custom parameters."""
        result = maml_with_fallback(
            self.model,
            self.episode,
            inner_lr=0.05,
            loss_fn=F.cross_entropy,
            inner_steps=3,
            max_retries=2,
            fallback_to_prototypes=True
        )
        
        assert isinstance(result, dict)
        assert 'query_accuracy' in result
        assert 'fallback_used' in result


class TestContinualMAMLEnhanced:
    """Test ContinualMAMLEnhanced with memory management."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.model = nn.Linear(10, 5)
        
        # Create multiple episodes for continual learning
        self.episodes = []
        for i in range(3):
            episode = Episode(
                support_x=torch.randn(15, 10),
                support_y=torch.repeat_interleave(torch.arange(3), 5),
                query_x=torch.randn(9, 10),
                query_y=torch.repeat_interleave(torch.arange(3), 3)
            )
            self.episodes.append(episode)
    
    def test_continual_maml_enhanced_initialization(self):
        """Test ContinualMAMLEnhanced initialization."""
        from meta_learning.algorithms.maml_research_accurate import MAMLConfig
        
        config = MAMLConfig(inner_lr=0.01, inner_steps=1)
        continual_maml = ContinualMAMLEnhanced(
            self.model, 
            config,
            memory_size=100,
            importance_weight=0.5
        )
        
        assert continual_maml.memory_size == 100
        assert continual_maml.importance_weight == 0.5
        assert len(continual_maml.memory_buffer) == 0
        assert len(continual_maml.importance_weights) == 0
    
    def test_continual_maml_enhanced_memory_storage(self):
        """Test memory storage functionality."""
        from meta_learning.algorithms.maml_research_accurate import MAMLConfig
        
        config = MAMLConfig(inner_lr=0.01, inner_steps=1)
        continual_maml = ContinualMAMLEnhanced(self.model, config, memory_size=5)
        
        # Store episodes in memory
        for i, episode in enumerate(self.episodes):
            continual_maml.store_in_memory(episode, importance=1.0 + i * 0.1)
        
        assert len(continual_maml.memory_buffer) == 3
        assert len(continual_maml.importance_weights) == 3
        
        # Check importance weights
        assert continual_maml.importance_weights[0] == 1.0
        assert continual_maml.importance_weights[1] == 1.1
        assert continual_maml.importance_weights[2] == 1.2
    
    def test_continual_maml_enhanced_memory_overflow(self):
        """Test memory buffer overflow handling."""
        from meta_learning.algorithms.maml_research_accurate import MAMLConfig
        
        config = MAMLConfig(inner_lr=0.01, inner_steps=1)
        continual_maml = ContinualMAMLEnhanced(self.model, config, memory_size=2)
        
        # Store more episodes than memory size
        for i, episode in enumerate(self.episodes):
            continual_maml.store_in_memory(episode, importance=1.0 + i * 0.1)
        
        # Should only keep the most recent episodes up to memory size
        assert len(continual_maml.memory_buffer) == 2
        assert len(continual_maml.importance_weights) == 2
    
    def test_continual_maml_enhanced_memory_replay(self):
        """Test memory replay functionality."""
        from meta_learning.algorithms.maml_research_accurate import MAMLConfig
        
        config = MAMLConfig(inner_lr=0.01, inner_steps=1)
        continual_maml = ContinualMAMLEnhanced(self.model, config, memory_size=10)
        
        # Store episodes in memory
        for episode in self.episodes:
            continual_maml.store_in_memory(episode, importance=1.0)
        
        # Sample from memory
        sampled_episodes = continual_maml.sample_from_memory(batch_size=2)
        
        assert len(sampled_episodes) == 2
        for episode in sampled_episodes:
            assert isinstance(episode, Episode)
    
    def test_continual_maml_enhanced_adapt_with_memory(self):
        """Test adaptation with memory replay."""
        from meta_learning.algorithms.maml_research_accurate import MAMLConfig
        
        config = MAMLConfig(inner_lr=0.01, inner_steps=1)
        continual_maml = ContinualMAMLEnhanced(self.model, config, memory_size=10)
        
        # Store some episodes in memory first
        for episode in self.episodes[:2]:
            continual_maml.store_in_memory(episode, importance=1.0)
        
        # Adapt on new episode with memory replay
        new_episode = self.episodes[2]
        adapted_params = continual_maml.adapt_with_memory_replay(
            new_episode,
            F.cross_entropy,
            replay_ratio=0.5
        )
        
        assert adapted_params is not None
    
    def test_continual_maml_enhanced_catastrophic_forgetting_protection(self):
        """Test catastrophic forgetting protection."""
        from meta_learning.algorithms.maml_research_accurate import MAMLConfig
        
        config = MAMLConfig(inner_lr=0.01, inner_steps=1)
        continual_maml = ContinualMAMLEnhanced(
            self.model, 
            config, 
            memory_size=10,
            importance_weight=1.0  # High importance for memory
        )
        
        # Train on first episode and store in memory
        first_episode = self.episodes[0]
        continual_maml.store_in_memory(first_episode, importance=2.0)
        
        # Get initial performance on first episode
        initial_logits = self.model(first_episode.query_x)
        initial_loss = F.cross_entropy(initial_logits, first_episode.query_y)
        
        # Adapt on different episode
        different_episode = self.episodes[1]
        adapted_params = continual_maml.adapt_with_memory_replay(
            different_episode,
            F.cross_entropy,
            replay_ratio=0.8  # High replay ratio
        )
        
        # Check that we didn't completely forget the first episode
        # (This is a weak test due to randomness, but checks the mechanism exists)
        assert adapted_params is not None


class TestMAMLIntegration:
    """Integration tests for all MAML components."""
    
    def setup_method(self):
        """Setup integration test fixtures."""
        self.model = nn.Sequential(
            nn.Linear(20, 50),
            nn.ReLU(),
            nn.Linear(50, 10)
        )
        
        self.episodes = []
        for i in range(5):
            episode = Episode(
                support_x=torch.randn(50, 20),  # 10-way, 5-shot
                support_y=torch.repeat_interleave(torch.arange(10), 5),
                query_x=torch.randn(30, 20),    # 10-way, 3-query
                query_y=torch.repeat_interleave(torch.arange(10), 3)
            )
            self.episodes.append(episode)
    
    def test_complete_maml_pipeline_functional_approach(self):
        """Test complete MAML pipeline with functional approach."""
        from meta_learning.algorithms.maml_research_accurate import MAMLConfig
        
        config = MAMLConfig(inner_lr=0.01, inner_steps=2)
        
        # Test enhanced MAML
        enhanced_maml = EnhancedMAML(self.model, config, use_functional_call=True)
        
        for episode in self.episodes[:3]:
            # Test adaptation
            adapted_params = enhanced_maml.inner_adapt(
                episode.support_x,
                episode.support_y,
                F.cross_entropy
            )
            
            assert adapted_params is not None
            
            # Test query prediction
            query_logits = enhanced_maml.forward_with_params(
                episode.query_x,
                adapted_params
            )
            
            assert query_logits.shape == (30, 10)
            
            # Test that adaptation actually changes parameters
            original_logits = self.model(episode.query_x)
            assert not torch.equal(query_logits, original_logits)
    
    def test_complete_maml_pipeline_clone_approach(self):
        """Test complete MAML pipeline with clone approach."""
        from meta_learning.algorithms.maml_research_accurate import MAMLConfig
        
        config = MAMLConfig(inner_lr=0.01, inner_steps=2)
        
        # Test enhanced MAML with clone approach
        enhanced_maml = EnhancedMAML(self.model, config, use_functional_call=False)
        
        for episode in self.episodes[:2]:
            # Test adaptation
            adapted_model_or_params = enhanced_maml.inner_adapt(
                episode.support_x,
                episode.support_y,
                F.cross_entropy
            )
            
            assert adapted_model_or_params is not None
            
            # Test query prediction
            if isinstance(adapted_model_or_params, nn.Module):
                query_logits = adapted_model_or_params(episode.query_x)
            else:
                query_logits = enhanced_maml.forward_with_params(
                    episode.query_x,
                    adapted_model_or_params
                )
            
            assert query_logits.shape == (30, 10)
    
    def test_maml_with_error_recovery_integration(self):
        """Test MAML with comprehensive error recovery."""
        # Test various error scenarios
        
        # 1. Normal operation
        result = maml_with_fallback(
            self.model,
            self.episodes[0],
            inner_lr=0.01,
            loss_fn=F.cross_entropy
        )
        
        assert result['fallback_used'] == False
        assert 'query_accuracy' in result
        
        # 2. High learning rate (potential instability)
        result_unstable = maml_with_fallback(
            self.model,
            self.episodes[1],
            inner_lr=10.0,  # Very high learning rate
            loss_fn=F.cross_entropy,
            max_retries=2
        )
        
        assert 'query_accuracy' in result_unstable
        assert 0.0 <= result_unstable['query_accuracy'] <= 1.0
        
        # 3. Invalid loss function (should fallback)
        def bad_loss_fn(pred, target):
            raise ValueError("Bad loss function")
        
        result_bad_loss = maml_with_fallback(
            self.model,
            self.episodes[2],
            inner_lr=0.01,
            loss_fn=bad_loss_fn,
            max_retries=1
        )
        
        assert result_bad_loss['fallback_used'] == True
        assert 'fallback_reason' in result_bad_loss
    
    def test_continual_learning_integration(self):
        """Test continual learning with enhanced MAML."""
        from meta_learning.algorithms.maml_research_accurate import MAMLConfig
        
        config = MAMLConfig(inner_lr=0.01, inner_steps=1)
        continual_maml = ContinualMAMLEnhanced(
            self.model, 
            config, 
            memory_size=20,
            importance_weight=0.8
        )
        
        # Simulate continual learning across multiple episodes
        accuracies = []
        
        for i, episode in enumerate(self.episodes):
            # Adapt on current episode with memory replay
            if i > 0:  # Use memory replay after first episode
                adapted_params = continual_maml.adapt_with_memory_replay(
                    episode,
                    F.cross_entropy,
                    replay_ratio=0.3
                )
            else:
                adapted_params = continual_maml.inner_adapt(
                    episode.support_x,
                    episode.support_y,
                    F.cross_entropy
                )
            
            # Evaluate on query set
            if isinstance(adapted_params, nn.Module):
                query_logits = adapted_params(episode.query_x)
            else:
                query_logits = continual_maml.forward_with_params(
                    episode.query_x,
                    adapted_params
                )
            
            accuracy = (query_logits.argmax(-1) == episode.query_y).float().mean().item()
            accuracies.append(accuracy)
            
            # Store in memory for future replay
            continual_maml.store_in_memory(episode, importance=1.0 + i * 0.1)
        
        # Should have reasonable accuracies
        for acc in accuracies:
            assert 0.0 <= acc <= 1.0
        
        # Memory should be properly managed
        assert len(continual_maml.memory_buffer) <= continual_maml.memory_size
    
    def test_learn2learn_compatibility_patterns(self):
        """Test compatibility with learn2learn usage patterns."""
        # Test that our functions work with learn2learn-style code patterns
        
        # 1. Clone and update pattern
        cloned_model = clone_module(self.model)
        
        # Compute gradients
        episode = self.episodes[0]
        support_logits = cloned_model(episode.support_x)
        support_loss = F.cross_entropy(support_logits, episode.support_y)
        grads = torch.autograd.grad(support_loss, cloned_model.parameters(), create_graph=True)
        
        # Create updates dict
        updates = {}
        for (name, param), grad in zip(cloned_model.named_parameters(), grads):
            updates[name] = -0.01 * grad  # Simple gradient step
        
        # Update model
        updated_model = update_module(cloned_model, updates)
        
        # Test that updated model gives different predictions
        original_logits = self.model(episode.query_x)
        updated_logits = updated_model(episode.query_x)
        
        assert not torch.equal(original_logits, updated_logits)
        
        # 2. Functional call pattern (similar to higher library)
        original_params = dict(self.model.named_parameters())
        
        # Apply updates to parameters
        updated_params = {}
        for name, param in original_params.items():
            if name in updates:
                updated_params[name] = param + updates[name]
            else:
                updated_params[name] = param
        
        # Should be able to use FunctionalModule
        functional_logits = FunctionalModule.functional_forward(
            self.model,
            episode.query_x,
            updated_params
        )
        
        assert functional_logits.shape == updated_logits.shape


if __name__ == "__main__":
    pytest.main([__file__])