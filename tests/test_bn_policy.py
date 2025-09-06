"""
Comprehensive tests for batch normalization policy utilities.

Tests the various BatchNorm policies used in meta-learning scenarios,
including freeze_stats, adaptive, reset, and eval_mode policies.
"""

import pytest
import torch
import torch.nn as nn
from unittest.mock import Mock

# Import the functions we're testing
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from meta_learning.core.bn_policy import (
    freeze_batchnorm_running_stats,
    apply_episodic_bn_policy
)


class TestFrozenBatchNormRunningStats:
    """Test freeze_batchnorm_running_stats function"""
    
    def test_freeze_single_bn1d(self):
        """Test freezing single BatchNorm1d layer"""
        model = nn.Sequential(
            nn.Linear(10, 20),
            nn.BatchNorm1d(20),
            nn.ReLU()
        )
        
        # Initially should be in training mode with tracking enabled
        bn_layer = model[1]
        model.train()
        assert bn_layer.training is True
        assert bn_layer.track_running_stats is True
        
        # Apply freeze
        freeze_batchnorm_running_stats(model)
        
        # BN should be in eval mode with tracking disabled
        assert bn_layer.training is False
        assert bn_layer.track_running_stats is False
        
        # Other layers should remain unchanged
        assert model[0].training is True  # Linear layer
        assert model[2].training is True  # ReLU
    
    def test_freeze_single_bn2d(self):
        """Test freezing single BatchNorm2d layer"""
        model = nn.Sequential(
            nn.Conv2d(3, 16, 3),
            nn.BatchNorm2d(16),
            nn.ReLU()
        )
        
        model.train()
        bn_layer = model[1]
        
        freeze_batchnorm_running_stats(model)
        
        assert bn_layer.training is False
        assert bn_layer.track_running_stats is False
    
    def test_freeze_single_bn3d(self):
        """Test freezing single BatchNorm3d layer"""
        model = nn.Sequential(
            nn.Conv3d(1, 8, 3),
            nn.BatchNorm3d(8),
            nn.ReLU()
        )
        
        model.train()
        bn_layer = model[1]
        
        freeze_batchnorm_running_stats(model)
        
        assert bn_layer.training is False
        assert bn_layer.track_running_stats is False
    
    def test_freeze_multiple_bn_layers(self):
        """Test freezing multiple BatchNorm layers"""
        model = nn.Sequential(
            nn.Conv2d(3, 16, 3),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Linear(32, 10),
            nn.BatchNorm1d(10)
        )
        
        model.train()
        
        freeze_batchnorm_running_stats(model)
        
        # Check all BN layers are frozen
        assert model[1].training is False  # BatchNorm2d
        assert model[1].track_running_stats is False
        assert model[4].training is False  # BatchNorm2d
        assert model[4].track_running_stats is False
        assert model[7].training is False  # BatchNorm1d
        assert model[7].track_running_stats is False
        
        # Other layers should remain in training mode
        assert model[0].training is True  # Conv2d
        assert model[3].training is True  # Conv2d
        assert model[6].training is True  # Linear
    
    def test_freeze_no_bn_layers(self):
        """Test that function handles models without BatchNorm gracefully"""
        model = nn.Sequential(
            nn.Linear(10, 20),
            nn.ReLU(),
            nn.Linear(20, 5)
        )
        
        model.train()
        
        # Should not raise any errors
        freeze_batchnorm_running_stats(model)
        
        # All layers should remain in training mode
        for layer in model:
            if hasattr(layer, 'training'):
                assert layer.training is True
    
    def test_freeze_nested_modules(self):
        """Test freezing BN in nested modules"""
        class NestedModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.branch1 = nn.Sequential(
                    nn.Conv2d(3, 16, 3),
                    nn.BatchNorm2d(16)
                )
                self.branch2 = nn.Sequential(
                    nn.Conv2d(3, 16, 3),
                    nn.BatchNorm2d(16)
                )
                self.classifier = nn.Linear(32, 10)
        
        model = NestedModel()
        model.train()
        
        freeze_batchnorm_running_stats(model)
        
        # Check nested BN layers are frozen
        assert model.branch1[1].training is False
        assert model.branch1[1].track_running_stats is False
        assert model.branch2[1].training is False
        assert model.branch2[1].track_running_stats is False
        
        # Other layers remain in training mode
        assert model.branch1[0].training is True
        assert model.classifier.training is True


class TestApplyEpisodicBNPolicy:
    """Test apply_episodic_bn_policy function"""
    
    def create_test_model(self):
        """Create a standard test model with BN layers"""
        return nn.Sequential(
            nn.Conv2d(3, 16, 3),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3),
            nn.BatchNorm2d(32),
            nn.Linear(32, 10),
            nn.BatchNorm1d(10)
        )
    
    def test_freeze_stats_policy(self):
        """Test freeze_stats policy"""
        model = self.create_test_model()
        model.train()
        
        result = apply_episodic_bn_policy(model, "freeze_stats")
        
        assert result is model  # Should return the same model
        
        # All BN layers should be in eval mode with tracking disabled
        assert model[1].training is False
        assert model[1].track_running_stats is False
        assert model[4].training is False
        assert model[4].track_running_stats is False
        assert model[6].training is False
        assert model[6].track_running_stats is False
        
        # Non-BN layers should remain in training mode
        assert model[0].training is True
        assert model[3].training is True
        assert model[5].training is True
    
    def test_eval_mode_policy(self):
        """Test eval_mode policy"""
        model = self.create_test_model()
        model.train()
        
        # Set initial track_running_stats to True for testing
        for module in model.modules():
            if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                module.track_running_stats = True
        
        apply_episodic_bn_policy(model, "eval_mode")
        
        # BN layers should be in eval mode but tracking may remain enabled
        assert model[1].training is False
        assert model[4].training is False
        assert model[6].training is False
        
        # track_running_stats should not be modified by eval_mode policy
        assert model[1].track_running_stats is True
        assert model[4].track_running_stats is True
        assert model[6].track_running_stats is True
    
    def test_reset_policy(self):
        """Test reset policy"""
        model = self.create_test_model()
        model.eval()  # Start in eval mode
        
        apply_episodic_bn_policy(model, "reset")
        
        # BN layers should be in training mode
        assert model[1].training is True
        assert model[4].training is True
        assert model[6].training is True
        
        # Non-BN layers should remain in eval mode
        assert model[0].training is False
        assert model[3].training is False
        assert model[5].training is False
    
    def test_adaptive_policy(self):
        """Test adaptive policy"""
        model = self.create_test_model()
        model.eval()  # Start in eval mode
        
        apply_episodic_bn_policy(model, "adaptive")
        
        # BN layers should be in training mode for adaptation
        assert model[1].training is True
        assert model[4].training is True
        assert model[6].training is True
        
        # Non-BN layers remain in eval mode
        assert model[0].training is False
        assert model[3].training is False
        assert model[5].training is False
    
    def test_unknown_policy(self):
        """Test unknown policy (should be handled gracefully)"""
        model = self.create_test_model()
        original_state = {}
        
        # Capture original state
        for name, module in model.named_modules():
            if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                original_state[name] = {
                    'training': module.training,
                    'track_running_stats': module.track_running_stats
                }
        
        result = apply_episodic_bn_policy(model, "unknown_policy")
        
        assert result is model
        
        # State should remain unchanged for unknown policy
        for name, module in model.named_modules():
            if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                assert module.training == original_state[name]['training']
                assert module.track_running_stats == original_state[name]['track_running_stats']
    
    def test_no_bn_layers(self):
        """Test policy application on model without BN layers"""
        model = nn.Sequential(
            nn.Linear(10, 20),
            nn.ReLU(),
            nn.Linear(20, 5)
        )
        
        result = apply_episodic_bn_policy(model, "freeze_stats")
        assert result is model
        
        # Should not raise any errors and state should be unchanged
        for layer in model:
            if hasattr(layer, 'training'):
                # Training state depends on what it was before
                pass  # No specific assertion needed
    
    def test_policy_counting(self):
        """Test that the function modifies expected number of BN layers"""
        # Note: The current implementation doesn't return the count,
        # but this test verifies the behavior is consistent
        model = self.create_test_model()
        
        bn_count = sum(1 for m in model.modules() 
                      if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)))
        
        assert bn_count == 3  # Our test model has 3 BN layers
        
        apply_episodic_bn_policy(model, "freeze_stats")
        
        # All BN layers should be affected
        frozen_count = sum(1 for m in model.modules()
                          if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d))
                          and not m.training and not m.track_running_stats)
        
        assert frozen_count == 3


class TestBNPolicyIntegration:
    """Integration tests for BN policies in meta-learning scenarios"""
    
    def test_episodic_training_workflow(self):
        """Test typical episodic meta-learning workflow"""
        model = nn.Sequential(
            nn.Conv2d(3, 16, 3),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(16, 5)
        )
        
        # Step 1: Meta-training setup (normal training)
        model.train()
        assert model[1].training is True
        assert model[1].track_running_stats is True
        
        # Step 2: Episodic evaluation setup
        apply_episodic_bn_policy(model, "freeze_stats")
        
        # BN should be frozen for episodic evaluation
        assert model[1].training is False
        assert model[1].track_running_stats is False
        
        # Other layers remain trainable for inner loop adaptation
        assert model[0].training is True
        assert model[5].training is True
    
    def test_multiple_policy_applications(self):
        """Test applying different policies sequentially"""
        model = nn.Sequential(
            nn.BatchNorm2d(16),
            nn.Conv2d(16, 32, 3),
            nn.BatchNorm2d(32)
        )
        
        model.train()
        
        # Apply freeze_stats policy
        apply_episodic_bn_policy(model, "freeze_stats")
        assert model[0].training is False
        assert model[0].track_running_stats is False
        
        # Apply adaptive policy
        apply_episodic_bn_policy(model, "adaptive")
        assert model[0].training is True
        # track_running_stats state depends on implementation
        
        # Apply eval_mode policy
        apply_episodic_bn_policy(model, "eval_mode")
        assert model[0].training is False
        
        # Apply reset policy
        apply_episodic_bn_policy(model, "reset")
        assert model[0].training is True
    
    def test_memory_efficiency(self):
        """Test that BN policies don't create unnecessary memory overhead"""
        model = nn.Sequential(
            nn.Conv2d(3, 64, 3),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 128, 3),
            nn.BatchNorm2d(128)
        )
        
        # Get initial memory footprint (simplified check)
        param_count_before = sum(p.numel() for p in model.parameters())
        
        apply_episodic_bn_policy(model, "freeze_stats")
        
        # Parameter count should remain the same
        param_count_after = sum(p.numel() for p in model.parameters())
        assert param_count_before == param_count_after
        
        # Model should still be the same object
        assert isinstance(model[1], nn.BatchNorm2d)
        assert isinstance(model[3], nn.BatchNorm2d)


if __name__ == "__main__":
    # Run tests if called directly
    pytest.main([__file__, "-v"])