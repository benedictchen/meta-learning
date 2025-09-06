"""
Comprehensive tests for BatchNorm policy implementation.

Tests leakage prevention, normalization replacement, and episodic isolation
for meta-learning scenarios.
"""
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from unittest.mock import MagicMock, patch
from typing import Dict, List, Tuple, Any

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../src'))

from meta_learning.research_patches.batch_norm_policy import (
    apply_episodic_bn_policy,
    EpisodicBatchNormManager,
    EpisodicMode,
    EpisodicNormalizationGuard,
    BatchNormLeakageDetector,
    validate_few_shot_model
)


class TestBatchNormPolicyApplication:
    """Test BatchNorm policy application and replacement."""
    
    def test_group_norm_replacement(self):
        """Test BatchNorm -> GroupNorm replacement."""
        # Model with BatchNorm layers
        model = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(128, 10)
        )
        
        # Apply GroupNorm policy
        model_fixed = apply_episodic_bn_policy(model, policy="group_norm", num_groups=8)
        
        # Check that BatchNorm layers are replaced
        has_batch_norm = any('BatchNorm' in str(type(m)) for m in model_fixed.modules())
        has_group_norm = any('GroupNorm' in str(type(m)) for m in model_fixed.modules())
        
        assert not has_batch_norm, "BatchNorm layers should be replaced"
        assert has_group_norm, "Should have GroupNorm layers"
        
        # Check that GroupNorm has correct number of groups
        for module in model_fixed.modules():
            if isinstance(module, nn.GroupNorm):
                assert module.num_groups == 8
    
    def test_layer_norm_replacement(self):
        """Test BatchNorm -> LayerNorm replacement."""
        model = nn.Sequential(
            nn.Linear(100, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Linear(32, 10)
        )
        
        model_fixed = apply_episodic_bn_policy(model, policy="layer_norm")
        
        has_batch_norm = any('BatchNorm' in str(type(m)) for m in model_fixed.modules())
        has_layer_norm = any('LayerNorm' in str(type(m)) for m in model_fixed.modules())
        
        assert not has_batch_norm
        assert has_layer_norm
    
    def test_instance_norm_replacement(self):
        """Test BatchNorm -> InstanceNorm replacement."""
        model = nn.Sequential(
            nn.Conv2d(3, 64, 3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3),
            nn.BatchNorm2d(128)
        )
        
        model_fixed = apply_episodic_bn_policy(model, policy="instance_norm")
        
        has_batch_norm = any('BatchNorm' in str(type(m)) for m in model_fixed.modules())
        has_instance_norm = any('InstanceNorm' in str(type(m)) for m in model_fixed.modules())
        
        assert not has_batch_norm
        assert has_instance_norm
    
    def test_freeze_policy(self):
        """Test BatchNorm freezing policy."""
        model = nn.Sequential(
            nn.Conv2d(3, 64, 3),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        
        # Train mode initially
        model.train()
        bn_layer = model[1]
        assert bn_layer.training
        
        model_fixed = apply_episodic_bn_policy(model, policy="freeze")
        bn_layer_fixed = model_fixed[1]
        
        # BatchNorm should be in eval mode (frozen)
        assert not bn_layer_fixed.training
        assert isinstance(bn_layer_fixed, nn.BatchNorm2d)  # Still BatchNorm, but frozen
    
    def test_mixed_batch_norm_types(self):
        """Test policy application on models with different BatchNorm types."""
        model = nn.Sequential(
            nn.Conv1d(10, 32, 3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv3d(64, 128, 3),
            nn.BatchNorm3d(128)
        )
        
        model_fixed = apply_episodic_bn_policy(model, policy="instance_norm")
        
        # Check all BatchNorm variants are replaced
        has_batch_norm = any('BatchNorm' in str(type(m)) for m in model_fixed.modules())
        has_instance_norm = any('InstanceNorm' in str(type(m)) for m in model_fixed.modules())
        
        assert not has_batch_norm
        assert has_instance_norm
        
        # Check correct dimensionality replacements
        instance_norm_types = [type(m) for m in model_fixed.modules() if 'InstanceNorm' in str(type(m))]
        expected_types = [nn.InstanceNorm1d, nn.InstanceNorm2d, nn.InstanceNorm3d]
        
        for expected_type in expected_types:
            assert any(t == expected_type for t in instance_norm_types)
    
    def test_preserve_non_batch_norm_layers(self):
        """Test that non-BatchNorm layers are preserved."""
        model = nn.Sequential(
            nn.Conv2d(3, 64, 3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, 32),
            nn.LayerNorm(32),  # Should be preserved
            nn.Linear(32, 10)
        )
        
        model_fixed = apply_episodic_bn_policy(model, policy="group_norm")
        
        # Check that other layers are preserved
        has_conv = any(isinstance(m, nn.Conv2d) for m in model_fixed.modules())
        has_relu = any(isinstance(m, nn.ReLU) for m in model_fixed.modules())
        has_dropout = any(isinstance(m, nn.Dropout) for m in model_fixed.modules())
        has_linear = any(isinstance(m, nn.Linear) for m in model_fixed.modules())
        has_layer_norm = any(isinstance(m, nn.LayerNorm) for m in model_fixed.modules())
        
        assert has_conv
        assert has_relu
        assert has_dropout
        assert has_linear
        assert has_layer_norm


class TestEpisodicBatchNormManager:
    """Test EpisodicBatchNormManager functionality."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.model = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        
        self.manager = EpisodicBatchNormManager()
    
    def test_prepare_model_for_episodes(self):
        """Test model preparation for episodic training."""
        # Initially in train mode
        self.model.train()
        bn_layers = [m for m in self.model.modules() if isinstance(m, nn.BatchNorm2d)]
        
        for bn in bn_layers:
            assert bn.training
            assert bn.track_running_stats
        
        # Prepare for episodes
        self.manager.prepare_model_for_episodes(self.model)
        
        # BatchNorm layers should be frozen
        for bn in bn_layers:
            assert not bn.training
            assert not bn.track_running_stats
    
    def test_restore_model_after_episodes(self):
        """Test model restoration after episodic training."""
        # Store original state
        original_training = self.model.training
        original_stats_tracking = []
        
        bn_layers = [m for m in self.model.modules() if isinstance(m, nn.BatchNorm2d)]
        for bn in bn_layers:
            original_stats_tracking.append(bn.track_running_stats)
        
        # Prepare and then restore
        self.manager.prepare_model_for_episodes(self.model)
        self.manager.restore_model_after_episodes(self.model)
        
        # Should be back to original state
        assert self.model.training == original_training
        for i, bn in enumerate(bn_layers):
            assert bn.track_running_stats == original_stats_tracking[i]
    
    def test_batch_norm_stats_isolation(self):
        """Test that BatchNorm stats don't leak between episodes."""
        bn_layer = nn.BatchNorm2d(64)
        bn_layer.train()
        
        # Simulate episode 1
        episode1_data = torch.randn(32, 64, 8, 8)
        self.manager.prepare_model_for_episodes(bn_layer)
        
        with torch.no_grad():
            _ = bn_layer(episode1_data)
        
        # Store stats after episode 1
        running_mean_after_ep1 = bn_layer.running_mean.clone()
        running_var_after_ep1 = bn_layer.running_var.clone()
        
        # Simulate episode 2 with different data
        episode2_data = torch.randn(32, 64, 8, 8) + 5.0  # Different distribution
        
        with torch.no_grad():
            _ = bn_layer(episode2_data)
        
        # Stats should be the same (no leakage)
        assert torch.equal(bn_layer.running_mean, running_mean_after_ep1)
        assert torch.equal(bn_layer.running_var, running_var_after_ep1)


class TestEpisodicMode:
    """Test EpisodicMode context manager."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.model = nn.Sequential(
            nn.Linear(10, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 5)
        )
    
    def test_episodic_mode_context_manager(self):
        """Test EpisodicMode as context manager."""
        # Store original state
        bn_layer = self.model[1]
        original_training = bn_layer.training
        original_track_stats = bn_layer.track_running_stats
        
        # Use context manager
        with EpisodicMode(self.model) as mode:
            # Inside context: should be prepared for episodes
            assert not bn_layer.training
            assert not bn_layer.track_running_stats
        
        # Outside context: should be restored
        assert bn_layer.training == original_training
        assert bn_layer.track_running_stats == original_track_stats
    
    def test_episodic_mode_exception_handling(self):
        """Test EpisodicMode properly restores state even on exceptions."""
        bn_layer = self.model[1]
        original_training = bn_layer.training
        original_track_stats = bn_layer.track_running_stats
        
        try:
            with EpisodicMode(self.model):
                # Verify state is changed
                assert not bn_layer.training
                # Raise exception
                raise ValueError("Test exception")
        except ValueError:
            pass
        
        # Should still be restored
        assert bn_layer.training == original_training
        assert bn_layer.track_running_stats == original_track_stats
    
    def test_episodic_mode_with_monitoring(self):
        """Test EpisodicMode with statistics monitoring."""
        monitor = MagicMock()
        
        with EpisodicMode(self.model, monitor=monitor) as mode:
            # Monitor should be called
            monitor.capture_initial_stats.assert_called_once()
        
        # Monitor should be called on exit
        monitor.validate_stats_isolation.assert_called_once()


class TestEpisodicNormalizationGuard:
    """Test EpisodicNormalizationGuard leak detection."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.model = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Linear(32, 10)
        )
        
        self.guard = EpisodicNormalizationGuard(strict_mode=True)
    
    def test_validate_episode_isolation_pass(self):
        """Test validation passes for proper episodic isolation."""
        # Prepare model properly
        manager = EpisodicBatchNormManager()
        manager.prepare_model_for_episodes(self.model)
        
        support_data = torch.randn(10, 3, 32, 32)
        query_data = torch.randn(15, 3, 32, 32)
        
        result = self.guard.validate_episode_isolation(self.model, support_data, query_data)
        
        assert result['passed']
        assert len(result['warnings']) == 0
        assert len(result['errors']) == 0
    
    def test_validate_episode_isolation_fail(self):
        """Test validation fails for improper setup."""
        # Don't prepare model (keep BatchNorm in training mode)
        self.model.train()
        
        support_data = torch.randn(10, 3, 32, 32)
        query_data = torch.randn(15, 3, 32, 32)
        
        result = self.guard.validate_episode_isolation(self.model, support_data, query_data)
        
        assert not result['passed']
        assert len(result['errors']) > 0
        
        # Should detect BatchNorm in training mode
        error_messages = ' '.join(result['errors'])
        assert 'training mode' in error_messages.lower()
    
    def test_detect_running_stats_leakage(self):
        """Test detection of running statistics leakage."""
        bn_layer = nn.BatchNorm2d(32)
        bn_layer.train()
        bn_layer.track_running_stats = True
        
        # This should be flagged as potential leakage
        issues = self.guard._check_batch_norm_configuration(bn_layer, "test_layer")
        
        assert len(issues) > 0
        issue_text = ' '.join(issues).lower()
        assert 'training mode' in issue_text or 'track_running_stats' in issue_text
    
    def test_strict_vs_lenient_mode(self):
        """Test difference between strict and lenient validation modes."""
        # Create model with potential issues
        self.model.train()
        
        support_data = torch.randn(5, 3, 32, 32)
        query_data = torch.randn(5, 3, 32, 32)
        
        # Strict mode
        guard_strict = EpisodicNormalizationGuard(strict_mode=True)
        result_strict = guard_strict.validate_episode_isolation(self.model, support_data, query_data)
        
        # Lenient mode
        guard_lenient = EpisodicNormalizationGuard(strict_mode=False)
        result_lenient = guard_lenient.validate_episode_isolation(self.model, support_data, query_data)
        
        # Strict mode should be more restrictive
        assert len(result_strict['errors']) >= len(result_lenient['errors'])
        assert not result_strict['passed'] or result_lenient['passed']


class TestBatchNormLeakageDetector:
    """Test BatchNormLeakageDetector functionality."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.detector = BatchNormLeakageDetector()
        
        self.model = nn.Sequential(
            nn.Conv2d(3, 64, 3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 32, 3),
            nn.BatchNorm2d(32)
        )
    
    def test_capture_initial_stats(self):
        """Test capturing initial BatchNorm statistics."""
        self.detector.capture_initial_stats(self.model)
        
        # Should have captured stats for both BatchNorm layers
        assert len(self.detector.initial_stats) == 2
        
        for layer_name, stats in self.detector.initial_stats.items():
            assert 'running_mean' in stats
            assert 'running_var' in stats
            assert 'num_batches_tracked' in stats
    
    def test_detect_no_leakage(self):
        """Test detection when there's no leakage."""
        # Capture initial stats
        self.detector.capture_initial_stats(self.model)
        
        # Prepare model properly (freeze BatchNorm)
        for module in self.model.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.eval()
                module.track_running_stats = False
        
        # Process some data
        data = torch.randn(16, 3, 28, 28)
        with torch.no_grad():
            _ = self.model(data)
        
        # Check for leakage
        leakage_report = self.detector.detect_leakage(self.model)
        
        assert not leakage_report['has_leakage']
        assert len(leakage_report['leaked_layers']) == 0
    
    def test_detect_leakage(self):
        """Test detection when there is leakage."""
        # Capture initial stats
        self.detector.capture_initial_stats(self.model)
        
        # Keep BatchNorm in training mode (will cause leakage)
        self.model.train()
        
        # Process data
        data = torch.randn(16, 3, 28, 28)
        with torch.no_grad():
            _ = self.model(data)
        
        # Check for leakage
        leakage_report = self.detector.detect_leakage(self.model)
        
        assert leakage_report['has_leakage']
        assert len(leakage_report['leaked_layers']) > 0
    
    def test_leakage_tolerance(self):
        """Test leakage detection with different tolerance levels."""
        # Capture initial stats
        self.detector.capture_initial_stats(self.model)
        
        # Manually modify stats slightly
        bn_layer = None
        for module in self.model.modules():
            if isinstance(module, nn.BatchNorm2d):
                bn_layer = module
                break
        
        if bn_layer is not None:
            bn_layer.running_mean += 1e-6  # Tiny change
        
        # Test with strict tolerance
        strict_detector = BatchNormLeakageDetector(tolerance=1e-8)
        strict_detector.initial_stats = self.detector.initial_stats.copy()
        strict_report = strict_detector.detect_leakage(self.model)
        
        # Test with lenient tolerance
        lenient_detector = BatchNormLeakageDetector(tolerance=1e-4)
        lenient_detector.initial_stats = self.detector.initial_stats.copy()
        lenient_report = lenient_detector.detect_leakage(self.model)
        
        # Strict should detect, lenient should not
        assert strict_report['has_leakage']
        assert not lenient_report['has_leakage']


class TestValidationUtilities:
    """Test high-level validation utilities."""
    
    def test_validate_few_shot_model_success(self):
        """Test successful validation of few-shot model."""
        # Create properly configured model
        model = nn.Sequential(
            nn.Conv2d(3, 32, 3),
            nn.GroupNorm(4, 32),  # GroupNorm instead of BatchNorm
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(32, 5)
        )
        
        support_data = torch.randn(15, 3, 32, 32)
        query_data = torch.randn(25, 3, 32, 32)
        
        result = validate_few_shot_model(model, support_data, query_data, strict=True)
        
        assert result['passed']
        assert len(result['warnings']) == 0
        assert len(result['errors']) == 0
    
    def test_validate_few_shot_model_failure(self):
        """Test validation failure for problematic model."""
        # Create problematic model
        model = nn.Sequential(
            nn.Conv2d(3, 32, 3),
            nn.BatchNorm2d(32),  # Problematic BatchNorm
            nn.ReLU(),
            nn.Linear(32, 5)
        )
        
        # Keep in training mode
        model.train()
        
        support_data = torch.randn(15, 3, 32, 32)
        query_data = torch.randn(25, 3, 32, 32)
        
        result = validate_few_shot_model(model, support_data, query_data, strict=True)
        
        assert not result['passed']
        assert len(result['errors']) > 0
        
        error_text = ' '.join(result['errors']).lower()
        assert 'batchnorm' in error_text or 'training' in error_text
    
    def test_validate_different_strict_levels(self):
        """Test validation with different strictness levels."""
        model = nn.Sequential(
            nn.Linear(10, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Linear(32, 5)
        )
        
        support_data = torch.randn(15, 10)
        query_data = torch.randn(25, 10)
        
        # Strict validation
        result_strict = validate_few_shot_model(model, support_data, query_data, strict=True)
        
        # Lenient validation
        result_lenient = validate_few_shot_model(model, support_data, query_data, strict=False)
        
        # Strict should be more restrictive
        assert len(result_strict['errors']) >= len(result_lenient['errors'])
        assert len(result_strict['warnings']) >= len(result_lenient['warnings'])


class TestLeakagePrevention:
    """Test actual leakage prevention in realistic scenarios."""
    
    def setup_method(self):
        """Setup realistic few-shot learning scenario."""
        # ResNet-like model with BatchNorm
        self.original_model = nn.Sequential(
            nn.Conv2d(3, 64, 7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(3, stride=2, padding=1),
            
            # Residual block simulation
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(128, 5)
        )
        
        # Create different episodes
        self.support_episode1 = torch.randn(25, 3, 84, 84)  # 5-way 5-shot
        self.query_episode1 = torch.randn(75, 3, 84, 84)
        
        self.support_episode2 = torch.randn(25, 3, 84, 84) + 2.0  # Different distribution
        self.query_episode2 = torch.randn(75, 3, 84, 84) + 2.0
    
    def test_leakage_with_original_model(self):
        """Test that original model with BatchNorm causes leakage."""
        self.original_model.train()
        detector = BatchNormLeakageDetector()
        
        # Process episode 1
        detector.capture_initial_stats(self.original_model)
        with torch.no_grad():
            _ = self.original_model(self.support_episode1)
        
        # Process episode 2
        with torch.no_grad():
            _ = self.original_model(self.support_episode2)
        
        # Should detect leakage
        leakage_report = detector.detect_leakage(self.original_model)
        assert leakage_report['has_leakage']
        assert len(leakage_report['leaked_layers']) > 0
    
    def test_no_leakage_with_fixed_model(self):
        """Test that fixed model prevents leakage."""
        # Apply GroupNorm policy
        fixed_model = apply_episodic_bn_policy(self.original_model, policy="group_norm")
        
        detector = BatchNormLeakageDetector()
        
        # Process episode 1
        detector.capture_initial_stats(fixed_model)
        with torch.no_grad():
            _ = fixed_model(self.support_episode1)
        
        # Process episode 2
        with torch.no_grad():
            _ = fixed_model(self.support_episode2)
        
        # Should not detect leakage (no BatchNorm layers)
        leakage_report = detector.detect_leakage(fixed_model)
        assert not leakage_report['has_leakage']
    
    def test_episodic_mode_prevents_leakage(self):
        """Test that EpisodicMode prevents leakage."""
        detector = BatchNormLeakageDetector()
        
        # Use EpisodicMode context manager
        with EpisodicMode(self.original_model):
            # Process episode 1
            detector.capture_initial_stats(self.original_model)
            with torch.no_grad():
                _ = self.original_model(self.support_episode1)
            
            # Process episode 2
            with torch.no_grad():
                _ = self.original_model(self.support_episode2)
        
        # Should not detect significant leakage
        leakage_report = detector.detect_leakage(self.original_model)
        assert not leakage_report['has_leakage'] or len(leakage_report['leaked_layers']) < 2


class TestPerformanceComparison:
    """Test performance impact of different policies."""
    
    def setup_method(self):
        """Setup performance comparison."""
        self.model = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(32, 5)
        )
        
        self.test_data = torch.randn(32, 3, 84, 84)
    
    def test_different_policies_forward_pass(self):
        """Test that different policies produce valid outputs."""
        policies = ["group_norm", "layer_norm", "instance_norm", "freeze"]
        
        # Original output
        self.model.eval()
        with torch.no_grad():
            original_output = self.model(self.test_data)
        
        for policy in policies:
            model_policy = apply_episodic_bn_policy(self.model, policy=policy)
            model_policy.eval()
            
            with torch.no_grad():
                policy_output = model_policy(self.test_data)
            
            # Outputs should have same shape
            assert policy_output.shape == original_output.shape
            
            # Outputs should be finite
            assert torch.isfinite(policy_output).all()
    
    def test_policy_consistency(self):
        """Test that policies produce consistent outputs across runs."""
        model_fixed = apply_episodic_bn_policy(self.model, policy="group_norm")
        model_fixed.eval()
        
        # Multiple forward passes should be consistent
        with torch.no_grad():
            output1 = model_fixed(self.test_data)
            output2 = model_fixed(self.test_data)
        
        assert torch.allclose(output1, output2, atol=1e-6)


if __name__ == "__main__":
    pytest.main([__file__])