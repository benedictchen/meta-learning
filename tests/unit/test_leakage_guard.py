"""
Comprehensive tests for leakage guard functionality.

Tests all critical leakage detection patterns that can invalidate
meta-learning results.
"""

import torch
import torch.nn as nn
import pytest
from typing import List
import warnings

from meta_learning.meta_learning_modules.leakage_guard import (
    LeakageGuard, LeakageType, LeakageViolation,
    leakage_guard_context, create_safe_normalizer
)


class TestLeakageGuard:
    """Test suite for LeakageGuard functionality."""
    
    @pytest.fixture
    def guard(self):
        """Create leakage guard for testing."""
        return LeakageGuard(strict_mode=False)  # Non-strict for testing
    
    @pytest.fixture
    def train_test_classes(self):
        """Standard train/test class split."""
        return [0, 1, 2, 3, 4], [5, 6, 7, 8, 9]
    
    def test_train_test_split_registration(self, guard, train_test_classes):
        """Test proper train/test split registration."""
        train_classes, test_classes = train_test_classes
        
        guard.register_train_test_split(train_classes, test_classes)
        
        assert guard.train_classes == set(train_classes)
        assert guard.test_classes == set(test_classes)
        assert len(guard.violations) == 0  # No violations for clean split
    
    def test_overlapping_train_test_classes(self, guard):
        """Test detection of overlapping train/test classes."""
        train_classes = [0, 1, 2, 3, 4]
        test_classes = [3, 4, 5, 6, 7]  # Overlap: 3, 4
        
        guard.register_train_test_split(train_classes, test_classes)
        
        # Should detect critical violation
        assert len(guard.violations) == 1
        violation = guard.violations[0]
        assert violation.violation_type == LeakageType.CLASS_STATISTICS
        assert violation.severity == "critical"
        assert "overlap" in violation.description.lower()
    
    def test_normalization_stats_validation_clean(self, guard, train_test_classes):
        """Test validation of clean normalization statistics."""
        train_classes, test_classes = train_test_classes
        guard.register_train_test_split(train_classes, test_classes)
        
        # Create stats from train data only
        train_data = torch.randn(100, 64)
        stats = {'mean': train_data.mean(0), 'std': train_data.std(0)}
        
        valid = guard.validate_normalization_stats(
            stats, train_classes, "train_normalization"
        )
        
        assert valid == True
        assert len(guard.violations) == 0
        assert "train_normalization" in guard.registered_stats
    
    def test_normalization_stats_validation_leakage(self, guard, train_test_classes):
        """Test detection of normalization statistics leakage."""
        train_classes, test_classes = train_test_classes
        guard.register_train_test_split(train_classes, test_classes)
        
        # Create stats from mixed train+test data (LEAKAGE)
        mixed_data = torch.randn(200, 64)
        stats = {'mean': mixed_data.mean(0), 'std': mixed_data.std(0)}
        
        valid = guard.validate_normalization_stats(
            stats, train_classes + test_classes, "mixed_normalization"
        )
        
        assert valid == False
        assert len(guard.violations) == 1
        violation = guard.violations[0]
        assert violation.violation_type == LeakageType.NORMALIZATION_STATS
        assert violation.severity == "critical"
        assert "both train and test classes" in violation.description
    
    def test_episode_isolation_validation_clean(self, guard, train_test_classes):
        """Test validation of properly isolated episodes."""
        train_classes, test_classes = train_test_classes
        guard.register_train_test_split(train_classes, test_classes)
        
        # Episode from train classes only
        valid = guard.validate_episode_isolation([0, 1, 2], [0, 1, 2], "train_episode")
        assert valid == True
        
        # Episode from test classes only
        valid = guard.validate_episode_isolation([5, 6, 7], [5, 6, 7], "test_episode")
        assert valid == True
        
        assert len(guard.violations) == 0
    
    def test_episode_isolation_validation_cross_split(self, guard, train_test_classes):
        """Test detection of episodes spanning train/test split."""
        train_classes, test_classes = train_test_classes
        guard.register_train_test_split(train_classes, test_classes)
        
        # Episode spanning train+test (LEAKAGE)
        valid = guard.validate_episode_isolation(
            [0, 1, 5], [0, 1, 5], "cross_split_episode"
        )
        
        assert valid == False
        assert len(guard.violations) == 1
        violation = guard.violations[0]
        assert violation.violation_type == LeakageType.CLASS_STATISTICS
        assert violation.severity == "critical"
        assert "spans train/test split" in violation.description
    
    def test_episode_isolation_support_query_mismatch(self, guard):
        """Test detection of support/query class mismatch."""
        # Different classes in support vs query
        valid = guard.validate_episode_isolation(
            [0, 1, 2], [0, 1, 3], "mismatched_episode"
        )
        
        assert valid == False
        assert len(guard.violations) == 1
        violation = guard.violations[0]
        assert "different classes" in violation.description
        assert violation.severity == "medium"
    
    def test_batchnorm_leakage_detection(self, guard):
        """Test detection of BatchNorm statistics leakage."""
        # Create model with BatchNorm
        model = nn.Sequential(
            nn.Conv2d(3, 32, 3),
            nn.BatchNorm2d(32),  # This will be problematic
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(32, 5)
        )
        
        # Set problematic state: training=True, track_running_stats=True
        model.train()
        for module in model.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.track_running_stats = True
        
        violations = guard.check_batchnorm_leakage(model, episode_boundary=True)
        
        assert len(violations) > 0
        violation = violations[0]
        assert violation.violation_type == LeakageType.BATCHNORM_RUNNING_STATS
        assert violation.severity == "high"
        assert "track_running_stats=True" in violation.description
    
    def test_optimizer_state_monitoring(self, guard):
        """Test monitoring of optimizer state for leakage."""
        # Create optimizer with some accumulated state
        param = nn.Parameter(torch.randn(10, 5))
        optimizer = torch.optim.Adam([param])
        
        # Simulate some optimization to build state
        loss = (param ** 2).sum()
        loss.backward()
        optimizer.step()
        
        violations = guard.monitor_optimizer_state(optimizer, episode_boundary=True)
        
        assert len(violations) > 0
        violation = violations[0]
        assert violation.violation_type == LeakageType.OPTIMIZER_MOMENTS
        assert "accumulated moments" in violation.description
        assert violation.severity == "medium"
    
    def test_strict_mode_raises_on_critical(self):
        """Test that strict mode raises on critical violations."""
        strict_guard = LeakageGuard(strict_mode=True)
        
        # This should raise due to critical violation
        with pytest.raises(ValueError, match="CRITICAL LEAKAGE DETECTED"):
            strict_guard.register_train_test_split([0, 1, 2], [2, 3, 4])  # Overlap
    
    def test_assert_no_leakage(self, guard):
        """Test assert_no_leakage functionality."""
        # Add some violations of different severities
        guard.violations = [
            LeakageViolation(LeakageType.NORMALIZATION_STATS, "test", "Low severity", "low", "fix"),
            LeakageViolation(LeakageType.CLASS_STATISTICS, "test", "High severity", "high", "fix")
        ]
        
        # Should pass for max_severity="high" 
        guard.assert_no_leakage(max_severity="high")
        
        # Should fail for max_severity="medium"
        with pytest.raises(AssertionError, match="Leakage violations detected"):
            guard.assert_no_leakage(max_severity="medium")
    
    def test_leakage_report(self, guard, train_test_classes):
        """Test comprehensive leakage report generation."""
        train_classes, test_classes = train_test_classes
        guard.register_train_test_split(train_classes, test_classes)
        
        # Add some test data and violations
        guard.validate_normalization_stats(
            {'mean': torch.zeros(5)}, train_classes, "test_stats"
        )
        guard.violations.append(
            LeakageViolation(LeakageType.BATCHNORM_RUNNING_STATS, "test", "Test", "medium", "fix")
        )
        
        report = guard.get_leakage_report()
        
        assert report['total_violations'] == 1
        assert 'batchnorm_running_stats' in report['violations_by_type']
        assert report['severity_counts']['medium'] == 1
        assert report['train_classes'] == train_classes
        assert report['test_classes'] == test_classes
        assert 'test_stats' in report['registered_stats']


class TestSafeNormalizer:
    """Test safe normalizer functionality."""
    
    def test_safe_normalizer_creation(self):
        """Test creation of safe normalizer."""
        train_data = torch.randn(100, 64)
        train_classes = [0, 1, 2, 3, 4]
        
        normalizer = create_safe_normalizer(train_data, train_classes)
        
        # Test that it's callable
        test_data = torch.randn(50, 64)
        normalized = normalizer(test_data)
        
        assert normalized.shape == test_data.shape
        assert not torch.allclose(normalized, test_data)  # Should be different
    
    def test_safe_normalizer_with_guard(self):
        """Test safe normalizer with leakage guard monitoring."""
        guard = LeakageGuard()
        train_data = torch.randn(100, 64)
        train_classes = [0, 1, 2, 3, 4]
        
        normalizer = create_safe_normalizer(train_data, train_classes, guard)
        
        # Should have registered the normalization stats
        assert len(guard.registered_stats) == 1
        assert "safe_normalizer" in guard.registered_stats
    
    def test_safe_normalizer_numerical_stability(self):
        """Test numerical stability of safe normalizer."""
        # Create data with some zero-variance features
        train_data = torch.randn(100, 64)
        train_data[:, 0] = 1.0  # Zero variance feature
        train_data[:, 1] = 0.0  # Another zero variance feature
        
        normalizer = create_safe_normalizer(train_data, [0, 1, 2])
        
        # Should handle zero variance gracefully
        test_data = torch.randn(50, 64)
        normalized = normalizer(test_data)
        
        assert torch.isfinite(normalized).all()
        assert not torch.isnan(normalized).any()


class TestLeakageGuardContext:
    """Test leakage guard context manager."""
    
    def test_context_manager_basic(self):
        """Test basic context manager functionality."""
        guard = LeakageGuard()
        train_classes = [0, 1, 2]
        test_classes = [3, 4, 5]
        
        with leakage_guard_context(guard, train_classes, test_classes) as ctx_guard:
            assert ctx_guard is guard
            assert guard.train_classes == set(train_classes)
            assert guard.test_classes == set(test_classes)
    
    def test_context_manager_with_violations(self, caplog):
        """Test context manager with violations."""
        guard = LeakageGuard(strict_mode=False)
        
        with leakage_guard_context(guard, [0, 1, 2], [2, 3, 4]):  # Overlap
            pass
            
        # Should log violations on exit
        assert "violations" in caplog.text.lower()
    
    def test_context_manager_without_classes(self):
        """Test context manager without specifying classes."""
        guard = LeakageGuard()
        
        with leakage_guard_context(guard) as ctx_guard:
            assert ctx_guard is guard
            assert guard.train_classes is None
            assert guard.test_classes is None


class TestIntegrationScenarios:
    """Integration tests for realistic leakage scenarios."""
    
    def test_complete_few_shot_scenario_clean(self):
        """Test clean few-shot scenario with no leakage."""
        guard = LeakageGuard(strict_mode=False)
        
        # Setup clean train/test split
        train_classes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        test_classes = [10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
        
        with leakage_guard_context(guard, train_classes, test_classes):
            # 1. Create normalizer from train data only
            train_data = torch.randn(1000, 128)
            normalizer = create_safe_normalizer(train_data, train_classes, guard)
            
            # 2. Validate clean episode
            guard.validate_episode_isolation([10, 11, 12], [10, 11, 12], "test_episode")
            
            # 3. Check clean model
            model = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 3))
            model.eval()  # No BatchNorm issues
            guard.check_batchnorm_leakage(model)
        
        # Should be completely clean
        assert len(guard.violations) == 0
    
    def test_complete_few_shot_scenario_with_leakage(self):
        """Test few-shot scenario with multiple types of leakage."""
        guard = LeakageGuard(strict_mode=False)
        
        train_classes = [0, 1, 2, 3, 4]
        test_classes = [3, 4, 5, 6, 7]  # Overlap: leakage #1
        
        with leakage_guard_context(guard, train_classes, test_classes):
            # Leakage #2: Normalization from mixed data
            mixed_data = torch.randn(1000, 128)
            stats = {'mean': mixed_data.mean(0), 'std': mixed_data.std(0)}
            guard.validate_normalization_stats(
                stats, train_classes + test_classes, "mixed_norm"
            )
            
            # Leakage #3: Episode spanning splits
            guard.validate_episode_isolation([0, 1, 5], [0, 1, 5], "cross_episode")
            
            # Leakage #4: BatchNorm in training mode
            model = nn.Sequential(
                nn.Linear(128, 64),
                nn.BatchNorm1d(64),
                nn.ReLU(), 
                nn.Linear(64, 3)
            )
            model.train()  # Training mode with track_running_stats
            guard.check_batchnorm_leakage(model, episode_boundary=True)
        
        # Should detect multiple violations
        assert len(guard.violations) >= 3
        
        # Check we have different types of violations
        violation_types = {v.violation_type for v in guard.violations}
        assert LeakageType.CLASS_STATISTICS in violation_types
        assert LeakageType.NORMALIZATION_STATS in violation_types
        assert LeakageType.BATCHNORM_RUNNING_STATS in violation_types


if __name__ == "__main__":
    pytest.main([__file__, "-v"])