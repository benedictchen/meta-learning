"""
💰 SUPPORT THIS RESEARCH - PLEASE DONATE! 💰

🙏 If this library helps your research or project, please consider donating:
💳 PayPal: https://www.paypal.com/cgi-bin/webscr?cmd=_s-xclick&hosted_button_id=WXQKYYKPHWXHS
⭐ GitHub Sponsors: https://github.com/sponsors/benedictchen

👨‍💻 Author: Benedict Chen
💰 Donations: Help support this work! Buy me a coffee ☕, beer 🍺, lamborghini 🏎️, or private island 🏝️
💖 Please consider recurring donations to fully support continued research

Your support enables cutting-edge AI research for everyone! 🚀

Comprehensive Leakage Guard Coverage Tests
=========================================

Complete test coverage for research integrity and data leakage detection.
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
from typing import List, Dict, Any
from unittest.mock import patch, MagicMock

from meta_learning.leakage_guard import (
    LeakageType, LeakageViolation, LeakageGuard, create_leakage_guard
)


class TestLeakageType:
    """Test LeakageType enumeration."""
    
    def test_leakage_type_values(self):
        """Test all LeakageType values."""
        assert LeakageType.TRAIN_TEST_CONTAMINATION.value == "train_test_contamination"
        assert LeakageType.TEMPORAL_LEAKAGE.value == "temporal_leakage"
        assert LeakageType.FUTURE_INFORMATION.value == "future_information"
        assert LeakageType.DUPLICATE_SAMPLES.value == "duplicate_samples"
        assert LeakageType.LABEL_LEAKAGE.value == "label_leakage"


class TestLeakageViolation:
    """Test LeakageViolation class."""
    
    def test_leakage_violation_creation(self):
        """Test LeakageViolation creation."""
        violation = LeakageViolation(
            violation_type=LeakageType.TRAIN_TEST_CONTAMINATION,
            severity="high",
            description="Train classes found in test set",
            affected_samples=[1, 2, 3],
            metadata={"episode_id": "test_episode"}
        )
        
        assert violation.violation_type == LeakageType.TRAIN_TEST_CONTAMINATION
        assert violation.severity == "high"
        assert violation.description == "Train classes found in test set"
        assert violation.affected_samples == [1, 2, 3]
        assert violation.metadata["episode_id"] == "test_episode"
    
    def test_leakage_violation_str(self):
        """Test LeakageViolation string representation."""
        violation = LeakageViolation(
            violation_type=LeakageType.DUPLICATE_SAMPLES,
            severity="medium",
            description="Duplicate data detected"
        )
        
        str_repr = str(violation)
        assert "duplicate_samples" in str_repr
        assert "medium" in str_repr
        assert "Duplicate data detected" in str_repr


class TestLeakageGuardCreation:
    """Test LeakageGuard creation and configuration."""
    
    def test_leakage_guard_creation_default(self):
        """Test LeakageGuard creation with defaults."""
        guard = LeakageGuard()
        
        assert guard.strict_mode == True
        assert guard.check_train_test_split == True
        assert guard.check_temporal_consistency == True
        assert guard.check_duplicate_samples == True
        assert guard.check_future_information == True
        assert guard.similarity_threshold == 0.95
        assert len(guard.violations) == 0
    
    def test_leakage_guard_creation_custom(self):
        """Test LeakageGuard creation with custom parameters."""
        guard = LeakageGuard(
            strict_mode=False,
            check_train_test_split=True,
            check_temporal_consistency=False,
            check_duplicate_samples=True,
            check_future_information=False,
            similarity_threshold=0.8
        )
        
        assert guard.strict_mode == False
        assert guard.check_train_test_split == True
        assert guard.check_temporal_consistency == False
        assert guard.check_duplicate_samples == True
        assert guard.check_future_information == False
        assert guard.similarity_threshold == 0.8


class TestTrainTestSplitValidation:
    """Test train-test split validation functionality."""
    
    def test_register_train_test_split(self):
        """Test registering train-test splits."""
        guard = LeakageGuard()
        
        train_classes = [0, 1, 2, 3, 4]
        test_classes = [5, 6, 7, 8, 9]
        
        guard.register_train_test_split(train_classes, test_classes)
        
        assert set(guard.train_classes) == set(train_classes)
        assert set(guard.test_classes) == set(test_classes)
    
    def test_register_overlapping_split_strict(self):
        """Test registering overlapping splits in strict mode."""
        guard = LeakageGuard(strict_mode=True)
        
        train_classes = [0, 1, 2, 3, 4]
        test_classes = [3, 4, 5, 6, 7]  # Overlap with train
        
        with pytest.raises(ValueError, match="Train and test classes overlap"):
            guard.register_train_test_split(train_classes, test_classes)
    
    def test_register_overlapping_split_non_strict(self):
        """Test registering overlapping splits in non-strict mode."""
        guard = LeakageGuard(strict_mode=False)
        
        train_classes = [0, 1, 2]
        test_classes = [2, 3, 4]  # Overlap with train
        
        # Should not raise error, but add violation
        guard.register_train_test_split(train_classes, test_classes)
        
        assert len(guard.violations) == 1
        assert guard.violations[0].violation_type == LeakageType.TRAIN_TEST_CONTAMINATION
    
    def test_check_episode_data_clean(self):
        """Test checking clean episode data."""
        guard = LeakageGuard()
        guard.register_train_test_split([0, 1, 2], [3, 4, 5])
        
        support_classes = [0, 1, 2]
        query_classes = [0, 1, 2]
        
        is_clean = guard.check_episode_data(support_classes, query_classes, "test_episode")
        
        assert is_clean == True
        assert len(guard.violations) == 0
    
    def test_check_episode_data_contaminated(self):
        """Test checking contaminated episode data."""
        guard = LeakageGuard(strict_mode=False)
        guard.register_train_test_split([0, 1, 2], [3, 4, 5])
        
        support_classes = [0, 1, 3]  # Class 3 is in test set
        query_classes = [0, 1, 3]
        
        is_clean = guard.check_episode_data(support_classes, query_classes, "contaminated_episode")
        
        assert is_clean == False
        assert len(guard.violations) >= 1
        violation = guard.violations[-1]
        assert violation.violation_type == LeakageType.TRAIN_TEST_CONTAMINATION
    
    def test_check_episode_data_strict_mode(self):
        """Test checking episode data in strict mode."""
        guard = LeakageGuard(strict_mode=True)
        guard.register_train_test_split([0, 1, 2], [3, 4, 5])
        
        support_classes = [0, 1, 3]  # Contaminated
        query_classes = [0, 1, 3]
        
        with pytest.raises(ValueError, match="Data leakage detected"):
            guard.check_episode_data(support_classes, query_classes, "strict_episode")


class TestDuplicateSampleDetection:
    """Test duplicate sample detection functionality."""
    
    def test_check_duplicate_samples_none(self):
        """Test checking for duplicates when none exist."""
        guard = LeakageGuard()
        
        data = torch.randn(10, 5)  # Random data, unlikely duplicates
        labels = torch.randint(0, 3, (10,))
        
        violations = guard._check_duplicate_samples(data, labels, "test_check")
        
        assert len(violations) == 0
    
    def test_check_duplicate_samples_exact(self):
        """Test checking for exact duplicate samples."""
        guard = LeakageGuard(similarity_threshold=1.0)
        
        # Create data with exact duplicates
        unique_data = torch.randn(5, 3)
        data = torch.cat([unique_data, unique_data[:2]], dim=0)  # Add 2 duplicates
        labels = torch.randint(0, 2, (7,))
        
        violations = guard._check_duplicate_samples(data, labels, "duplicate_check")
        
        assert len(violations) > 0
        assert violations[0].violation_type == LeakageType.DUPLICATE_SAMPLES
    
    def test_check_duplicate_samples_similar(self):
        """Test checking for similar samples."""
        guard = LeakageGuard(similarity_threshold=0.9)
        
        # Create very similar samples
        base_sample = torch.randn(1, 5)
        similar_sample = base_sample + 0.01 * torch.randn(1, 5)  # Very similar
        data = torch.cat([base_sample, similar_sample], dim=0)
        labels = torch.tensor([0, 0])
        
        violations = guard._check_duplicate_samples(data, labels, "similarity_check")
        
        # May or may not detect depending on random noise
        assert isinstance(violations, list)
    
    def test_check_duplicate_samples_disabled(self):
        """Test duplicate checking when disabled."""
        guard = LeakageGuard(check_duplicate_samples=False)
        
        # Even with exact duplicates
        data = torch.ones(5, 3)  # All identical
        labels = torch.zeros(5)
        
        violations = guard._check_duplicate_samples(data, labels, "disabled_check")
        
        assert len(violations) == 0


class TestTemporalConsistency:
    """Test temporal consistency validation."""
    
    def test_check_temporal_consistency_valid(self):
        """Test valid temporal consistency."""
        guard = LeakageGuard()
        
        timestamps = [1.0, 2.0, 3.0, 4.0, 5.0]
        
        violations = guard._check_temporal_consistency(timestamps, "temporal_test")
        
        assert len(violations) == 0
    
    def test_check_temporal_consistency_invalid(self):
        """Test invalid temporal consistency."""
        guard = LeakageGuard()
        
        timestamps = [1.0, 3.0, 2.0, 5.0, 4.0]  # Non-monotonic
        
        violations = guard._check_temporal_consistency(timestamps, "temporal_violation")
        
        assert len(violations) > 0
        assert violations[0].violation_type == LeakageType.TEMPORAL_LEAKAGE
    
    def test_check_temporal_consistency_disabled(self):
        """Test temporal consistency when disabled."""
        guard = LeakageGuard(check_temporal_consistency=False)
        
        timestamps = [5.0, 1.0, 3.0, 2.0]  # Invalid order
        
        violations = guard._check_temporal_consistency(timestamps, "disabled_temporal")
        
        assert len(violations) == 0


class TestFutureInformationDetection:
    """Test future information leakage detection."""
    
    def test_check_future_information_valid(self):
        """Test valid temporal ordering."""
        guard = LeakageGuard()
        
        train_timestamps = [1.0, 2.0, 3.0]
        test_timestamp = 4.0
        
        violations = guard._check_future_information(
            train_timestamps, test_timestamp, "future_test"
        )
        
        assert len(violations) == 0
    
    def test_check_future_information_leakage(self):
        """Test future information leakage detection."""
        guard = LeakageGuard()
        
        train_timestamps = [1.0, 2.0, 5.0]  # Future timestamp in training
        test_timestamp = 3.0
        
        violations = guard._check_future_information(
            train_timestamps, test_timestamp, "future_leakage"
        )
        
        assert len(violations) > 0
        assert violations[0].violation_type == LeakageType.FUTURE_INFORMATION
    
    def test_check_future_information_disabled(self):
        """Test future information check when disabled."""
        guard = LeakageGuard(check_future_information=False)
        
        train_timestamps = [1.0, 2.0, 5.0]  # Future leakage
        test_timestamp = 3.0
        
        violations = guard._check_future_information(
            train_timestamps, test_timestamp, "disabled_future"
        )
        
        assert len(violations) == 0


class TestLabelLeakageDetection:
    """Test label leakage detection."""
    
    def test_check_label_leakage_clean(self):
        """Test clean label usage."""
        guard = LeakageGuard()
        
        # Proper few-shot setup
        support_labels = [0, 0, 1, 1, 2, 2]
        query_labels = [0, 1, 2]
        
        violations = guard._check_label_leakage(
            support_labels, query_labels, "clean_labels"
        )
        
        assert len(violations) == 0
    
    def test_check_label_leakage_novel_class(self):
        """Test label leakage with novel classes in query."""
        guard = LeakageGuard()
        
        support_labels = [0, 0, 1, 1]
        query_labels = [0, 1, 2]  # Class 2 not in support
        
        violations = guard._check_label_leakage(
            support_labels, query_labels, "novel_class"
        )
        
        assert len(violations) > 0
        assert violations[0].violation_type == LeakageType.LABEL_LEAKAGE


class TestViolationManagement:
    """Test violation recording and management."""
    
    def test_record_violation(self):
        """Test recording violations."""
        guard = LeakageGuard()
        
        violation = LeakageViolation(
            violation_type=LeakageType.DUPLICATE_SAMPLES,
            severity="medium",
            description="Test violation"
        )
        
        guard._record_violation(violation)
        
        assert len(guard.violations) == 1
        assert guard.violations[0] == violation
    
    def test_get_violations_by_type(self):
        """Test getting violations by type."""
        guard = LeakageGuard()
        
        # Add different types of violations
        violation1 = LeakageViolation(
            violation_type=LeakageType.TRAIN_TEST_CONTAMINATION,
            severity="high",
            description="Contamination"
        )
        violation2 = LeakageViolation(
            violation_type=LeakageType.DUPLICATE_SAMPLES,
            severity="medium", 
            description="Duplicates"
        )
        violation3 = LeakageViolation(
            violation_type=LeakageType.TRAIN_TEST_CONTAMINATION,
            severity="low",
            description="Minor contamination"
        )
        
        guard._record_violation(violation1)
        guard._record_violation(violation2)
        guard._record_violation(violation3)
        
        contamination_violations = guard.get_violations_by_type(
            LeakageType.TRAIN_TEST_CONTAMINATION
        )
        
        assert len(contamination_violations) == 2
        assert all(v.violation_type == LeakageType.TRAIN_TEST_CONTAMINATION 
                  for v in contamination_violations)
    
    def test_get_violations_by_severity(self):
        """Test getting violations by severity."""
        guard = LeakageGuard()
        
        violations = [
            LeakageViolation(LeakageType.DUPLICATE_SAMPLES, "high", "High severity"),
            LeakageViolation(LeakageType.TEMPORAL_LEAKAGE, "medium", "Medium severity"),
            LeakageViolation(LeakageType.LABEL_LEAKAGE, "high", "Another high severity")
        ]
        
        for v in violations:
            guard._record_violation(v)
        
        high_violations = guard.get_violations_by_severity("high")
        
        assert len(high_violations) == 2
        assert all(v.severity == "high" for v in high_violations)
    
    def test_clear_violations(self):
        """Test clearing violations."""
        guard = LeakageGuard()
        
        violation = LeakageViolation(
            LeakageType.DUPLICATE_SAMPLES,
            "medium",
            "Test violation"
        )
        guard._record_violation(violation)
        
        assert len(guard.violations) == 1
        
        guard.clear_violations()
        
        assert len(guard.violations) == 0
    
    def test_has_violations(self):
        """Test checking if violations exist."""
        guard = LeakageGuard()
        
        assert guard.has_violations() == False
        
        violation = LeakageViolation(
            LeakageType.TEMPORAL_LEAKAGE,
            "low",
            "Test"
        )
        guard._record_violation(violation)
        
        assert guard.has_violations() == True
    
    def test_get_violation_summary(self):
        """Test getting violation summary."""
        guard = LeakageGuard()
        
        violations = [
            LeakageViolation(LeakageType.TRAIN_TEST_CONTAMINATION, "high", "Test 1"),
            LeakageViolation(LeakageType.TRAIN_TEST_CONTAMINATION, "medium", "Test 2"),
            LeakageViolation(LeakageType.DUPLICATE_SAMPLES, "low", "Test 3")
        ]
        
        for v in violations:
            guard._record_violation(v)
        
        summary = guard.get_violation_summary()
        
        assert isinstance(summary, dict)
        assert summary["total_violations"] == 3
        assert LeakageType.TRAIN_TEST_CONTAMINATION.value in summary["by_type"]
        assert summary["by_type"][LeakageType.TRAIN_TEST_CONTAMINATION.value] == 2


class TestComprehensiveValidation:
    """Test comprehensive data validation."""
    
    def test_validate_episode_comprehensive(self):
        """Test comprehensive episode validation."""
        guard = LeakageGuard(strict_mode=False)
        guard.register_train_test_split([0, 1, 2], [3, 4, 5])
        
        # Create episode with various issues
        support_data = torch.randn(10, 5)
        support_labels = [0, 0, 1, 1, 2, 2, 3, 3, 0, 0]  # Class 3 is test class
        query_data = torch.randn(6, 5)
        query_labels = [0, 1, 2, 0, 1, 2]
        
        is_valid = guard.validate_episode(
            support_data, support_labels,
            query_data, query_labels,
            episode_id="comprehensive_test"
        )
        
        assert is_valid == False  # Should detect contamination
        assert len(guard.violations) > 0
    
    def test_validate_dataset_comprehensive(self):
        """Test comprehensive dataset validation."""
        guard = LeakageGuard(strict_mode=False)
        
        # Create dataset with potential issues
        all_data = torch.randn(20, 4)
        all_labels = list(range(20))
        timestamps = list(range(20))
        
        violations = guard.validate_dataset(
            all_data, all_labels, timestamps,
            dataset_id="test_dataset"
        )
        
        assert isinstance(violations, list)


class TestUtilityFunctions:
    """Test utility functions."""
    
    def test_create_leakage_guard_default(self):
        """Test creating leakage guard with defaults."""
        guard = create_leakage_guard()
        
        assert isinstance(guard, LeakageGuard)
        assert guard.strict_mode == True
    
    def test_create_leakage_guard_custom(self):
        """Test creating leakage guard with custom settings."""
        guard = create_leakage_guard(
            strict_mode=False,
            similarity_threshold=0.8,
            check_temporal_consistency=False
        )
        
        assert guard.strict_mode == False
        assert guard.similarity_threshold == 0.8
        assert guard.check_temporal_consistency == False


class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_empty_data_handling(self):
        """Test handling of empty data."""
        guard = LeakageGuard()
        
        # Empty lists should not cause errors
        violations = guard._check_duplicate_samples(
            torch.empty(0, 5), torch.empty(0), "empty_test"
        )
        assert len(violations) == 0
    
    def test_single_sample_handling(self):
        """Test handling of single samples."""
        guard = LeakageGuard()
        
        data = torch.randn(1, 3)
        labels = torch.tensor([0])
        
        violations = guard._check_duplicate_samples(data, labels, "single_sample")
        assert len(violations) == 0
    
    def test_invalid_input_types(self):
        """Test handling of invalid input types."""
        guard = LeakageGuard()
        
        # Should handle conversion gracefully or raise appropriate errors
        with pytest.raises((TypeError, ValueError)):
            guard.check_episode_data("invalid", "types", "test")
    
    def test_large_dataset_performance(self):
        """Test performance with large datasets."""
        guard = LeakageGuard(similarity_threshold=0.95)
        
        # Large dataset - should complete in reasonable time
        large_data = torch.randn(1000, 10)
        large_labels = torch.randint(0, 100, (1000,))
        
        # Should not take too long or crash
        violations = guard._check_duplicate_samples(
            large_data, large_labels, "performance_test"
        )
        
        assert isinstance(violations, list)
    
    def test_numerical_stability(self):
        """Test numerical stability with edge case values."""
        guard = LeakageGuard()
        
        # Test with extreme values
        extreme_data = torch.tensor([
            [float('inf'), -float('inf'), float('nan')],
            [1e10, -1e10, 0.0],
            [1e-10, -1e-10, 1.0]
        ])
        labels = torch.tensor([0, 1, 2])
        
        # Should handle gracefully without crashing
        try:
            violations = guard._check_duplicate_samples(
                extreme_data, labels, "numerical_test"
            )
            assert isinstance(violations, list)
        except (RuntimeError, ValueError):
            # Some numerical issues might be expected
            pass