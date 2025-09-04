"""
Tests for Task Difficulty Estimation Solutions
==============================================

Tests all 5 difficulty estimation methods with proper configurations and fallbacks.
"""

import pytest
import torch
# DEMO/TEST REPRODUCIBILITY: Set seed for consistent synthetic data
torch.manual_seed(42)  # Reproducible test/demo data
import numpy as np
from unittest.mock import patch, MagicMock
from meta_learning.meta_learning_modules.utils_modules.statistical_evaluation import (
    estimate_difficulty, TaskDifficultyConfig,
    _estimate_difficulty_intra_class_variance,
    _estimate_difficulty_inter_class_separation,
    _estimate_difficulty_mdl_complexity,
    _estimate_difficulty_gradient_based,
    _estimate_difficulty_entropy
)


class TestDifficultyEstimation:
    """Test suite for task difficulty estimation methods."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.n_samples = 50
        self.n_features = 32
        self.n_classes = 5
        
        # Create test data with known difficulty characteristics
        self.easy_task_data = torch.randn(self.n_samples, self.n_features) * 0.1  # Low variance
        self.hard_task_data = torch.randn(self.n_samples, self.n_features) * 2.0  # High variance
        
        # Create labels
        self.task_labels = torch.repeat_interleave(torch.arange(self.n_classes), self.n_samples // self.n_classes)
        
    def test_intra_class_variance_method(self):
        """Test intra-class variance difficulty estimation."""
        config = TaskDifficultyConfig(
            method="intra_class_variance",
            variance_normalization=10.0
        )
        
        # Easy task should have lower difficulty
        easy_difficulty = estimate_difficulty(self.easy_task_data, "intra_class_variance", self.task_labels, config)
        
        # Hard task should have higher difficulty
        hard_difficulty = estimate_difficulty(self.hard_task_data, "intra_class_variance", self.task_labels, config)
        
        # Check difficulty is in valid range
        assert 0.0 <= easy_difficulty <= 1.0
        assert 0.0 <= hard_difficulty <= 1.0
        
        # Hard task should be more difficult
        assert hard_difficulty > easy_difficulty
        
    def test_intra_class_variance_unlabeled(self):
        """Test intra-class variance with unlabeled data."""
        config = TaskDifficultyConfig(
            method="intra_class_variance",
            assume_balanced_classes=True,
            samples_per_class_hint=10
        )
        
        difficulty = estimate_difficulty(self.easy_task_data, "intra_class_variance", None, config)
        
        assert 0.0 <= difficulty <= 1.0
        
    @patch('sklearn.discriminant_analysis.LinearDiscriminantAnalysis')
    def test_inter_class_separation_method(self, mock_lda):
        """Test inter-class separation difficulty estimation."""
        # Mock LDA
        mock_lda_instance = MagicMock()
        mock_lda_instance.fit.return_value = mock_lda_instance
        mock_lda_instance.score.return_value = 0.8  # High accuracy = low difficulty
        mock_lda.return_value = mock_lda_instance
        
        config = TaskDifficultyConfig(
            method="inter_class_separation",
            use_lda=True
        )
        
        difficulty = estimate_difficulty(self.easy_task_data, "inter_class_separation", self.task_labels, config)
        
        assert 0.0 <= difficulty <= 1.0
        assert difficulty < 0.5  # High accuracy should result in low difficulty
        
    @patch('sklearn.metrics.silhouette_score')
    def test_inter_class_separation_silhouette(self, mock_silhouette):
        """Test inter-class separation with silhouette score."""
        mock_silhouette.return_value = 0.6  # Good separation
        
        config = TaskDifficultyConfig(
            method="inter_class_separation",
            use_lda=False
        )
        
        difficulty = estimate_difficulty(self.easy_task_data, "inter_class_separation", self.task_labels, config)
        
        assert 0.0 <= difficulty <= 1.0
        mock_silhouette.assert_called_once()
        
    def test_inter_class_separation_no_labels(self):
        """Test inter-class separation requires labels."""
        config = TaskDifficultyConfig(method="inter_class_separation")
        
        with pytest.raises(ValueError, match="Inter-class separation requires labeled data"):
            estimate_difficulty(self.easy_task_data, "inter_class_separation", None, config)
            
    def test_inter_class_separation_single_class(self):
        """Test inter-class separation with single class."""
        single_class_labels = torch.zeros(self.n_samples, dtype=torch.long)
        config = TaskDifficultyConfig(method="inter_class_separation")
        
        with pytest.raises(ValueError, match="Inter-class separation requires labeled data"):
            estimate_difficulty(self.easy_task_data, "inter_class_separation", single_class_labels, config)
            
    def test_inter_class_separation_import_error(self):
        """Test handling of missing sklearn dependency."""
        config = TaskDifficultyConfig(method="inter_class_separation")
        
        with patch('sklearn.discriminant_analysis.LinearDiscriminantAnalysis', side_effect=ImportError("No sklearn")):
            with pytest.raises(ImportError, match="sklearn required for inter-class separation"):
                estimate_difficulty(self.easy_task_data, "inter_class_separation", self.task_labels, config)
                
    def test_mdl_complexity_method(self):
        """Test MDL complexity difficulty estimation."""
        config = TaskDifficultyConfig(
            method="mdl_complexity",
            compression_algorithm="zlib"
        )
        
        difficulty = estimate_difficulty(self.easy_task_data, "mdl_complexity", self.task_labels, config)
        
        assert 0.1 <= difficulty <= 0.9  # MDL clips to this range
        
    def test_mdl_different_algorithms(self):
        """Test MDL with different compression algorithms."""
        algorithms = ["zlib", "bz2", "lzma"]
        
        for algorithm in algorithms:
            config = TaskDifficultyConfig(
                method="mdl_complexity",
                compression_algorithm=algorithm
            )
            
            difficulty = estimate_difficulty(self.easy_task_data, "mdl_complexity", None, config)
            assert 0.1 <= difficulty <= 0.9
            
    def test_mdl_invalid_algorithm(self):
        """Test MDL with invalid compression algorithm."""
        config = TaskDifficultyConfig(
            method="mdl_complexity",
            compression_algorithm="invalid"
        )
        
        with pytest.raises(ValueError, match="Unsupported compression algorithm"):
            estimate_difficulty(self.easy_task_data, "mdl_complexity", None, config)
            
    def test_gradient_based_method(self):
        """Test gradient-based difficulty estimation."""
        config = TaskDifficultyConfig(
            method="gradient_based",
            gradient_steps=3,
            learning_rate=0.01,
            hidden_size=16
        )
        
        difficulty = estimate_difficulty(self.easy_task_data, "gradient_based", self.task_labels, config)
        
        assert 0.0 <= difficulty <= 1.0
        
    def test_gradient_based_no_labels(self):
        """Test gradient-based method requires labels."""
        config = TaskDifficultyConfig(method="gradient_based")
        
        with pytest.raises(ValueError, match="Gradient-based difficulty estimation requires labels"):
            estimate_difficulty(self.easy_task_data, "gradient_based", None, config)
            
    def test_entropy_method(self):
        """Test entropy-based difficulty estimation."""
        config = TaskDifficultyConfig(method="entropy")
        
        difficulty = estimate_difficulty(self.easy_task_data, "entropy", None, config)
        
        assert 0.0 <= difficulty <= 1.0
        
    def test_entropy_fallback_handling(self):
        """Test entropy method handles edge cases gracefully."""
        # Test with extreme values
        extreme_data = torch.tensor([[float('inf'), -float('inf'), 0.0] * 10]).reshape(10, 3)
        config = TaskDifficultyConfig(method="entropy")
        
        difficulty = estimate_difficulty(extreme_data, "entropy", None, config)
        
        # Should fallback to 0.5 for problematic data
        assert difficulty == 0.5
        
    def test_unknown_method(self):
        """Test error handling for unknown methods."""
        config = TaskDifficultyConfig(method="unknown_method")
        
        with pytest.raises(ValueError, match="Unknown difficulty estimation method"):
            estimate_difficulty(self.easy_task_data, "unknown_method", None, config)
            
    def test_fallback_mechanism(self):
        """Test fallback to simpler method when primary fails."""
        config = TaskDifficultyConfig(
            method="gradient_based",
            fallback_method="entropy",
            warn_on_fallback=False
        )
        
        # Force gradient-based to fail by providing no labels
        difficulty = estimate_difficulty(self.easy_task_data, "gradient_based", None, config)
        
        # Should get entropy-based result as fallback
        assert 0.0 <= difficulty <= 1.0


class TestTaskDifficultyConfig:
    """Test task difficulty configuration class."""
    
    def test_config_defaults(self):
        """Test default configuration values."""
        config = TaskDifficultyConfig()
        
        assert config.method == "intra_class_variance"
        assert config.variance_normalization == 10.0
        assert config.use_lda == True
        assert config.compression_algorithm == "zlib"
        assert config.fallback_method == "entropy"
        assert config.warn_on_fallback == True
        
    def test_config_customization(self):
        """Test custom configuration values."""
        config = TaskDifficultyConfig(
            method="gradient_based",
            gradient_steps=5,
            learning_rate=0.001,
            hidden_size=64,
            gradient_norm_scale=200.0
        )
        
        assert config.method == "gradient_based"
        assert config.gradient_steps == 5
        assert config.learning_rate == 0.001
        assert config.hidden_size == 64
        assert config.gradient_norm_scale == 200.0


class TestDifficultyEstimationIntegration:
    """Integration tests for difficulty estimation."""
    
    def test_difficulty_consistency_across_methods(self):
        """Test that difficulty estimates are consistent for the same data."""
        easy_data = torch.randn(20, 16) * 0.1
        hard_data = torch.randn(20, 16) * 3.0
        labels = torch.repeat_interleave(torch.arange(4), 5)
        
        methods_requiring_labels = ["intra_class_variance", "inter_class_separation", "gradient_based"]
        methods_no_labels = ["mdl_complexity", "entropy"]
        
        # Test methods that require labels
        for method in methods_requiring_labels:
            if method == "inter_class_separation":
                # Skip sklearn-dependent test in basic suite
                continue
                
            config = TaskDifficultyConfig(method=method)
            
            with patch('sklearn.discriminant_analysis.LinearDiscriminantAnalysis') if method == "inter_class_separation" else patch('builtins.None'):
                try:
                    easy_diff = estimate_difficulty(easy_data, method, labels, config)
                    hard_diff = estimate_difficulty(hard_data, method, labels, config)
                    
                    assert 0.0 <= easy_diff <= 1.0
                    assert 0.0 <= hard_diff <= 1.0
                    
                    if method == "intra_class_variance":
                        # For variance-based method, hard data should be more difficult
                        assert hard_diff > easy_diff
                        
                except ImportError:
                    # Skip if dependencies not available
                    continue
                    
        # Test methods that don't require labels
        for method in methods_no_labels:
            config = TaskDifficultyConfig(method=method)
            
            easy_diff = estimate_difficulty(easy_data, method, None, config)
            hard_diff = estimate_difficulty(hard_data, method, None, config)
            
            assert 0.0 <= easy_diff <= 1.0
            assert 0.0 <= hard_diff <= 1.0
            
    def test_difficulty_with_different_data_characteristics(self):
        """Test difficulty estimation with various data characteristics."""
        n_samples, n_features = 30, 20
        
        # Well-separated classes (easy)
        easy_data = torch.zeros(n_samples, n_features)
        easy_data[:10] = torch.randn(10, n_features) + 3.0  # Class 1
        easy_data[10:20] = torch.randn(10, n_features) - 3.0  # Class 2
        easy_data[20:] = torch.randn(10, n_features)  # Class 3
        easy_labels = torch.tensor([0]*10 + [1]*10 + [2]*10)
        
        # Overlapping classes (hard)
        hard_data = torch.randn(n_samples, n_features) * 0.5
        hard_labels = easy_labels  # Same labels but overlapping data
        
        config = TaskDifficultyConfig(method="intra_class_variance")
        
        easy_difficulty = estimate_difficulty(easy_data, "intra_class_variance", easy_labels, config)
        hard_difficulty = estimate_difficulty(hard_data, "intra_class_variance", hard_labels, config)
        
        # Hard task should have higher difficulty
        assert hard_difficulty > easy_difficulty
        
    def test_difficulty_estimation_edge_cases(self):
        """Test difficulty estimation with edge cases."""
        # Single sample per class
        tiny_data = torch.randn(3, 10)
        tiny_labels = torch.tensor([0, 1, 2])
        
        config = TaskDifficultyConfig(method="intra_class_variance")
        difficulty = estimate_difficulty(tiny_data, "intra_class_variance", tiny_labels, config)
        
        assert 0.0 <= difficulty <= 1.0
        
        # All same values (no variance)
        constant_data = torch.ones(10, 5)
        constant_labels = torch.tensor([0, 0, 1, 1, 2, 2, 2, 2, 2, 2])
        
        difficulty = estimate_difficulty(constant_data, "intra_class_variance", constant_labels, config)
        assert difficulty == 0.0  # No variance = easy
        
    def test_difficulty_estimation_performance(self):
        """Test performance of difficulty estimation methods."""
        # Large dataset
        large_data = torch.randn(1000, 100)
        large_labels = torch.repeat_interleave(torch.arange(10), 100)
        
        config = TaskDifficultyConfig(method="intra_class_variance")
        
        # Should complete quickly
        difficulty = estimate_difficulty(large_data, "intra_class_variance", large_labels, config)
        
        assert 0.0 <= difficulty <= 1.0
        
    def test_difficulty_estimation_numerical_stability(self):
        """Test numerical stability with extreme values."""
        # Very large values
        large_data = torch.randn(20, 10) * 1000
        large_labels = torch.repeat_interleave(torch.arange(4), 5)
        
        config = TaskDifficultyConfig(method="intra_class_variance")
        difficulty = estimate_difficulty(large_data, "intra_class_variance", large_labels, config)
        
        assert torch.isfinite(torch.tensor(difficulty))
        assert 0.0 <= difficulty <= 1.0
        
        # Very small values
        small_data = torch.randn(20, 10) * 1e-6
        difficulty = estimate_difficulty(small_data, "intra_class_variance", large_labels, config)
        
        assert torch.isfinite(torch.tensor(difficulty))
        assert 0.0 <= difficulty <= 1.0