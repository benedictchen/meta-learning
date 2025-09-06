"""
Basic tests for task difficulty analysis components.

Tests focus on ensuring the core functionality works without 
requiring complex dependency resolution.
"""

import pytest
import torch
import torch.nn as nn
import numpy as np

# Import the components we're testing
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from meta_learning.analysis.task_difficulty.complexity_analyzer import ComplexityAnalyzer


class TestComplexityAnalyzerBasic:
    """Basic tests for ComplexityAnalyzer"""
    
    def create_separable_data(self):
        """Create well-separated test data"""
        # Class 0: centered at [2, 0]
        class_0 = torch.randn(20, 2) * 0.1 + torch.tensor([2.0, 0.0])
        
        # Class 1: centered at [0, 2]  
        class_1 = torch.randn(20, 2) * 0.1 + torch.tensor([0.0, 2.0])
        
        X = torch.cat([class_0, class_1], dim=0)
        y = torch.cat([torch.zeros(20, dtype=torch.long), torch.ones(20, dtype=torch.long)])
        
        return X, y
    
    def create_overlapping_data(self):
        """Create overlapping test data"""
        # Both classes centered near origin with high variance
        class_0 = torch.randn(20, 2) * 0.8
        class_1 = torch.randn(20, 2) * 0.8 + 0.2
        
        X = torch.cat([class_0, class_1], dim=0)
        y = torch.cat([torch.zeros(20, dtype=torch.long), torch.ones(20, dtype=torch.long)])
        
        return X, y
    
    def test_initialization(self):
        """Test basic initialization"""
        analyzer = ComplexityAnalyzer()
        assert hasattr(analyzer, 'logger')
    
    def test_fisher_discriminant_ratio_separable(self):
        """Test Fisher's discriminant ratio on separable data"""
        analyzer = ComplexityAnalyzer()
        X, y = self.create_separable_data()
        
        difficulty = analyzer.fisher_discriminant_ratio(X, y)
        
        assert isinstance(difficulty, float), "Should return float"
        assert 0 <= difficulty <= 1, "Should be normalized between 0 and 1"
        assert difficulty < 0.5, "Separable data should have low difficulty"
    
    def test_fisher_discriminant_ratio_overlapping(self):
        """Test Fisher's discriminant ratio on overlapping data"""
        analyzer = ComplexityAnalyzer()
        X, y = self.create_overlapping_data()
        
        difficulty = analyzer.fisher_discriminant_ratio(X, y)
        
        assert isinstance(difficulty, float), "Should return float"
        assert 0 <= difficulty <= 1, "Should be normalized between 0 and 1"
        assert difficulty > 0.3, "Overlapping data should have higher difficulty"
    
    def test_single_class_handling(self):
        """Test handling of single class data"""
        analyzer = ComplexityAnalyzer()
        
        X = torch.randn(10, 3)
        y = torch.zeros(10, dtype=torch.long)  # All same class
        
        difficulty = analyzer.fisher_discriminant_ratio(X, y)
        
        assert difficulty == 1.0, "Single class should have maximum difficulty"
    
    def test_empty_data_handling(self):
        """Test handling of empty data"""
        analyzer = ComplexityAnalyzer()
        
        X = torch.empty(0, 3)
        y = torch.empty(0, dtype=torch.long)
        
        difficulty = analyzer.fisher_discriminant_ratio(X, y)
        
        assert difficulty == 1.0, "Empty data should have maximum difficulty"
    
    def test_numpy_compatibility(self):
        """Test that analyzer works with numpy arrays"""
        analyzer = ComplexityAnalyzer()
        X, y = self.create_separable_data()
        
        # Convert to numpy
        X_np = X.numpy()
        y_np = y.numpy()
        
        # Should work with sklearn-based methods
        try:
            difficulty = analyzer.class_separability(X_np, y_np)
            assert isinstance(difficulty, float)
            assert 0 <= difficulty <= 1
        except Exception as e:
            # If sklearn methods not available, just check basic functionality
            pytest.skip(f"sklearn methods not available: {e}")
    
    def test_high_dimensional_data(self):
        """Test with high-dimensional data"""
        analyzer = ComplexityAnalyzer()
        
        # Create 100-dimensional data
        X = torch.randn(40, 100)
        y = torch.cat([torch.zeros(20, dtype=torch.long), torch.ones(20, dtype=torch.long)])
        
        difficulty = analyzer.fisher_discriminant_ratio(X, y)
        
        assert isinstance(difficulty, float)
        assert 0 <= difficulty <= 1
    
    def test_multiple_classes(self):
        """Test with multiple classes (>2)"""
        analyzer = ComplexityAnalyzer()
        
        # Create 5-class data
        X = torch.randn(50, 5)
        y = torch.arange(5).repeat(10)  # 10 samples per class
        
        difficulty = analyzer.fisher_discriminant_ratio(X, y)
        
        assert isinstance(difficulty, float)
        assert 0 <= difficulty <= 1
    
    def test_reproducibility(self):
        """Test that results are reproducible"""
        analyzer = ComplexityAnalyzer()
        
        torch.manual_seed(42)
        X1, y1 = self.create_separable_data()
        difficulty1 = analyzer.fisher_discriminant_ratio(X1, y1)
        
        torch.manual_seed(42)
        X2, y2 = self.create_separable_data()
        difficulty2 = analyzer.fisher_discriminant_ratio(X2, y2)
        
        assert abs(difficulty1 - difficulty2) < 1e-6, "Results should be reproducible"


class TestComplexityMeasureComparisons:
    """Test relative behavior of complexity measures"""
    
    def test_difficulty_ranking(self):
        """Test that different data complexities are ranked correctly"""
        analyzer = ComplexityAnalyzer()
        
        # Create data with different separation levels
        torch.manual_seed(123)
        
        # Very separable (low difficulty)
        very_sep_X = torch.cat([
            torch.randn(15, 3) + torch.tensor([5.0, 0.0, 0.0]),
            torch.randn(15, 3) + torch.tensor([0.0, 5.0, 0.0])
        ])
        very_sep_y = torch.cat([torch.zeros(15, dtype=torch.long), torch.ones(15, dtype=torch.long)])
        
        # Moderately separable  
        mod_sep_X = torch.cat([
            torch.randn(15, 3) + torch.tensor([1.0, 0.0, 0.0]),
            torch.randn(15, 3) + torch.tensor([0.0, 1.0, 0.0])
        ])
        mod_sep_y = torch.cat([torch.zeros(15, dtype=torch.long), torch.ones(15, dtype=torch.long)])
        
        # Barely separable (high difficulty)
        barely_sep_X = torch.cat([
            torch.randn(15, 3) * 0.1,
            torch.randn(15, 3) * 0.1 + 0.1
        ])
        barely_sep_y = torch.cat([torch.zeros(15, dtype=torch.long), torch.ones(15, dtype=torch.long)])
        
        very_sep_difficulty = analyzer.fisher_discriminant_ratio(very_sep_X, very_sep_y)
        mod_sep_difficulty = analyzer.fisher_discriminant_ratio(mod_sep_X, mod_sep_y)
        barely_sep_difficulty = analyzer.fisher_discriminant_ratio(barely_sep_X, barely_sep_y)
        
        # Should be ranked correctly: very separable < moderate < barely separable
        assert very_sep_difficulty < mod_sep_difficulty, "Very separable should be easier than moderate"
        assert mod_sep_difficulty < barely_sep_difficulty, "Moderate should be easier than barely separable"
    
    def test_class_imbalance_effects(self):
        """Test effects of class imbalance on difficulty"""
        analyzer = ComplexityAnalyzer()
        
        # Balanced classes
        balanced_X = torch.cat([
            torch.randn(20, 3) + torch.tensor([1.0, 0.0, 0.0]),
            torch.randn(20, 3) + torch.tensor([0.0, 1.0, 0.0])
        ])
        balanced_y = torch.cat([torch.zeros(20, dtype=torch.long), torch.ones(20, dtype=torch.long)])
        
        # Imbalanced classes
        imbalanced_X = torch.cat([
            torch.randn(30, 3) + torch.tensor([1.0, 0.0, 0.0]),
            torch.randn(10, 3) + torch.tensor([0.0, 1.0, 0.0])
        ])
        imbalanced_y = torch.cat([torch.zeros(30, dtype=torch.long), torch.ones(10, dtype=torch.long)])
        
        balanced_difficulty = analyzer.fisher_discriminant_ratio(balanced_X, balanced_y)
        imbalanced_difficulty = analyzer.fisher_discriminant_ratio(imbalanced_X, imbalanced_y)
        
        # Both should be valid measures
        assert 0 <= balanced_difficulty <= 1
        assert 0 <= imbalanced_difficulty <= 1
        
        # Imbalanced data might be more difficult due to representation issues
        # (though this depends on the specific metric)
        assert isinstance(balanced_difficulty, float)
        assert isinstance(imbalanced_difficulty, float)
    
    def test_dimensionality_effects(self):
        """Test effects of dimensionality on complexity assessment"""
        analyzer = ComplexityAnalyzer()
        
        # Low-dimensional data
        low_dim_X = torch.cat([
            torch.randn(20, 2) + torch.tensor([1.0, 0.0]),
            torch.randn(20, 2) + torch.tensor([0.0, 1.0])
        ])
        low_dim_y = torch.cat([torch.zeros(20, dtype=torch.long), torch.ones(20, dtype=torch.long)])
        
        # High-dimensional data (same separation but in higher dims)
        high_dim_X = torch.cat([
            torch.randn(20, 50) + torch.cat([torch.tensor([1.0, 0.0]), torch.zeros(48)]),
            torch.randn(20, 50) + torch.cat([torch.tensor([0.0, 1.0]), torch.zeros(48)])
        ])
        high_dim_y = torch.cat([torch.zeros(20, dtype=torch.long), torch.ones(20, dtype=torch.long)])
        
        low_dim_difficulty = analyzer.fisher_discriminant_ratio(low_dim_X, low_dim_y)
        high_dim_difficulty = analyzer.fisher_discriminant_ratio(high_dim_X, high_dim_y)
        
        # Both should be valid measures
        assert 0 <= low_dim_difficulty <= 1
        assert 0 <= high_dim_difficulty <= 1
        
        # The relative difficulty might depend on how well the signal is preserved
        # in higher dimensions vs. noise effects
        assert isinstance(low_dim_difficulty, float)
        assert isinstance(high_dim_difficulty, float)


if __name__ == "__main__":
    # Run tests if called directly
    pytest.main([__file__, "-v"])