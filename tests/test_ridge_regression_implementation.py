"""
Tests for Ridge Regression Implementation
========================================

Comprehensive tests for the modular ridge regression implementation,
covering core functionality, Episode integration, and error handling.
"""

import pytest
import torch
import numpy as np
from meta_learning.algorithms.ridge_regression import (
    RidgeRegression,
    ridge_regression,
    safe_ridge_regression,
    ridge_cross_validation
)
from meta_learning.algorithms.ridge_regression.ridge_core import (
    create_ridge_classifier,
    batch_ridge_regression
)
from meta_learning.algorithms.ridge_regression.integration import (
    RidgeEpisodeClassifier,
    ridge_episode_loss,
    ridge_episode_accuracy,
    analyze_episode_difficulty
)
from meta_learning.core.episode import Episode


class TestRidgeRegressionCore:
    """Test core RidgeRegression class functionality."""
    
    def test_basic_classification(self):
        """Test basic classification with synthetic data."""
        torch.manual_seed(42)
        X = torch.randn(50, 20)
        y = torch.randint(0, 3, (50,))
        
        model = RidgeRegression(reg_lambda=0.01)
        fit_info = model.fit(X, y)
        predictions = model.predict(X)
        
        assert predictions.shape == (50, 3)  # 3 classes
        assert fit_info['method'] in ['standard', 'woodbury']
        assert fit_info['n_samples'] == 50
        assert fit_info['n_features'] == 20
        
    def test_regression_mode(self):
        """Test regression with continuous targets."""
        torch.manual_seed(42)
        X = torch.randn(30, 15)
        y = torch.randn(30)
        
        model = RidgeRegression(reg_lambda=0.1)
        fit_info = model.fit(X, y)
        predictions = model.predict(X)
        
        assert predictions.shape == (30, 1)
        assert not torch.isnan(predictions).any()
        
    def test_woodbury_vs_standard_consistency(self):
        """Test that Woodbury and standard solvers give similar results."""
        torch.manual_seed(42)
        X = torch.randn(15, 25)  # n_samples < n_features (triggers Woodbury)
        y = torch.randint(0, 2, (15,))
        
        model_standard = RidgeRegression(reg_lambda=0.1, use_woodbury=False)
        model_woodbury = RidgeRegression(reg_lambda=0.1, use_woodbury=True)
        
        fit_info1 = model_standard.fit(X, y)
        fit_info2 = model_woodbury.fit(X, y)
        
        pred1 = model_standard.predict(X)
        pred2 = model_woodbury.predict(X)
        
        assert fit_info1['method'] == 'standard'
        assert fit_info2['method'] == 'woodbury'
        # Results should be similar (not exact due to numerical differences)
        assert torch.allclose(pred1, pred2, atol=1e-3)
        
    def test_preprocessing_options(self):
        """Test different preprocessing methods."""
        torch.manual_seed(42)
        X = torch.randn(40, 10) * 10 + 5  # Non-normalized data
        y = torch.randint(0, 2, (40,))
        
        # Test standardization
        model_std = RidgeRegression(preprocessing='standardize')
        model_std.fit(X, y)
        pred_std = model_std.predict(X)
        
        # Test normalization
        model_norm = RidgeRegression(preprocessing='normalize')
        model_norm.fit(X, y)
        pred_norm = model_norm.predict(X)
        
        # Test no preprocessing
        model_none = RidgeRegression(preprocessing='none')
        model_none.fit(X, y)
        pred_none = model_none.predict(X)
        
        assert pred_std.shape == pred_norm.shape == pred_none.shape
        # Different preprocessing should give different results
        assert not torch.allclose(pred_std, pred_none, atol=1e-2)
        
    def test_uncertainty_estimation(self):
        """Test uncertainty estimation functionality."""
        torch.manual_seed(42)
        X = torch.randn(25, 8)
        y = torch.randint(0, 3, (25,))
        
        model = RidgeRegression(reg_lambda=0.05)
        model.fit(X, y)
        
        predictions, uncertainty = model.predict(X, return_uncertainty=True)
        
        assert predictions.shape == (25, 3)
        assert 'prediction_std' in uncertainty
        assert 'parameter_std' in uncertainty
        assert 'covariance_matrix' in uncertainty
        assert uncertainty['prediction_std'].shape == (25, 3)


class TestRidgeFunctionInterface:
    """Test functional ridge regression interface."""
    
    def test_function_interface_basic(self):
        """Test ridge_regression function."""
        torch.manual_seed(42)
        X = torch.randn(20, 10)
        y = torch.randint(0, 2, (20,))
        
        weights, bias = ridge_regression(X, y, reg_lambda=0.1, num_classes=2)
        
        assert weights.shape == (10, 2)
        assert bias.shape == (1, 2)
        assert not torch.isnan(weights).any()
        assert not torch.isnan(bias).any()
        
    def test_safe_ridge_regression(self):
        """Test safe wrapper functionality."""
        torch.manual_seed(42)
        X = torch.randn(15, 8)
        y = torch.randint(0, 3, (15,))
        
        # Normal case should work
        weights, bias = safe_ridge_regression(X, y, reg_lambda=0.1, num_classes=3)
        assert weights.shape == (8, 3)
        assert bias.shape == (1, 3)
        
        # Extreme case should fallback gracefully
        extreme_X = torch.full((5, 5), float('inf'))
        extreme_y = torch.randint(0, 2, (5,))
        
        with pytest.warns(UserWarning):
            safe_weights, safe_bias = safe_ridge_regression(
                extreme_X, extreme_y, reg_lambda=0.1, num_classes=2
            )
        
        # Should return zeros as fallback
        assert torch.allclose(safe_weights, torch.zeros(5, 2))
        
    def test_cross_validation(self):
        """Test cross-validation for lambda selection."""
        torch.manual_seed(42)
        X = torch.randn(50, 12)
        y = torch.randint(0, 2, (50,))
        
        lambda_candidates = [0.001, 0.01, 0.1, 1.0]
        best_lambda = ridge_cross_validation(
            X, y, lambda_candidates=lambda_candidates, n_folds=3
        )
        
        assert best_lambda in lambda_candidates
        assert isinstance(best_lambda, (int, float))


class TestEpisodeIntegration:
    """Test Episode-based integration."""
    
    def test_ridge_episode_classifier(self):
        """Test RidgeEpisodeClassifier functionality."""
        torch.manual_seed(42)
        support_x = torch.randn(20, 16)
        support_y = torch.randint(0, 4, (20,))
        query_x = torch.randn(10, 16)
        query_y = torch.randint(0, 4, (10,))
        
        episode = Episode(support_x, support_y, query_x, query_y)
        
        classifier = RidgeEpisodeClassifier(reg_lambda=0.1)
        query_logits = classifier(episode)
        
        assert query_logits.shape == (10, 4)
        assert not torch.isnan(query_logits).any()
        
    def test_episode_loss_and_accuracy(self):
        """Test episode loss and accuracy computation."""
        torch.manual_seed(42)
        support_x = torch.randn(15, 12)
        support_y = torch.randint(0, 3, (15,))
        query_x = torch.randn(9, 12)
        query_y = torch.randint(0, 3, (9,))
        
        episode = Episode(support_x, support_y, query_x, query_y)
        
        # Test different loss types
        ce_loss = ridge_episode_loss(episode, loss_type='cross_entropy')
        mse_loss = ridge_episode_loss(episode, loss_type='mse')
        acc_loss = ridge_episode_loss(episode, loss_type='accuracy')
        
        assert isinstance(ce_loss.item(), float)
        assert isinstance(mse_loss.item(), float)
        assert isinstance(acc_loss.item(), float)
        assert 0 <= acc_loss.item() <= 1  # Accuracy loss is 1 - accuracy
        
        # Test accuracy directly
        accuracy = ridge_episode_accuracy(episode, reg_lambda=0.1)
        assert isinstance(accuracy, float)
        assert 0 <= accuracy <= 1
        
    def test_episode_difficulty_analysis(self):
        """Test episode difficulty analysis."""
        torch.manual_seed(42)
        
        # Create easy episode (well-separated classes)
        easy_support_x = torch.cat([
            torch.randn(5, 10) + 2,  # Class 0
            torch.randn(5, 10) - 2   # Class 1
        ])
        easy_support_y = torch.cat([torch.zeros(5), torch.ones(5)]).long()
        easy_query_x = torch.randn(4, 10)
        easy_query_y = torch.randint(0, 2, (4,))
        easy_episode = Episode(easy_support_x, easy_support_y, easy_query_x, easy_query_y)
        
        # Create hard episode (overlapping classes)
        hard_support_x = torch.cat([
            torch.randn(5, 10) * 0.1,  # Class 0
            torch.randn(5, 10) * 0.1   # Class 1
        ])
        hard_support_y = torch.cat([torch.zeros(5), torch.ones(5)]).long()
        hard_query_x = torch.randn(4, 10)
        hard_query_y = torch.randint(0, 2, (4,))
        hard_episode = Episode(hard_support_x, hard_support_y, hard_query_x, hard_query_y)
        
        easy_analysis = analyze_episode_difficulty(easy_episode)
        hard_analysis = analyze_episode_difficulty(hard_episode)
        
        assert 'difficulty_score' in easy_analysis
        assert 'difficulty_score' in hard_analysis
        assert 'n_samples' in easy_analysis
        assert 'ridge_accuracy' in easy_analysis
        
        # Easy episode should have higher accuracy than hard episode
        assert easy_analysis['ridge_accuracy'] >= hard_analysis['ridge_accuracy']


class TestBatchProcessing:
    """Test batch processing capabilities."""
    
    def test_batch_ridge_regression(self):
        """Test batch processing of multiple ridge regression problems."""
        torch.manual_seed(42)
        batch_size = 3
        n_samples, n_features, n_outputs = 10, 8, 2
        
        X_batch = torch.randn(batch_size, n_samples, n_features)
        Y_batch = torch.randn(batch_size, n_samples, n_outputs)
        
        weights_batch = batch_ridge_regression(X_batch, Y_batch, reg_lambda=0.1)
        
        assert weights_batch.shape == (batch_size, n_features, n_outputs)
        assert not torch.isnan(weights_batch).any()
        
    def test_create_ridge_classifier(self):
        """Test classifier creation with defaults."""
        classifier = create_ridge_classifier(reg_lambda=0.05, preprocessing='normalize')
        
        assert isinstance(classifier, RidgeRegression)
        assert classifier.reg_lambda == 0.05
        assert classifier.preprocessing == 'normalize'
        assert classifier.bias == True
        assert classifier.lambda_selection == 'cv'


class TestNumericalStability:
    """Test numerical stability and edge cases."""
    
    def test_ill_conditioned_data(self):
        """Test behavior with ill-conditioned data."""
        torch.manual_seed(42)
        # Create nearly singular matrix
        X = torch.randn(20, 10)
        X[:, 1] = X[:, 0] + 1e-8  # Make columns nearly dependent
        y = torch.randint(0, 2, (20,))
        
        model = RidgeRegression(reg_lambda=1e-3)  # Small regularization
        
        with pytest.warns(UserWarning, match="condition number"):
            fit_info = model.fit(X, y)
            predictions = model.predict(X)
        
        assert predictions.shape == (20, 2)
        # Should still produce finite results despite conditioning issues
        assert torch.isfinite(predictions).all()
        
    def test_small_dataset_edge_case(self):
        """Test behavior with very small datasets."""
        torch.manual_seed(42)
        X = torch.randn(3, 5)  # Tiny dataset
        y = torch.randint(0, 2, (3,))
        
        model = RidgeRegression(reg_lambda=0.1)
        fit_info = model.fit(X, y)
        predictions = model.predict(X)
        
        assert predictions.shape == (3, 2)
        assert torch.isfinite(predictions).all()
        
    def test_large_regularization(self):
        """Test behavior with very large regularization."""
        torch.manual_seed(42)
        X = torch.randn(25, 10)
        y = torch.randint(0, 3, (25,))
        
        model = RidgeRegression(reg_lambda=1000.0)  # Very large lambda
        fit_info = model.fit(X, y)
        predictions = model.predict(X)
        
        assert predictions.shape == (25, 3)
        # With large regularization, predictions should be close to uniform
        uniform_pred = torch.full_like(predictions, 1.0/3)
        assert torch.allclose(predictions, uniform_pred, atol=0.1)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])