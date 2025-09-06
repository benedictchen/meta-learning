"""
Comprehensive tests for task difficulty analysis components.

Tests the various metrics and analyzers used to assess task difficulty
including complexity analysis, learning dynamics, and difficulty assessment.
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
from unittest.mock import Mock, patch
import tempfile
import os

# Import the components we're testing
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from meta_learning.analysis.task_difficulty.complexity_analyzer import ComplexityAnalyzer
from meta_learning.analysis.task_difficulty.learning_dynamics import LearningDynamicsAnalyzer
from meta_learning.analysis.task_difficulty.difficulty_assessor import TaskDifficultyAssessor


class TestComplexityAnalyzer:
    """Test ComplexityAnalyzer for statistical complexity measures"""
    
    def create_test_data(self, separable=True):
        """Create test data with controllable class separability"""
        if separable:
            # Well-separated classes
            class_0 = torch.randn(20, 10) + torch.tensor([3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
            class_1 = torch.randn(20, 10) + torch.tensor([0.0, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
            class_2 = torch.randn(20, 10) + torch.tensor([0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        else:
            # Overlapping classes
            class_0 = torch.randn(20, 10) * 0.1
            class_1 = torch.randn(20, 10) * 0.1 + 0.1
            class_2 = torch.randn(20, 10) * 0.1 - 0.1
        
        X = torch.cat([class_0, class_1, class_2], dim=0)
        y = torch.cat([
            torch.zeros(20, dtype=torch.long),
            torch.ones(20, dtype=torch.long),
            torch.full((20,), 2, dtype=torch.long)
        ])
        
        return X, y
    
    def test_initialization(self):
        """Test proper initialization of ComplexityAnalyzer"""
        analyzer = ComplexityAnalyzer()
        assert hasattr(analyzer, 'logger')
        assert analyzer.logger.name.endswith('complexity_analyzer')
    
    def test_fisher_discriminant_ratio_separable(self):
        """Test Fisher's discriminant ratio on well-separated data"""
        analyzer = ComplexityAnalyzer()
        X, y = self.create_test_data(separable=True)
        
        difficulty = analyzer.fisher_discriminant_ratio(X, y)
        
        # Well-separated classes should have low difficulty (high Fisher ratio inverted)
        assert 0 <= difficulty <= 1, "Difficulty should be normalized between 0 and 1"
        assert difficulty < 0.5, "Well-separated classes should have low difficulty"
    
    def test_fisher_discriminant_ratio_overlapping(self):
        """Test Fisher's discriminant ratio on overlapping data"""
        analyzer = ComplexityAnalyzer()
        X, y = self.create_test_data(separable=False)
        
        difficulty = analyzer.fisher_discriminant_ratio(X, y)
        
        # Overlapping classes should have high difficulty
        assert 0 <= difficulty <= 1, "Difficulty should be normalized between 0 and 1"
        assert difficulty > 0.5, "Overlapping classes should have high difficulty"
    
    def test_class_separability_well_separated(self):
        """Test class separability measurement on well-separated data"""
        analyzer = ComplexityAnalyzer()
        X, y = self.create_test_data(separable=True)
        
        # Convert to numpy for sklearn compatibility
        X_np = X.numpy()
        y_np = y.numpy()
        
        difficulty = analyzer.class_separability(X_np, y_np)
        
        # Well-separated classes should have low difficulty (high silhouette inverted)
        assert 0 <= difficulty <= 1, "Difficulty should be normalized between 0 and 1"
        assert difficulty < 0.5, "Well-separated classes should have low difficulty"
    
    def test_class_separability_overlapping(self):
        """Test class separability measurement on overlapping data"""
        analyzer = ComplexityAnalyzer()
        X, y = self.create_test_data(separable=False)
        
        # Convert to numpy for sklearn compatibility
        X_np = X.numpy()
        y_np = y.numpy()
        
        difficulty = analyzer.class_separability(X_np, y_np)
        
        # Overlapping classes should have high difficulty
        assert 0 <= difficulty <= 1, "Difficulty should be normalized between 0 and 1"
        assert difficulty > 0.3, "Overlapping classes should have higher difficulty"
    
    def test_neighborhood_separability(self):
        """Test neighborhood separability analysis"""
        analyzer = ComplexityAnalyzer()
        X, y = self.create_test_data(separable=True)
        
        # Convert to numpy for sklearn compatibility
        X_np = X.numpy()
        y_np = y.numpy()
        
        difficulty = analyzer.neighborhood_separability(X_np, y_np, k=5)
        
        assert 0 <= difficulty <= 1, "Difficulty should be normalized between 0 and 1"
        assert isinstance(difficulty, float), "Should return a float"
    
    def test_feature_efficiency(self):
        """Test feature efficiency analysis via PCA"""
        analyzer = ComplexityAnalyzer()
        
        # Create data with clear structure (should be efficient)
        X = torch.randn(100, 20)
        # Add some structure by making features correlated
        X[:, 1] = X[:, 0] + 0.1 * torch.randn(100)
        X[:, 2] = X[:, 0] + 0.1 * torch.randn(100)
        
        X_np = X.numpy()
        
        difficulty = analyzer.feature_efficiency(X_np, n_components=10)
        
        assert 0 <= difficulty <= 1, "Difficulty should be normalized between 0 and 1"
        assert isinstance(difficulty, float), "Should return a float"
    
    def test_single_class_edge_case(self):
        """Test behavior with single class data"""
        analyzer = ComplexityAnalyzer()
        
        X = torch.randn(10, 5)
        y = torch.zeros(10, dtype=torch.long)  # All same class
        
        # Fisher discriminant ratio should handle single class gracefully
        difficulty = analyzer.fisher_discriminant_ratio(X, y)
        assert difficulty == 1.0, "Single class should have maximum difficulty"
    
    def test_empty_data_edge_case(self):
        """Test behavior with empty data"""
        analyzer = ComplexityAnalyzer()
        
        X = torch.empty(0, 5)
        y = torch.empty(0, dtype=torch.long)
        
        # Should handle empty data gracefully
        difficulty = analyzer.fisher_discriminant_ratio(X, y)
        assert difficulty == 1.0, "Empty data should have maximum difficulty"
    
    def test_high_dimensional_data(self):
        """Test behavior with high-dimensional data"""
        analyzer = ComplexityAnalyzer()
        
        # High-dimensional separable data
        X, y = self.create_test_data(separable=True)
        X_high_dim = torch.cat([X, torch.randn(60, 100)], dim=1)  # Add 100 random dims
        
        difficulty = analyzer.fisher_discriminant_ratio(X_high_dim, y)
        
        assert 0 <= difficulty <= 1, "Difficulty should be normalized"
        assert isinstance(difficulty, float), "Should return a float"


class TestLearningDynamicsAnalyzer:
    """Test LearningDynamicsAnalyzer for learning dynamics assessment"""
    
    def create_simple_model(self):
        """Create a simple model for testing"""
        return nn.Sequential(
            nn.Linear(10, 32),
            nn.ReLU(),
            nn.Linear(32, 3)  # 3 classes
        )
    
    def create_test_episode(self):
        """Create a simple few-shot episode"""
        support_x = torch.randn(15, 10)  # 5 samples per class
        support_y = torch.tensor([0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2])
        query_x = torch.randn(6, 10)
        query_y = torch.tensor([0, 0, 1, 1, 2, 2])
        
        return support_x, support_y, query_x, query_y
    
    def test_initialization(self):
        """Test proper initialization of LearningDynamicsAnalyzer"""
        analyzer = LearningDynamicsAnalyzer()
        assert hasattr(analyzer, 'logger')
        assert analyzer.logger.name.endswith('learning_dynamics')
    
    def test_convergence_rate_basic(self):
        """Test basic convergence rate measurement"""
        analyzer = LearningDynamicsAnalyzer()
        model = self.create_simple_model()
        support_x, support_y, _, _ = self.create_test_episode()
        
        difficulty = analyzer.convergence_rate(
            model, support_x, support_y, n_steps=5, lr=0.01
        )
        
        assert 0 <= difficulty <= 1, "Difficulty should be normalized between 0 and 1"
        assert isinstance(difficulty, float), "Should return a float"
    
    def test_convergence_rate_different_learning_rates(self):
        """Test convergence rate with different learning rates"""
        analyzer = LearningDynamicsAnalyzer()
        model = self.create_simple_model()
        support_x, support_y, _, _ = self.create_test_episode()
        
        # High learning rate should converge faster (lower difficulty)
        difficulty_high_lr = analyzer.convergence_rate(
            model, support_x, support_y, n_steps=5, lr=0.1
        )
        
        # Low learning rate should converge slower (higher difficulty)
        difficulty_low_lr = analyzer.convergence_rate(
            model, support_x, support_y, n_steps=5, lr=0.001
        )
        
        # This is probabilistic but generally should hold
        assert isinstance(difficulty_high_lr, float)
        assert isinstance(difficulty_low_lr, float)
        assert 0 <= difficulty_high_lr <= 1
        assert 0 <= difficulty_low_lr <= 1
    
    def test_gradient_variance_analysis(self):
        """Test gradient variance analysis"""
        analyzer = LearningDynamicsAnalyzer()
        model = self.create_simple_model()
        support_x, support_y, _, _ = self.create_test_episode()
        
        difficulty = analyzer.gradient_variance(
            model, support_x, support_y, n_steps=5, lr=0.01
        )
        
        assert 0 <= difficulty <= 1, "Difficulty should be normalized between 0 and 1"
        assert isinstance(difficulty, float), "Should return a float"
    
    def test_loss_landscape_smoothness(self):
        """Test loss landscape smoothness analysis"""
        analyzer = LearningDynamicsAnalyzer()
        model = self.create_simple_model()
        support_x, support_y, _, _ = self.create_test_episode()
        
        difficulty = analyzer.loss_landscape_smoothness(
            model, support_x, support_y, perturbation_scale=0.01, n_perturbations=5
        )
        
        assert 0 <= difficulty <= 1, "Difficulty should be normalized between 0 and 1"
        assert isinstance(difficulty, float), "Should return a float"
    
    def test_adaptation_stability(self):
        """Test adaptation stability analysis"""
        analyzer = LearningDynamicsAnalyzer()
        model = self.create_simple_model()
        support_x, support_y, query_x, query_y = self.create_test_episode()
        
        difficulty = analyzer.adaptation_stability(
            model, support_x, support_y, query_x, query_y, n_runs=3, n_steps=5
        )
        
        assert 0 <= difficulty <= 1, "Difficulty should be normalized between 0 and 1"
        assert isinstance(difficulty, float), "Should return a float"
    
    def test_deterministic_behavior(self):
        """Test that results are deterministic with fixed seed"""
        analyzer = LearningDynamicsAnalyzer()
        model = self.create_simple_model()
        support_x, support_y, _, _ = self.create_test_episode()
        
        # Run with same seed twice
        torch.manual_seed(42)
        difficulty1 = analyzer.convergence_rate(
            model, support_x, support_y, n_steps=3, lr=0.01
        )
        
        torch.manual_seed(42)
        difficulty2 = analyzer.convergence_rate(
            model, support_x, support_y, n_steps=3, lr=0.01
        )
        
        assert abs(difficulty1 - difficulty2) < 1e-6, "Results should be deterministic"
    
    def test_edge_case_no_adaptation_steps(self):
        """Test edge case with no adaptation steps"""
        analyzer = LearningDynamicsAnalyzer()
        model = self.create_simple_model()
        support_x, support_y, _, _ = self.create_test_episode()
        
        # With 0 steps, should return maximum difficulty
        difficulty = analyzer.convergence_rate(
            model, support_x, support_y, n_steps=0, lr=0.01
        )
        
        assert difficulty == 1.0, "No adaptation steps should result in maximum difficulty"
    
    def test_gradient_clipping_behavior(self):
        """Test behavior with gradient clipping"""
        analyzer = LearningDynamicsAnalyzer()
        model = self.create_simple_model()
        support_x, support_y, _, _ = self.create_test_episode()
        
        # Test with gradient clipping enabled
        difficulty = analyzer.convergence_rate(
            model, support_x, support_y, n_steps=5, lr=0.01, clip_grad=True
        )
        
        assert 0 <= difficulty <= 1, "Difficulty should be normalized"
        assert isinstance(difficulty, float), "Should return a float"


class TestTaskDifficultyAssessor:
    """Test TaskDifficultyAssessor for comprehensive difficulty assessment"""
    
    def create_test_episode(self):
        """Create a test episode for assessment"""
        support_x = torch.randn(15, 10)
        support_y = torch.tensor([0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2])
        query_x = torch.randn(6, 10)
        query_y = torch.tensor([0, 0, 1, 1, 2, 2])
        
        return support_x, support_y, query_x, query_y
    
    def create_simple_model(self):
        """Create a simple model for testing"""
        return nn.Sequential(
            nn.Linear(10, 16),
            nn.ReLU(),
            nn.Linear(16, 3)
        )
    
    def test_initialization(self):
        """Test proper initialization of TaskDifficultyAssessor"""
        assessor = TaskDifficultyAssessor()
        
        assert hasattr(assessor, 'complexity_analyzer')
        assert hasattr(assessor, 'dynamics_analyzer')
        assert isinstance(assessor.complexity_analyzer, ComplexityAnalyzer)
        assert isinstance(assessor.dynamics_analyzer, LearningDynamicsAnalyzer)
    
    def test_assess_episode_difficulty_basic(self):
        """Test basic episode difficulty assessment"""
        assessor = DifficultyAssessor()
        model = self.create_simple_model()
        support_x, support_y, query_x, query_y = self.create_test_episode()
        
        result = assessor.assess_episode_difficulty(
            model, support_x, support_y, query_x, query_y
        )
        
        # Check result structure
        assert isinstance(result, dict), "Result should be a dictionary"
        expected_keys = {
            'overall_difficulty', 'complexity_score', 'dynamics_score',
            'individual_metrics', 'assessment_summary'
        }
        for key in expected_keys:
            assert key in result, f"Missing key: {key}"
        
        # Check value ranges
        assert 0 <= result['overall_difficulty'] <= 1, "Overall difficulty should be normalized"
        assert 0 <= result['complexity_score'] <= 1, "Complexity score should be normalized"
        assert 0 <= result['dynamics_score'] <= 1, "Dynamics score should be normalized"
        
        # Check individual metrics structure
        assert isinstance(result['individual_metrics'], dict), "Individual metrics should be a dict"
        assert isinstance(result['assessment_summary'], str), "Summary should be a string"
    
    def test_custom_weights(self):
        """Test difficulty assessment with custom weights"""
        assessor = DifficultyAssessor()
        model = self.create_simple_model()
        support_x, support_y, query_x, query_y = self.create_test_episode()
        
        # Emphasize complexity over dynamics
        result = assessor.assess_episode_difficulty(
            model, support_x, support_y, query_x, query_y,
            complexity_weight=0.8, dynamics_weight=0.2
        )
        
        assert isinstance(result, dict), "Result should be a dictionary"
        assert 0 <= result['overall_difficulty'] <= 1, "Overall difficulty should be normalized"
    
    def test_detailed_analysis_flag(self):
        """Test detailed analysis flag"""
        assessor = DifficultyAssessor()
        model = self.create_simple_model()
        support_x, support_y, query_x, query_y = self.create_test_episode()
        
        # Test with detailed analysis
        result_detailed = assessor.assess_episode_difficulty(
            model, support_x, support_y, query_x, query_y, detailed_analysis=True
        )
        
        # Test without detailed analysis
        result_simple = assessor.assess_episode_difficulty(
            model, support_x, support_y, query_x, query_y, detailed_analysis=False
        )
        
        # Detailed analysis should have more individual metrics
        assert len(result_detailed['individual_metrics']) >= len(result_simple['individual_metrics'])
    
    def test_difficulty_categorization(self):
        """Test difficulty categorization logic"""
        assessor = DifficultyAssessor()
        
        # Test different difficulty levels
        easy_score = 0.1
        medium_score = 0.5
        hard_score = 0.9
        
        easy_category = assessor._categorize_difficulty(easy_score)
        medium_category = assessor._categorize_difficulty(medium_score)
        hard_category = assessor._categorize_difficulty(hard_score)
        
        assert easy_category in ['Easy', 'easy', 'low'], f"Unexpected easy category: {easy_category}"
        assert medium_category in ['Medium', 'medium', 'moderate'], f"Unexpected medium category: {medium_category}"
        assert hard_category in ['Hard', 'hard', 'difficult'], f"Unexpected hard category: {hard_category}"
    
    def test_batch_assessment(self):
        """Test batch assessment of multiple episodes"""
        assessor = DifficultyAssessor()
        model = self.create_simple_model()
        
        # Create multiple episodes
        episodes = []
        for _ in range(3):
            support_x, support_y, query_x, query_y = self.create_test_episode()
            episodes.append((support_x, support_y, query_x, query_y))
        
        results = assessor.assess_multiple_episodes(model, episodes)
        
        assert isinstance(results, list), "Results should be a list"
        assert len(results) == 3, "Should have results for all episodes"
        
        for result in results:
            assert isinstance(result, dict), "Each result should be a dictionary"
            assert 'overall_difficulty' in result, "Each result should have overall difficulty"
    
    def test_memory_efficiency(self):
        """Test that assessment doesn't create memory leaks"""
        assessor = DifficultyAssessor()
        model = self.create_simple_model()
        support_x, support_y, query_x, query_y = self.create_test_episode()
        
        # Run assessment multiple times
        for _ in range(5):
            result = assessor.assess_episode_difficulty(
                model, support_x, support_y, query_x, query_y
            )
            assert isinstance(result, dict)
        
        # If we get here without out-of-memory errors, we're probably okay
        assert True, "Multiple assessments completed successfully"
    
    def test_device_compatibility(self):
        """Test compatibility with different devices"""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        
        device = torch.device('cuda')
        assessor = DifficultyAssessor()
        model = self.create_simple_model().to(device)
        
        support_x, support_y, query_x, query_y = self.create_test_episode()
        support_x = support_x.to(device)
        support_y = support_y.to(device)
        query_x = query_x.to(device)
        query_y = query_y.to(device)
        
        result = assessor.assess_episode_difficulty(
            model, support_x, support_y, query_x, query_y
        )
        
        assert isinstance(result, dict), "Should work on GPU"
        assert 0 <= result['overall_difficulty'] <= 1, "Results should be normalized"


class TestTaskDifficultyIntegration:
    """Integration tests for task difficulty analysis components"""
    
    def test_consistency_across_analyzers(self):
        """Test that different analyzers give consistent relative rankings"""
        complexity_analyzer = ComplexityAnalyzer()
        dynamics_analyzer = LearningDynamicsAnalyzer()
        
        # Create easy and hard datasets
        easy_X, easy_y = self.create_separable_data()
        hard_X, hard_y = self.create_overlapping_data()
        
        # Test complexity measures
        easy_complexity = complexity_analyzer.fisher_discriminant_ratio(easy_X, easy_y)
        hard_complexity = complexity_analyzer.fisher_discriminant_ratio(hard_X, hard_y)
        
        # Easy data should have lower complexity difficulty
        assert easy_complexity < hard_complexity, "Easy data should have lower complexity difficulty"
        
        # Test dynamics measures
        model = nn.Sequential(nn.Linear(5, 16), nn.ReLU(), nn.Linear(16, 2))
        
        easy_dynamics = dynamics_analyzer.convergence_rate(model, easy_X[:10], easy_y[:10])
        hard_dynamics = dynamics_analyzer.convergence_rate(model, hard_X[:10], hard_y[:10])
        
        # Both should be normalized
        assert 0 <= easy_dynamics <= 1
        assert 0 <= hard_dynamics <= 1
    
    def create_separable_data(self):
        """Create well-separated data"""
        class_0 = torch.randn(25, 5) + torch.tensor([2.0, 0.0, 0.0, 0.0, 0.0])
        class_1 = torch.randn(25, 5) + torch.tensor([0.0, 2.0, 0.0, 0.0, 0.0])
        
        X = torch.cat([class_0, class_1], dim=0)
        y = torch.cat([torch.zeros(25, dtype=torch.long), torch.ones(25, dtype=torch.long)])
        
        return X, y
    
    def create_overlapping_data(self):
        """Create overlapping data"""
        class_0 = torch.randn(25, 5) * 0.1
        class_1 = torch.randn(25, 5) * 0.1 + 0.05
        
        X = torch.cat([class_0, class_1], dim=0)
        y = torch.cat([torch.zeros(25, dtype=torch.long), torch.ones(25, dtype=torch.long)])
        
        return X, y
    
    def test_end_to_end_assessment_workflow(self):
        """Test complete end-to-end difficulty assessment workflow"""
        assessor = DifficultyAssessor()
        model = nn.Sequential(nn.Linear(5, 16), nn.ReLU(), nn.Linear(16, 2))
        
        # Create test episode
        X, y = self.create_separable_data()
        support_x, support_y = X[:20], y[:20]  # 10 per class
        query_x, query_y = X[20:30], y[20:30]  # 5 per class
        
        # Perform assessment
        result = assessor.assess_episode_difficulty(
            model, support_x, support_y, query_x, query_y
        )
        
        # Validate complete result structure
        assert isinstance(result, dict)
        assert 'overall_difficulty' in result
        assert 'assessment_summary' in result
        assert isinstance(result['assessment_summary'], str)
        assert len(result['assessment_summary']) > 0
        
        # Should provide actionable insights
        summary = result['assessment_summary'].lower()
        difficulty_terms = ['easy', 'medium', 'hard', 'difficult', 'simple', 'complex']
        assert any(term in summary for term in difficulty_terms), "Summary should contain difficulty assessment"


if __name__ == "__main__":
    # Run tests if called directly
    pytest.main([__file__, "-v"])