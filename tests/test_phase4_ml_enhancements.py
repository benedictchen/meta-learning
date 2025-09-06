#!/usr/bin/env python3
"""
Test Phase 4 ML-Powered Enhancement Components
==============================================

Tests for the advanced ML-powered features in MetaLearningToolkit:
- FailurePredictionModel
- AlgorithmSelector  
- ABTestingFramework
- CrossTaskKnowledgeTransfer
- PerformanceMonitor
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
import time
from typing import Dict, Any, List
from unittest.mock import patch, MagicMock

from meta_learning.toolkit import (
    MetaLearningToolkit, 
    FailurePredictionModel,
    AlgorithmSelector,
    ABTestingFramework, 
    CrossTaskKnowledgeTransfer,
    PerformanceMonitor
)
from meta_learning.core.episode import Episode


class TestFailurePredictionModel:
    """Test the ML-based failure prediction system."""

    def test_initialization(self):
        """Test FailurePredictionModel initialization."""
        model = FailurePredictionModel()
        
        assert model.feature_history == []
        assert model.failure_history == []
        assert model.prediction_threshold == 0.7
        assert hasattr(model, 'extract_features')
        assert hasattr(model, 'predict_failure_risk')
        assert hasattr(model, 'update_with_outcome')

    def test_feature_extraction(self):
        """Test feature extraction from episodes and algorithm state."""
        model = FailurePredictionModel()
        
        # Create test episode
        support_x = torch.randn(10, 28, 28)  # 5 classes, 2 shots each
        support_y = torch.repeat_interleave(torch.arange(5), 2)
        query_x = torch.randn(15, 28, 28)
        query_y = torch.repeat_interleave(torch.arange(5), 3)
        episode = Episode(support_x, support_y, query_x, query_y)
        
        # Algorithm state
        algorithm_state = {
            'learning_rate': 0.01,
            'inner_steps': 3,
            'loss_history': [1.5, 1.2, 0.8, 0.5]
        }
        
        features = model.extract_features(episode, algorithm_state)
        
        # Verify feature structure
        assert isinstance(features, np.ndarray)
        assert len(features) == 8  # Expected number of features
        
        # Verify feature values are reasonable
        assert features[0] == 10  # n_support
        assert features[1] == 5   # n_classes
        assert 0.0 <= features[2] <= 1.0  # class_balance
        assert features[3] >= 0.0  # avg_distance
        assert features[4] == 0.01  # learning_rate
        assert features[5] == 3     # inner_steps
        assert features[6] == 0.75  # avg_loss
        assert features[7] == 4     # len(loss_history)

    def test_failure_prediction_insufficient_data(self):
        """Test failure prediction with insufficient historical data."""
        model = FailurePredictionModel()
        
        episode = Episode(torch.randn(5, 10), torch.arange(5), 
                         torch.randn(10, 10), torch.repeat_interleave(torch.arange(5), 2))
        algorithm_state = {'learning_rate': 0.01, 'inner_steps': 1, 'loss_history': []}
        
        # With no history, should return neutral prediction
        risk = model.predict_failure_risk(episode, algorithm_state)
        assert risk == 0.5

    def test_failure_prediction_with_history(self):
        """Test failure prediction with sufficient historical data."""
        model = FailurePredictionModel()
        
        # Build up history with some failures and successes
        for i in range(15):
            episode = Episode(torch.randn(5, 10), torch.arange(5),
                            torch.randn(10, 10), torch.repeat_interleave(torch.arange(5), 2))
            algorithm_state = {'learning_rate': 0.01 * (i + 1), 'inner_steps': 1, 'loss_history': [1.0]}
            failed = i % 3 == 0  # Every third episode fails
            
            model.update_with_outcome(episode, algorithm_state, failed)
        
        # Test prediction on new episode
        test_episode = Episode(torch.randn(5, 10), torch.arange(5),
                             torch.randn(10, 10), torch.repeat_interleave(torch.arange(5), 2))
        test_state = {'learning_rate': 0.05, 'inner_steps': 1, 'loss_history': [1.0]}
        
        risk = model.predict_failure_risk(test_episode, test_state)
        
        # Should return a meaningful prediction between 0 and 1
        assert 0.0 <= risk <= 1.0
        assert risk != 0.5  # Should not be neutral with history

    def test_outcome_update_and_history_management(self):
        """Test updating model with outcomes and history management."""
        model = FailurePredictionModel()
        
        episode = Episode(torch.randn(3, 10), torch.arange(3),
                         torch.randn(6, 10), torch.repeat_interleave(torch.arange(3), 2))
        algorithm_state = {'learning_rate': 0.01, 'inner_steps': 2, 'loss_history': [0.5]}
        
        # Test successful outcome
        model.update_with_outcome(episode, algorithm_state, failed=False)
        assert len(model.feature_history) == 1
        assert len(model.failure_history) == 1
        assert model.failure_history[0] == 0.0
        
        # Test failed outcome
        model.update_with_outcome(episode, algorithm_state, failed=True)
        assert len(model.feature_history) == 2
        assert len(model.failure_history) == 2
        assert model.failure_history[1] == 1.0

    def test_history_truncation(self):
        """Test that history is properly truncated to prevent memory issues."""
        model = FailurePredictionModel()
        
        episode = Episode(torch.randn(3, 5), torch.arange(3),
                         torch.randn(6, 5), torch.repeat_interleave(torch.arange(3), 2))
        algorithm_state = {'learning_rate': 0.01, 'inner_steps': 1, 'loss_history': []}
        
        # Add more than the limit (1000)
        for i in range(1100):
            model.update_with_outcome(episode, algorithm_state, failed=i % 2 == 0)
        
        # Should be truncated to 500
        assert len(model.feature_history) == 500
        assert len(model.failure_history) == 500


class TestAlgorithmSelector:
    """Test automatic algorithm selection system."""

    def test_initialization(self):
        """Test AlgorithmSelector initialization."""
        selector = AlgorithmSelector()
        
        assert 'maml' in selector.algorithm_performance
        assert 'test_time_compute' in selector.algorithm_performance
        assert 'protonet' in selector.algorithm_performance
        
        for alg_perf in selector.algorithm_performance.values():
            assert isinstance(alg_perf, list)
            assert len(alg_perf) == 0

    def test_algorithm_selection_heuristics(self):
        """Test heuristic-based algorithm selection."""
        selector = AlgorithmSelector()
        
        # Very few-shot scenario
        few_shot_episode = Episode(
            torch.randn(3, 10), torch.arange(3),  # 3 classes, 1 shot each
            torch.randn(6, 10), torch.repeat_interleave(torch.arange(3), 2)
        )
        selected = selector.select_algorithm(few_shot_episode)
        assert selected == 'test_time_compute'  # Best for very few-shot
        
        # Many classes scenario
        many_class_episode = Episode(
            torch.randn(22, 10), torch.repeat_interleave(torch.arange(11), 2),  # 11 classes
            torch.randn(33, 10), torch.repeat_interleave(torch.arange(11), 3)
        )
        selected = selector.select_algorithm(many_class_episode)
        assert selected == 'protonet'  # Best for many classes
        
        # General scenario
        general_episode = Episode(
            torch.randn(10, 10), torch.repeat_interleave(torch.arange(5), 2),  # 5 classes, 2 shots
            torch.randn(15, 10), torch.repeat_interleave(torch.arange(5), 3)
        )
        selected = selector.select_algorithm(general_episode)
        assert selected == 'maml'  # General purpose

    def test_performance_tracking(self):
        """Test performance tracking for algorithm selection."""
        selector = AlgorithmSelector()
        
        episode = Episode(torch.randn(6, 10), torch.repeat_interleave(torch.arange(3), 2),
                         torch.randn(9, 10), torch.repeat_interleave(torch.arange(3), 3))
        
        # Track performance for MAML
        selector.update_performance('maml', episode, 0.85)
        
        assert len(selector.algorithm_performance['maml']) == 1
        perf_entry = selector.algorithm_performance['maml'][0]
        
        assert perf_entry['accuracy'] == 0.85
        assert perf_entry['n_support'] == 6
        assert perf_entry['n_classes'] == 3
        assert 'timestamp' in perf_entry

    def test_performance_history_limits(self):
        """Test that performance history is limited to prevent memory issues."""
        selector = AlgorithmSelector()
        
        episode = Episode(torch.randn(4, 5), torch.repeat_interleave(torch.arange(2), 2),
                         torch.randn(6, 5), torch.repeat_interleave(torch.arange(2), 3))
        
        # Add more than the limit (100)
        for i in range(120):
            selector.update_performance('maml', episode, 0.5 + i * 0.001)
        
        # Should be truncated to 50
        assert len(selector.algorithm_performance['maml']) == 50


class TestABTestingFramework:
    """Test A/B testing framework for algorithm comparison."""

    def test_initialization(self):
        """Test ABTestingFramework initialization."""
        framework = ABTestingFramework()
        
        assert framework.test_groups == {}
        assert framework.results_cache == {}

    def test_ab_test_creation(self):
        """Test creating A/B test configurations."""
        framework = ABTestingFramework()
        
        algorithms = ['maml', 'protonet', 'test_time_compute']
        framework.create_ab_test('algorithm_comparison', algorithms)
        
        assert 'algorithm_comparison' in framework.test_groups
        test_config = framework.test_groups['algorithm_comparison']
        
        assert test_config['algorithms'] == algorithms
        assert len(test_config['allocation_ratio']) == 3
        assert sum(test_config['allocation_ratio']) == pytest.approx(1.0)
        assert 'results' in test_config

    def test_ab_test_creation_custom_allocation(self):
        """Test creating A/B test with custom allocation ratios."""
        framework = ABTestingFramework()
        
        algorithms = ['maml', 'protonet']
        allocation = [0.7, 0.3]
        framework.create_ab_test('weighted_test', algorithms, allocation)
        
        test_config = framework.test_groups['weighted_test']
        assert test_config['allocation_ratio'] == allocation

    def test_ab_test_creation_validation(self):
        """Test validation in A/B test creation."""
        framework = ABTestingFramework()
        
        # Mismatched lengths should raise error
        with pytest.raises(ValueError):
            framework.create_ab_test('bad_test', ['alg1', 'alg2'], [0.5, 0.3, 0.2])

    def test_algorithm_assignment(self):
        """Test deterministic algorithm assignment."""
        framework = ABTestingFramework()
        framework.create_ab_test('test1', ['alg1', 'alg2'], [0.5, 0.5])
        
        # Same episode_id should get same assignment
        episode_id = 'episode_123'
        assignment1 = framework.assign_algorithm('test1', episode_id)
        assignment2 = framework.assign_algorithm('test1', episode_id)
        
        assert assignment1 == assignment2
        assert assignment1 in ['alg1', 'alg2']

    def test_algorithm_assignment_distribution(self):
        """Test that algorithm assignment respects allocation ratios."""
        framework = ABTestingFramework()
        framework.create_ab_test('distribution_test', ['alg1', 'alg2'], [0.8, 0.2])
        
        assignments = []
        for i in range(1000):
            episode_id = f'episode_{i}'
            assignment = framework.assign_algorithm('distribution_test', episode_id)
            assignments.append(assignment)
        
        alg1_count = assignments.count('alg1')
        alg2_count = assignments.count('alg2')
        
        # Should roughly match 80/20 distribution (within 5% tolerance)
        assert 750 <= alg1_count <= 850  # 80% ± 5%
        assert 150 <= alg2_count <= 250  # 20% ± 5%

    def test_result_recording(self):
        """Test recording A/B test results."""
        framework = ABTestingFramework()
        framework.create_ab_test('result_test', ['alg1', 'alg2'])
        
        result = {'accuracy': 0.85, 'loss': 0.15}
        framework.record_result('result_test', 'alg1', result)
        
        assert len(framework.test_groups['result_test']['results']['alg1']) == 1
        assert framework.test_groups['result_test']['results']['alg1'][0] == result

    def test_ab_test_analysis(self):
        """Test A/B test analysis and results."""
        framework = ABTestingFramework()
        framework.create_ab_test('analysis_test', ['alg1', 'alg2'])
        
        # Record some results
        for i in range(10):
            framework.record_result('analysis_test', 'alg1', {'accuracy': 0.8 + i * 0.01})
            framework.record_result('analysis_test', 'alg2', {'accuracy': 0.7 + i * 0.01})
        
        analysis = framework.analyze_ab_test('analysis_test')
        
        assert 'alg1' in analysis
        assert 'alg2' in analysis
        
        # Check alg1 stats
        alg1_stats = analysis['alg1']
        assert 'mean_accuracy' in alg1_stats
        assert 'std_accuracy' in alg1_stats
        assert 'n_samples' in alg1_stats
        assert alg1_stats['n_samples'] == 10
        assert alg1_stats['mean_accuracy'] == pytest.approx(0.845, abs=0.01)

    def test_nonexistent_test_handling(self):
        """Test handling of non-existent tests."""
        framework = ABTestingFramework()
        
        with pytest.raises(ValueError):
            framework.assign_algorithm('nonexistent_test', 'episode_1')
        
        analysis = framework.analyze_ab_test('nonexistent_test')
        assert analysis == {}


class TestCrossTaskKnowledgeTransfer:
    """Test cross-task knowledge transfer system."""

    def test_initialization(self):
        """Test CrossTaskKnowledgeTransfer initialization."""
        transfer = CrossTaskKnowledgeTransfer()
        
        assert transfer.task_embeddings == {}
        assert transfer.knowledge_base == {}
        assert transfer.transfer_history == []

    def test_task_embedding_computation(self):
        """Test task embedding computation."""
        transfer = CrossTaskKnowledgeTransfer()
        
        episode = Episode(
            torch.randn(6, 20), torch.repeat_interleave(torch.arange(3), 2),
            torch.randn(9, 20), torch.repeat_interleave(torch.arange(3), 3)
        )
        
        embedding = transfer.compute_task_embedding(episode)
        
        assert isinstance(embedding, np.ndarray)
        assert len(embedding) == 23  # 3 + 10 + 10 (task stats + mean + std features)
        
        # Check task statistics
        assert embedding[0] == 6   # n_support
        assert embedding[1] == 3   # n_classes
        assert 0.0 <= embedding[2] <= 1.0  # class_balance

    def test_similar_task_finding(self):
        """Test finding similar tasks for knowledge transfer."""
        transfer = CrossTaskKnowledgeTransfer()
        
        # Store some task embeddings
        for i in range(5):
            episode = Episode(
                torch.randn(4, 10), torch.repeat_interleave(torch.arange(2), 2),
                torch.randn(6, 10), torch.repeat_interleave(torch.arange(2), 3)
            )
            embedding = transfer.compute_task_embedding(episode)
            transfer.task_embeddings[f'task_{i}'] = embedding
        
        # Find similar tasks for new episode
        query_episode = Episode(
            torch.randn(4, 10), torch.repeat_interleave(torch.arange(2), 2),
            torch.randn(6, 10), torch.repeat_interleave(torch.arange(2), 3)
        )
        
        similar_tasks = transfer.find_similar_tasks(query_episode, top_k=3)
        
        assert len(similar_tasks) == 3
        for task_id, similarity in similar_tasks:
            assert isinstance(task_id, str)
            assert 0.0 <= similarity <= 1.0
        
        # Results should be sorted by similarity
        similarities = [sim for _, sim in similar_tasks]
        assert similarities == sorted(similarities, reverse=True)

    def test_knowledge_transfer(self):
        """Test knowledge transfer from similar tasks."""
        transfer = CrossTaskKnowledgeTransfer()
        
        # Store successful task knowledge
        episode1 = Episode(torch.randn(4, 10), torch.arange(2).repeat(2),
                          torch.randn(6, 10), torch.arange(2).repeat(3))
        
        task_result = {
            'accuracy': 0.9,
            'learning_rate': 0.05,
            'inner_steps': 3
        }
        transfer.store_task_knowledge(episode1, 'successful_task', task_result)
        
        # Try to transfer knowledge to similar episode
        similar_episode = Episode(torch.randn(4, 10), torch.arange(2).repeat(2),
                                torch.randn(6, 10), torch.arange(2).repeat(3))
        
        base_config = {'learning_rate': 0.01, 'inner_steps': 1}
        transferred_config = transfer.transfer_knowledge(similar_episode, base_config)
        
        # Should potentially transfer knowledge (depends on similarity threshold)
        assert isinstance(transferred_config, dict)
        assert 'learning_rate' in transferred_config
        assert 'inner_steps' in transferred_config

    def test_knowledge_storage_and_cleanup(self):
        """Test knowledge storage and automatic cleanup."""
        transfer = CrossTaskKnowledgeTransfer()
        
        # Store knowledge up to the limit
        for i in range(600):  # More than the 500 limit
            episode = Episode(torch.randn(2, 5), torch.arange(2),
                            torch.randn(4, 5), torch.arange(2).repeat(2))
            result = {'accuracy': 0.5 + i * 0.001, 'learning_rate': 0.01}
            transfer.store_task_knowledge(episode, f'task_{i}', result)
        
        # Should be cleaned up to reasonable size
        assert len(transfer.knowledge_base) <= 500
        assert len(transfer.task_embeddings) <= 500

    def test_transfer_history_tracking(self):
        """Test that transfer history is properly tracked."""
        transfer = CrossTaskKnowledgeTransfer()
        
        # Store a successful task
        episode1 = Episode(torch.randn(4, 8), torch.arange(2).repeat(2),
                          torch.randn(6, 8), torch.arange(2).repeat(3))
        result = {'accuracy': 0.85, 'learning_rate': 0.03, 'inner_steps': 2}
        transfer.store_task_knowledge(episode1, 'source_task', result)
        
        # Create very similar episode
        similar_episode = Episode(torch.randn(4, 8), torch.arange(2).repeat(2),
                                torch.randn(6, 8), torch.arange(2).repeat(3))
        
        # Try transfer (may or may not happen based on similarity)
        base_config = {'learning_rate': 0.01}
        transfer.transfer_knowledge(similar_episode, base_config)
        
        # Check that transfer system is tracking
        assert hasattr(transfer, 'transfer_history')
        assert isinstance(transfer.transfer_history, list)


class TestPerformanceMonitor:
    """Test advanced performance monitoring system."""

    def test_initialization(self):
        """Test PerformanceMonitor initialization."""
        monitor = PerformanceMonitor()
        
        assert monitor.metrics_history == []
        assert monitor.performance_trends == {}
        assert 'accuracy_drop' in monitor.alert_thresholds
        assert 'loss_spike' in monitor.alert_thresholds
        assert 'training_time' in monitor.alert_thresholds

    def test_metrics_recording(self):
        """Test recording performance metrics."""
        monitor = PerformanceMonitor()
        
        metrics = {'accuracy': 0.85, 'loss': 0.15, 'training_time': 5.2}
        monitor.record_metrics(metrics)
        
        assert len(monitor.metrics_history) == 1
        entry = monitor.metrics_history[0]
        
        assert 'timestamp' in entry
        assert entry['metrics'] == metrics
        assert isinstance(entry['timestamp'], float)

    def test_metrics_history_management(self):
        """Test that metrics history is properly managed."""
        monitor = PerformanceMonitor()
        
        # Add more than the limit
        for i in range(1100):
            metrics = {'accuracy': 0.5 + i * 0.001}
            monitor.record_metrics(metrics, timestamp=float(i))
        
        # Should be truncated
        assert len(monitor.metrics_history) == 500

    def test_trend_analysis(self):
        """Test performance trend analysis."""
        monitor = PerformanceMonitor()
        
        # Record metrics with clear trend
        for i in range(15):
            metrics = {
                'accuracy': 0.5 + i * 0.02,  # Increasing trend
                'loss': 1.0 - i * 0.05,      # Decreasing trend
                'training_time': 10.0 + i    # Increasing trend
            }
            monitor.record_metrics(metrics, timestamp=float(i))
        
        # Check trends were computed
        assert 'accuracy' in monitor.performance_trends
        assert 'loss' in monitor.performance_trends
        assert 'training_time' in monitor.performance_trends
        
        # Accuracy should have positive trend
        acc_trend = monitor.performance_trends['accuracy']
        assert 'trend' in acc_trend
        assert 'recent_mean' in acc_trend
        assert 'recent_std' in acc_trend
        assert acc_trend['trend'] > 0  # Increasing

    def test_alert_system(self):
        """Test performance alert system."""
        monitor = PerformanceMonitor()
        
        # Record some baseline performance
        for i in range(10):
            monitor.record_metrics({'accuracy': 0.8}, timestamp=float(i))
        
        # Record a significant drop in accuracy
        with pytest.warns(UserWarning, match="Performance Alert"):
            monitor.record_metrics({'accuracy': 0.6}, timestamp=10.0)

    def test_optimization_suggestions(self):
        """Test optimization suggestions based on trends."""
        monitor = PerformanceMonitor()
        
        # Create decreasing accuracy trend
        for i in range(15):
            metrics = {'accuracy': 0.9 - i * 0.01}  # Decreasing
            monitor.record_metrics(metrics, timestamp=float(i))
        
        suggestions = monitor.get_optimization_suggestions()
        
        assert isinstance(suggestions, list)
        assert len(suggestions) > 0
        assert any('learning rate' in suggestion.lower() for suggestion in suggestions)


class TestPhase4Integration:
    """Test integration of all Phase 4 ML components in MetaLearningToolkit."""

    def test_phase4_components_initialization(self):
        """Test that Phase 4 components are properly initialized."""
        toolkit = MetaLearningToolkit()
        
        # Check that all Phase 4 components exist
        assert hasattr(toolkit, 'failure_prediction_model')
        assert hasattr(toolkit, 'algorithm_selector')
        assert hasattr(toolkit, 'ab_testing_framework')
        assert hasattr(toolkit, 'knowledge_transfer_system')
        assert hasattr(toolkit, 'performance_monitor')
        
        assert isinstance(toolkit.failure_prediction_model, FailurePredictionModel)
        assert isinstance(toolkit.algorithm_selector, AlgorithmSelector)
        assert isinstance(toolkit.ab_testing_framework, ABTestingFramework)
        assert isinstance(toolkit.knowledge_transfer_system, CrossTaskKnowledgeTransfer)
        assert isinstance(toolkit.performance_monitor, PerformanceMonitor)

    def test_enable_failure_prediction(self):
        """Test enabling failure prediction system."""
        toolkit = MetaLearningToolkit()
        
        assert not toolkit.failure_prediction_enabled
        
        toolkit.enable_failure_prediction(enable_ml_prediction=True)
        
        assert toolkit.failure_prediction_enabled

    def test_enable_automatic_algorithm_selection(self):
        """Test enabling automatic algorithm selection."""
        toolkit = MetaLearningToolkit()
        
        assert not toolkit.auto_algorithm_selection_enabled
        
        toolkit.enable_automatic_algorithm_selection(enable_data_analysis=True)
        
        assert toolkit.auto_algorithm_selection_enabled

    def test_enable_realtime_optimization(self):
        """Test enabling real-time optimization with A/B testing."""
        toolkit = MetaLearningToolkit()
        
        assert not toolkit.realtime_optimization_enabled
        
        toolkit.enable_realtime_optimization(enable_ab_testing=True)
        
        assert toolkit.realtime_optimization_enabled
        # Should have created a default A/B test
        assert 'learning_rate_test' in toolkit.ab_testing_framework.test_groups

    def test_enable_cross_task_knowledge_transfer(self):
        """Test enabling cross-task knowledge transfer."""
        toolkit = MetaLearningToolkit()
        
        assert not toolkit.cross_task_transfer_enabled
        
        toolkit.enable_cross_task_knowledge_transfer(enable_continual_improvement=True)
        
        assert toolkit.cross_task_transfer_enabled

    def test_prediction_and_prevention_workflow(self):
        """Test the failure prediction and prevention workflow."""
        toolkit = MetaLearningToolkit()
        toolkit.enable_failure_prediction()
        
        episode = Episode(torch.randn(6, 10), torch.repeat_interleave(torch.arange(3), 2),
                         torch.randn(9, 10), torch.repeat_interleave(torch.arange(3), 3))
        algorithm_state = {'learning_rate': 0.01, 'inner_steps': 1, 'loss_history': [1.0]}
        
        prediction = toolkit.predict_and_prevent_failures(episode, algorithm_state)
        
        assert isinstance(prediction, dict)
        assert 'failure_risk' in prediction
        assert 'recommendations' in prediction
        assert 0.0 <= prediction['failure_risk'] <= 1.0
        assert isinstance(prediction['recommendations'], list)

    def test_optimization_insights(self):
        """Test getting optimization insights from all systems."""
        toolkit = MetaLearningToolkit()
        toolkit.enable_failure_prediction()
        toolkit.enable_realtime_optimization()
        
        insights = toolkit.get_optimization_insights()
        
        assert isinstance(insights, dict)
        assert 'performance_trends' in insights
        assert 'optimization_suggestions' in insights
        assert 'knowledge_transfers' in insights
        assert 'ab_test_results' in insights

    def test_episode_outcome_recording(self):
        """Test recording episode outcomes across all ML systems."""
        toolkit = MetaLearningToolkit()
        toolkit.enable_failure_prediction()
        toolkit.enable_automatic_algorithm_selection()
        toolkit.enable_realtime_optimization()
        toolkit.enable_cross_task_knowledge_transfer()
        
        episode = Episode(torch.randn(4, 10), torch.repeat_interleave(torch.arange(2), 2),
                         torch.randn(6, 10), torch.repeat_interleave(torch.arange(2), 3))
        
        results = {
            'query_accuracy': 0.85,
            'query_loss': 0.15,
            'learning_rate': 0.01,
            'inner_steps': 2
        }
        
        toolkit.record_episode_outcome(episode, results, 'maml')
        
        # Check that all systems were updated
        assert len(toolkit.failure_prediction_model.feature_history) == 1
        assert len(toolkit.algorithm_selector.algorithm_performance['maml']) == 1
        assert len(toolkit.performance_monitor.metrics_history) == 1
        assert len(toolkit.knowledge_transfer_system.knowledge_base) == 1


if __name__ == "__main__":
    pytest.main([__file__])