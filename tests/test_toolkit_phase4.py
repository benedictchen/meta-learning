"""
Test Suite for Phase 4 Toolkit Enhancements
===========================================

Comprehensive tests for the advanced Phase 4 features including:
- Machine learning-based failure prediction and auto-recovery
- Automatic algorithm selection based on data characteristics
- Real-time performance optimization with A/B testing
- Cross-task knowledge transfer and continual improvement

Author: Test Suite Generator
"""

import pytest
import torch
import numpy as np
import tempfile
import shutil
import json
import pickle
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, List, Any

from meta_learning.toolkit import MetaLearningToolkit
from meta_learning.shared.types import Episode


class TestFailurePrediction:
    """Test machine learning-based failure prediction and auto-recovery."""
    
    def setup_method(self):
        """Set up toolkit for testing."""
        self.toolkit = MetaLearningToolkit()
        self.toolkit.enable_failure_prediction(
            enable_ml_prediction=True,
            enable_auto_recovery=True
        )
    
    def test_enable_failure_prediction(self):
        """Test enabling failure prediction functionality."""
        # Check that failure prediction components are initialized
        assert hasattr(self.toolkit, 'failure_predictor')
        assert hasattr(self.toolkit, 'failure_history')
        assert hasattr(self.toolkit, 'recovery_strategies')
        assert hasattr(self.toolkit, 'failure_threshold')
        
        # Check default values
        assert self.toolkit.failure_threshold == 0.8
        assert isinstance(self.toolkit.failure_history, list)
        assert isinstance(self.toolkit.recovery_strategies, dict)
    
    def test_collect_failure_features(self):
        """Test collecting features for failure prediction."""
        # Create sample episode data
        episode = Episode(
            support_x=torch.randn(25, 3, 84, 84),
            support_y=torch.randint(0, 5, (25,)),
            query_x=torch.randn(75, 3, 84, 84),
            query_y=torch.randint(0, 5, (75,))
        )
        
        algorithm_params = {
            'learning_rate': 0.001,
            'batch_size': 32,
            'n_inner_steps': 5
        }
        
        features = self.toolkit.collect_failure_features(episode, algorithm_params)
        
        # Check that features are collected
        assert isinstance(features, dict)
        assert 'data_variance' in features
        assert 'class_imbalance' in features
        assert 'data_complexity' in features
        assert 'learning_rate' in features
        assert 'batch_size' in features
        assert 'n_inner_steps' in features
        
        # Check feature types and ranges
        assert isinstance(features['data_variance'], float)
        assert isinstance(features['class_imbalance'], float)
        assert features['class_imbalance'] >= 0.0
        assert features['learning_rate'] == 0.001
        assert features['batch_size'] == 32
    
    def test_predict_failure_probability(self):
        """Test ML-based failure probability prediction."""
        # Create sample features
        features = {
            'data_variance': 0.5,
            'class_imbalance': 0.3,
            'data_complexity': 0.7,
            'learning_rate': 0.001,
            'batch_size': 32,
            'n_inner_steps': 5
        }
        
        probability = self.toolkit.predict_failure_probability(features)
        
        # Check that probability is valid
        assert isinstance(probability, float)
        assert 0.0 <= probability <= 1.0
    
    def test_record_failure_outcome(self):
        """Test recording failure outcomes for model training."""
        features = {
            'data_variance': 0.8,
            'class_imbalance': 0.6,
            'data_complexity': 0.9,
            'learning_rate': 0.01,
            'batch_size': 16
        }
        
        # Record successful outcome
        self.toolkit.record_failure_outcome(features, failed=False, performance=0.85)
        
        # Record failed outcome
        self.toolkit.record_failure_outcome(features, failed=True, performance=0.25)
        
        # Check that outcomes are recorded
        assert len(self.toolkit.failure_history) == 2
        
        # Check first record
        record1 = self.toolkit.failure_history[0]
        assert record1['features'] == features
        assert record1['failed'] == False
        assert record1['performance'] == 0.85
        assert 'timestamp' in record1
        
        # Check second record
        record2 = self.toolkit.failure_history[1]
        assert record2['failed'] == True
        assert record2['performance'] == 0.25
    
    def test_get_recovery_strategy(self):
        """Test getting appropriate recovery strategy."""
        features = {
            'data_variance': 0.9,  # High variance
            'class_imbalance': 0.8,  # High imbalance
            'data_complexity': 0.7,
            'learning_rate': 0.001,
            'batch_size': 32
        }
        
        strategy = self.toolkit.get_recovery_strategy(features)
        
        assert isinstance(strategy, dict)
        assert 'name' in strategy
        assert 'adjustments' in strategy
        assert 'description' in strategy
        
        # Should recommend appropriate adjustments for high variance and imbalance
        adjustments = strategy['adjustments']
        assert isinstance(adjustments, dict)
    
    def test_apply_recovery_strategy(self):
        """Test applying recovery strategy to algorithm parameters."""
        original_params = {
            'learning_rate': 0.001,
            'batch_size': 32,
            'n_inner_steps': 5,
            'regularization': 0.01
        }
        
        strategy = {
            'name': 'reduce_learning_rate',
            'adjustments': {
                'learning_rate_multiplier': 0.5,
                'batch_size_multiplier': 0.75,
                'regularization_multiplier': 2.0
            },
            'description': 'Reduce learning rate and increase regularization'
        }
        
        adjusted_params = self.toolkit.apply_recovery_strategy(original_params, strategy)
        
        assert adjusted_params['learning_rate'] == 0.0005  # 0.001 * 0.5
        assert adjusted_params['batch_size'] == 24  # 32 * 0.75
        assert adjusted_params['regularization'] == 0.02  # 0.01 * 2.0
        assert adjusted_params['n_inner_steps'] == 5  # Unchanged
    
    def test_should_trigger_recovery(self):
        """Test recovery trigger conditions."""
        # Test with high failure probability
        high_prob_features = {'failure_probability': 0.9}
        assert self.toolkit.should_trigger_recovery(high_prob_features)
        
        # Test with low failure probability
        low_prob_features = {'failure_probability': 0.3}
        assert not self.toolkit.should_trigger_recovery(low_prob_features)
        
        # Test with threshold boundary
        threshold_features = {'failure_probability': 0.8}
        assert not self.toolkit.should_trigger_recovery(threshold_features)  # Exactly at threshold
    
    def test_update_failure_predictor(self):
        """Test updating the ML failure predictor model."""
        # Add some training data
        for i in range(10):
            features = {
                'data_variance': np.random.random(),
                'class_imbalance': np.random.random(),
                'data_complexity': np.random.random(),
                'learning_rate': 0.001,
                'batch_size': 32
            }
            failed = np.random.random() > 0.7  # 30% failure rate
            performance = np.random.random() if not failed else np.random.random() * 0.5
            
            self.toolkit.record_failure_outcome(features, failed, performance)
        
        # Update the predictor
        self.toolkit.update_failure_predictor()
        
        # Test that predictor can make predictions
        test_features = {
            'data_variance': 0.5,
            'class_imbalance': 0.3,
            'data_complexity': 0.6,
            'learning_rate': 0.001,
            'batch_size': 32
        }
        
        prediction = self.toolkit.predict_failure_probability(test_features)
        assert isinstance(prediction, float)
        assert 0.0 <= prediction <= 1.0


class TestAutomaticAlgorithmSelection:
    """Test automatic algorithm selection based on data characteristics."""
    
    def setup_method(self):
        """Set up toolkit for testing."""
        self.toolkit = MetaLearningToolkit()
        self.toolkit.enable_automatic_algorithm_selection(
            enable_data_analysis=True,
            fallback_algorithm="maml"
        )
    
    def test_enable_algorithm_selection(self):
        """Test enabling algorithm selection functionality."""
        assert hasattr(self.toolkit, 'algorithm_selector')
        assert hasattr(self.toolkit, 'algorithm_performance_history')
        assert hasattr(self.toolkit, 'data_characteristics_cache')
        assert self.toolkit.fallback_algorithm == "maml"
    
    def test_analyze_data_characteristics(self):
        """Test analyzing data characteristics for algorithm selection."""
        episode = Episode(
            support_x=torch.randn(25, 3, 84, 84),
            support_y=torch.randint(0, 5, (25,)),
            query_x=torch.randn(75, 3, 84, 84),
            query_y=torch.randint(0, 5, (75,))
        )
        
        characteristics = self.toolkit.analyze_data_characteristics(episode)
        
        assert isinstance(characteristics, dict)
        assert 'n_classes' in characteristics
        assert 'shots_per_class' in characteristics
        assert 'data_dimensionality' in characteristics
        assert 'data_variance' in characteristics
        assert 'class_separability' in characteristics
        assert 'feature_complexity' in characteristics
        
        # Check values
        assert characteristics['n_classes'] == 5
        assert characteristics['shots_per_class'] == 5  # 25 support samples / 5 classes
        assert characteristics['data_dimensionality'] == 3 * 84 * 84
        assert isinstance(characteristics['data_variance'], float)
        assert isinstance(characteristics['class_separability'], float)
    
    def test_recommend_algorithm(self):
        """Test algorithm recommendation based on data characteristics."""
        # Test with different data characteristics
        characteristics1 = {
            'n_classes': 5,
            'shots_per_class': 1,  # Few-shot
            'data_variance': 0.3,
            'class_separability': 0.8,
            'feature_complexity': 0.4
        }
        
        recommendation1 = self.toolkit.recommend_algorithm(characteristics1)
        assert isinstance(recommendation1, dict)
        assert 'algorithm' in recommendation1
        assert 'confidence' in recommendation1
        assert 'reasoning' in recommendation1
        assert 0.0 <= recommendation1['confidence'] <= 1.0
        
        # Test with different characteristics (many-shot)
        characteristics2 = {
            'n_classes': 20,
            'shots_per_class': 50,  # Many-shot
            'data_variance': 0.8,
            'class_separability': 0.3,
            'feature_complexity': 0.9
        }
        
        recommendation2 = self.toolkit.recommend_algorithm(characteristics2)
        
        # Recommendations might be different for different data characteristics
        assert isinstance(recommendation2, dict)
        assert 'algorithm' in recommendation2
    
    def test_record_algorithm_performance(self):
        """Test recording algorithm performance for future recommendations."""
        characteristics = {
            'n_classes': 5,
            'shots_per_class': 5,
            'data_variance': 0.5,
            'class_separability': 0.7
        }
        
        # Record performance for different algorithms
        self.toolkit.record_algorithm_performance("maml", characteristics, 0.85, 120.5)
        self.toolkit.record_algorithm_performance("prototypical", characteristics, 0.78, 95.2)
        self.toolkit.record_algorithm_performance("matching_networks", characteristics, 0.71, 88.1)
        
        # Check that performance is recorded
        assert len(self.toolkit.algorithm_performance_history) == 3
        
        # Check first record
        record = self.toolkit.algorithm_performance_history[0]
        assert record['algorithm'] == "maml"
        assert record['characteristics'] == characteristics
        assert record['accuracy'] == 0.85
        assert record['training_time'] == 120.5
        assert 'timestamp' in record
    
    def test_get_best_algorithm_for_characteristics(self):
        """Test finding best algorithm for given data characteristics."""
        # Create similar characteristics
        base_chars = {
            'n_classes': 5,
            'shots_per_class': 5,
            'data_variance': 0.5,
            'class_separability': 0.7
        }
        
        # Record performance history
        similar_chars1 = base_chars.copy()
        similar_chars1['data_variance'] = 0.52
        self.toolkit.record_algorithm_performance("maml", similar_chars1, 0.85, 120.0)
        
        similar_chars2 = base_chars.copy()
        similar_chars2['class_separability'] = 0.68
        self.toolkit.record_algorithm_performance("prototypical", similar_chars2, 0.90, 100.0)
        
        # Find best algorithm
        best_algo = self.toolkit.get_best_algorithm_for_characteristics(base_chars)
        
        assert isinstance(best_algo, str)
        # Should return prototypical due to higher accuracy (0.90 > 0.85)
        assert best_algo == "prototypical"
    
    def test_update_algorithm_selector(self):
        """Test updating the algorithm selection model."""
        # Add performance data for multiple algorithms and characteristics
        for _ in range(20):
            characteristics = {
                'n_classes': np.random.randint(2, 20),
                'shots_per_class': np.random.randint(1, 10),
                'data_variance': np.random.random(),
                'class_separability': np.random.random(),
                'feature_complexity': np.random.random()
            }
            
            algorithms = ["maml", "prototypical", "matching_networks", "relation_networks"]
            algorithm = np.random.choice(algorithms)
            accuracy = np.random.uniform(0.4, 0.95)
            training_time = np.random.uniform(50, 200)
            
            self.toolkit.record_algorithm_performance(algorithm, characteristics, accuracy, training_time)
        
        # Update the selector
        self.toolkit.update_algorithm_selector()
        
        # Test that selector can make recommendations
        test_characteristics = {
            'n_classes': 5,
            'shots_per_class': 3,
            'data_variance': 0.6,
            'class_separability': 0.4,
            'feature_complexity': 0.7
        }
        
        recommendation = self.toolkit.recommend_algorithm(test_characteristics)
        assert isinstance(recommendation, dict)
        assert recommendation['algorithm'] in ["maml", "prototypical", "matching_networks", "relation_networks"]
    
    def test_fallback_algorithm_selection(self):
        """Test fallback to default algorithm when no data available."""
        # Clear performance history
        self.toolkit.algorithm_performance_history.clear()
        
        characteristics = {
            'n_classes': 10,
            'shots_per_class': 2,
            'data_variance': 0.8,
            'class_separability': 0.3
        }
        
        recommendation = self.toolkit.recommend_algorithm(characteristics)
        
        # Should fallback to default algorithm
        assert recommendation['algorithm'] == "maml"
        assert recommendation['confidence'] < 0.5  # Low confidence for fallback


class TestRealtimeOptimization:
    """Test real-time performance optimization with A/B testing."""
    
    def setup_method(self):
        """Set up toolkit for testing."""
        self.toolkit = MetaLearningToolkit()
        self.toolkit.enable_realtime_optimization(
            enable_ab_testing=True,
            optimization_interval=50
        )
    
    def test_enable_realtime_optimization(self):
        """Test enabling real-time optimization functionality."""
        assert hasattr(self.toolkit, 'ab_test_manager')
        assert hasattr(self.toolkit, 'optimization_history')
        assert hasattr(self.toolkit, 'current_experiments')
        assert hasattr(self.toolkit, 'optimization_metrics')
        assert self.toolkit.optimization_interval == 50
    
    def test_create_ab_test(self):
        """Test creating A/B test experiments."""
        control_params = {
            'learning_rate': 0.001,
            'batch_size': 32,
            'n_inner_steps': 5
        }
        
        test_variants = [
            {'learning_rate': 0.002, 'batch_size': 32, 'n_inner_steps': 5},
            {'learning_rate': 0.001, 'batch_size': 64, 'n_inner_steps': 5},
            {'learning_rate': 0.001, 'batch_size': 32, 'n_inner_steps': 10}
        ]
        
        experiment_id = self.toolkit.create_ab_test(
            "learning_rate_optimization",
            control_params,
            test_variants,
            traffic_split=0.25
        )
        
        assert isinstance(experiment_id, str)
        assert experiment_id in self.toolkit.current_experiments
        
        experiment = self.toolkit.current_experiments[experiment_id]
        assert experiment['name'] == "learning_rate_optimization"
        assert experiment['control_params'] == control_params
        assert experiment['test_variants'] == test_variants
        assert experiment['traffic_split'] == 0.25
        assert 'created_at' in experiment
        assert 'results' in experiment
    
    def test_assign_experiment_variant(self):
        """Test assigning users/tasks to experiment variants."""
        # Create test experiment
        control_params = {'learning_rate': 0.001}
        test_variants = [{'learning_rate': 0.002}, {'learning_rate': 0.0005}]
        
        exp_id = self.toolkit.create_ab_test(
            "test_assignment",
            control_params,
            test_variants,
            traffic_split=0.3
        )
        
        # Test assignments for multiple task IDs
        assignments = []
        for task_id in range(100):
            variant, params = self.toolkit.assign_experiment_variant(exp_id, f"task_{task_id}")
            assignments.append(variant)
            
            # Check that params match variant
            if variant == "control":
                assert params == control_params
            else:
                variant_idx = int(variant.split('_')[1])
                assert params == test_variants[variant_idx]
        
        # Check distribution (should be roughly 70% control, 30% test variants)
        control_count = assignments.count("control")
        test_count = len(assignments) - control_count
        
        # Allow some variance in random assignment
        assert control_count >= 50  # At least 50% should be control
        assert test_count >= 20     # At least 20% should be test variants
    
    def test_record_experiment_result(self):
        """Test recording results for A/B test experiments."""
        control_params = {'learning_rate': 0.001}
        test_variants = [{'learning_rate': 0.002}]
        
        exp_id = self.toolkit.create_ab_test("test_results", control_params, test_variants)
        
        # Record results for different variants
        self.toolkit.record_experiment_result(exp_id, "control", "task_1", 0.85, 120.0)
        self.toolkit.record_experiment_result(exp_id, "variant_0", "task_2", 0.88, 110.0)
        self.toolkit.record_experiment_result(exp_id, "control", "task_3", 0.82, 125.0)
        
        experiment = self.toolkit.current_experiments[exp_id]
        results = experiment['results']
        
        assert len(results) == 3
        assert results[0]['variant'] == "control"
        assert results[0]['task_id'] == "task_1"
        assert results[0]['accuracy'] == 0.85
        assert results[0]['training_time'] == 120.0
    
    def test_analyze_experiment_results(self):
        """Test analyzing A/B test experiment results."""
        control_params = {'learning_rate': 0.001}
        test_variants = [{'learning_rate': 0.002}]
        
        exp_id = self.toolkit.create_ab_test("analysis_test", control_params, test_variants)
        
        # Add results with clear difference
        # Control: lower performance
        for i in range(20):
            accuracy = np.random.normal(0.75, 0.05)  # Mean 0.75, std 0.05
            time = np.random.normal(120, 10)
            self.toolkit.record_experiment_result(exp_id, "control", f"control_task_{i}", accuracy, time)
        
        # Variant: higher performance
        for i in range(20):
            accuracy = np.random.normal(0.85, 0.05)  # Mean 0.85, std 0.05
            time = np.random.normal(100, 10)
            self.toolkit.record_experiment_result(exp_id, "variant_0", f"variant_task_{i}", accuracy, time)
        
        analysis = self.toolkit.analyze_experiment_results(exp_id)
        
        assert isinstance(analysis, dict)
        assert 'control_stats' in analysis
        assert 'variant_stats' in analysis
        assert 'significance_test' in analysis
        assert 'recommendation' in analysis
        
        # Check control stats
        control_stats = analysis['control_stats']
        assert 'mean_accuracy' in control_stats
        assert 'mean_training_time' in control_stats
        assert 'sample_size' in control_stats
        assert control_stats['sample_size'] == 20
        
        # Check variant stats
        variant_stats = analysis['variant_stats']
        assert 'variant_0' in variant_stats
        variant_0_stats = variant_stats['variant_0']
        assert variant_0_stats['sample_size'] == 20
        
        # With the data we generated, variant should perform better
        assert variant_0_stats['mean_accuracy'] > control_stats['mean_accuracy']
    
    def test_update_optimization_strategy(self):
        """Test updating optimization strategy based on results."""
        # Create and run experiment
        control_params = {'learning_rate': 0.001, 'batch_size': 32}
        test_variants = [
            {'learning_rate': 0.002, 'batch_size': 32},
            {'learning_rate': 0.001, 'batch_size': 64}
        ]
        
        exp_id = self.toolkit.create_ab_test("strategy_update", control_params, test_variants)
        
        # Add results showing variant_1 (batch_size=64) performs best
        for i in range(10):
            self.toolkit.record_experiment_result(exp_id, "control", f"c_{i}", 0.75, 120)
            self.toolkit.record_experiment_result(exp_id, "variant_0", f"v0_{i}", 0.78, 115)
            self.toolkit.record_experiment_result(exp_id, "variant_1", f"v1_{i}", 0.88, 100)
        
        # Update strategy
        updated_params = self.toolkit.update_optimization_strategy(exp_id)
        
        assert isinstance(updated_params, dict)
        # Should adopt the best performing variant's parameters
        assert updated_params['batch_size'] == 64  # From variant_1
    
    def test_get_current_best_params(self):
        """Test getting current best parameters across all experiments."""
        # Initially should return default parameters
        best_params = self.toolkit.get_current_best_params()
        assert isinstance(best_params, dict)
        
        # After running experiments, should return optimized parameters
        control_params = {'learning_rate': 0.001}
        test_variants = [{'learning_rate': 0.005}]  # Much higher LR
        
        exp_id = self.toolkit.create_ab_test("best_params_test", control_params, test_variants)
        
        # Add results showing higher LR performs better
        for i in range(5):
            self.toolkit.record_experiment_result(exp_id, "control", f"c_{i}", 0.70, 120)
            self.toolkit.record_experiment_result(exp_id, "variant_0", f"v_{i}", 0.90, 80)
        
        # Update and get best params
        self.toolkit.update_optimization_strategy(exp_id)
        updated_best = self.toolkit.get_current_best_params()
        
        # Should include the better learning rate
        assert updated_best['learning_rate'] == 0.005


class TestCrossTaskKnowledgeTransfer:
    """Test cross-task knowledge transfer and continual improvement."""
    
    def setup_method(self):
        """Set up toolkit for testing."""
        self.toolkit = MetaLearningToolkit()
        self.toolkit.enable_cross_task_knowledge_transfer(
            enable_continual_improvement=True,
            memory_size=100
        )
    
    def test_enable_knowledge_transfer(self):
        """Test enabling knowledge transfer functionality."""
        assert hasattr(self.toolkit, 'knowledge_memory')
        assert hasattr(self.toolkit, 'task_similarity_cache')
        assert hasattr(self.toolkit, 'transfer_history')
        assert hasattr(self.toolkit, 'continual_learner')
        assert self.toolkit.knowledge_memory_size == 100
    
    def test_extract_task_features(self):
        """Test extracting features from tasks for similarity computation."""
        episode = Episode(
            support_x=torch.randn(25, 3, 84, 84),
            support_y=torch.randint(0, 5, (25,)),
            query_x=torch.randn(75, 3, 84, 84),
            query_y=torch.randint(0, 5, (75,))
        )
        
        features = self.toolkit.extract_task_features(episode, "task_123")
        
        assert isinstance(features, dict)
        assert 'task_id' in features
        assert 'n_classes' in features
        assert 'n_support' in features
        assert 'n_query' in features
        assert 'data_statistics' in features
        assert 'feature_representation' in features
        
        assert features['task_id'] == "task_123"
        assert features['n_classes'] == 5
        assert features['n_support'] == 25
        assert features['n_query'] == 75
        assert isinstance(features['feature_representation'], torch.Tensor)
    
    def test_compute_task_similarity(self):
        """Test computing similarity between tasks."""
        # Create two similar tasks
        episode1 = Episode(
            support_x=torch.randn(25, 3, 84, 84),
            support_y=torch.randint(0, 5, (25,)),
            query_x=torch.randn(75, 3, 84, 84),
            query_y=torch.randint(0, 5, (75,))
        )
        
        episode2 = Episode(
            support_x=torch.randn(25, 3, 84, 84),
            support_y=torch.randint(0, 5, (25,)),
            query_x=torch.randn(75, 3, 84, 84),
            query_y=torch.randint(0, 5, (75,))
        )
        
        features1 = self.toolkit.extract_task_features(episode1, "task_1")
        features2 = self.toolkit.extract_task_features(episode2, "task_2")
        
        similarity = self.toolkit.compute_task_similarity(features1, features2)
        
        assert isinstance(similarity, float)
        assert 0.0 <= similarity <= 1.0
    
    def test_store_task_knowledge(self):
        """Test storing knowledge from completed tasks."""
        episode = Episode(
            support_x=torch.randn(25, 3, 84, 84),
            support_y=torch.randint(0, 5, (25,)),
            query_x=torch.randn(75, 3, 84, 84),
            query_y=torch.randint(0, 5, (75,))
        )
        
        algorithm_params = {
            'learning_rate': 0.001,
            'n_inner_steps': 5
        }
        
        performance_metrics = {
            'accuracy': 0.85,
            'loss': 0.45,
            'training_time': 120.0
        }
        
        # Store task knowledge
        self.toolkit.store_task_knowledge(
            episode, 
            "test_task", 
            algorithm_params, 
            performance_metrics,
            model_state={"param1": torch.randn(10)}
        )
        
        # Check that knowledge is stored
        assert len(self.toolkit.knowledge_memory) == 1
        
        stored = self.toolkit.knowledge_memory[0]
        assert stored['task_id'] == "test_task"
        assert stored['algorithm_params'] == algorithm_params
        assert stored['performance_metrics'] == performance_metrics
        assert 'task_features' in stored
        assert 'model_state' in stored
        assert 'timestamp' in stored
    
    def test_retrieve_relevant_knowledge(self):
        """Test retrieving relevant knowledge for new tasks."""
        # Store knowledge for multiple tasks
        for i in range(10):
            episode = Episode(
                support_x=torch.randn(25, 3, 84, 84),
                support_y=torch.randint(0, 5, (25,)),
                query_x=torch.randn(75, 3, 84, 84),
                query_y=torch.randint(0, 5, (75,))
            )
            
            self.toolkit.store_task_knowledge(
                episode,
                f"stored_task_{i}",
                {'learning_rate': 0.001 + i * 0.0001},
                {'accuracy': 0.7 + i * 0.02}
            )
        
        # Create new task to find relevant knowledge for
        new_episode = Episode(
            support_x=torch.randn(25, 3, 84, 84),
            support_y=torch.randint(0, 5, (25,)),
            query_x=torch.randn(75, 3, 84, 84),
            query_y=torch.randint(0, 5, (75,))
        )
        
        relevant_knowledge = self.toolkit.retrieve_relevant_knowledge(
            new_episode, 
            "new_task",
            top_k=3
        )
        
        assert isinstance(relevant_knowledge, list)
        assert len(relevant_knowledge) <= 3
        
        # Each item should have similarity score
        for item in relevant_knowledge:
            assert 'knowledge' in item
            assert 'similarity' in item
            assert isinstance(item['similarity'], float)
            assert 0.0 <= item['similarity'] <= 1.0
        
        # Should be sorted by similarity (descending)
        if len(relevant_knowledge) > 1:
            for i in range(len(relevant_knowledge) - 1):
                assert relevant_knowledge[i]['similarity'] >= relevant_knowledge[i + 1]['similarity']
    
    def test_transfer_learning_initialization(self):
        """Test using transferred knowledge for model initialization."""
        # Store some knowledge first
        episode = Episode(
            support_x=torch.randn(25, 3, 84, 84),
            support_y=torch.randint(0, 5, (25,)),
            query_x=torch.randn(75, 3, 84, 84),
            query_y=torch.randint(0, 5, (75,))
        )
        
        model_state = {
            'embedding_layer': torch.randn(128, 256),
            'classifier_weights': torch.randn(5, 128)
        }
        
        self.toolkit.store_task_knowledge(
            episode,
            "source_task",
            {'learning_rate': 0.001},
            {'accuracy': 0.90},
            model_state=model_state
        )
        
        # Create new task
        new_episode = Episode(
            support_x=torch.randn(25, 3, 84, 84),
            support_y=torch.randint(0, 5, (25,)),
            query_x=torch.randn(75, 3, 84, 84),
            query_y=torch.randint(0, 5, (75,))
        )
        
        # Get transfer initialization
        init_params = self.toolkit.get_transfer_initialization(new_episode, "new_task")
        
        assert isinstance(init_params, dict)
        if len(self.toolkit.knowledge_memory) > 0:
            # Should return some initialization parameters
            assert len(init_params) > 0
    
    def test_continual_learning_update(self):
        """Test continual learning updates."""
        # Store initial knowledge
        for i in range(5):
            episode = Episode(
                support_x=torch.randn(25, 3, 84, 84),
                support_y=torch.randint(0, 5, (25,)),
                query_x=torch.randn(75, 3, 84, 84),
                query_y=torch.randint(0, 5, (75,))
            )
            
            self.toolkit.store_task_knowledge(
                episode,
                f"initial_task_{i}",
                {'learning_rate': 0.001},
                {'accuracy': 0.75 + i * 0.02}
            )
        
        # Perform continual learning update
        new_episode = Episode(
            support_x=torch.randn(25, 3, 84, 84),
            support_y=torch.randint(0, 5, (25,)),
            query_x=torch.randn(75, 3, 84, 84),
            query_y=torch.randint(0, 5, (75,))
        )
        
        update_result = self.toolkit.continual_learning_update(
            new_episode,
            "continual_task",
            {'accuracy': 0.88, 'loss': 0.35}
        )
        
        assert isinstance(update_result, dict)
        assert 'updated_knowledge' in update_result
        assert 'consolidation_loss' in update_result
        assert 'memory_efficiency' in update_result
    
    def test_memory_management(self):
        """Test memory management for knowledge storage."""
        # Fill memory beyond capacity
        for i in range(150):  # More than memory_size=100
            episode = Episode(
                support_x=torch.randn(25, 3, 84, 84),
                support_y=torch.randint(0, 5, (25,)),
                query_x=torch.randn(75, 3, 84, 84),
                query_y=torch.randint(0, 5, (75,))
            )
            
            self.toolkit.store_task_knowledge(
                episode,
                f"memory_task_{i}",
                {'learning_rate': 0.001},
                {'accuracy': 0.7 + (i % 20) * 0.01}  # Vary performance
            )
        
        # Memory should not exceed capacity
        assert len(self.toolkit.knowledge_memory) <= 100
        
        # Should keep high-performing tasks
        performances = [item['performance_metrics']['accuracy'] for item in self.toolkit.knowledge_memory]
        avg_performance = np.mean(performances)
        
        # Average performance should be decent (old low-performing tasks evicted)
        assert avg_performance > 0.75
    
    def test_knowledge_consolidation(self):
        """Test consolidating similar knowledge entries."""
        # Store multiple similar tasks
        base_episode = Episode(
            support_x=torch.randn(25, 3, 84, 84),
            support_y=torch.randint(0, 5, (25,)),
            query_x=torch.randn(75, 3, 84, 84),
            query_y=torch.randint(0, 5, (75,))
        )
        
        # Store very similar tasks
        for i in range(10):
            # Add small noise to create similar but not identical tasks
            noisy_episode = Episode(
                support_x=base_episode.support_x + torch.randn_like(base_episode.support_x) * 0.01,
                support_y=base_episode.support_y,
                query_x=base_episode.query_x + torch.randn_like(base_episode.query_x) * 0.01,
                query_y=base_episode.query_y
            )
            
            self.toolkit.store_task_knowledge(
                noisy_episode,
                f"similar_task_{i}",
                {'learning_rate': 0.001},
                {'accuracy': 0.85 + np.random.normal(0, 0.02)}
            )
        
        initial_memory_size = len(self.toolkit.knowledge_memory)
        
        # Perform consolidation
        consolidation_result = self.toolkit.consolidate_knowledge(similarity_threshold=0.9)
        
        assert isinstance(consolidation_result, dict)
        assert 'original_size' in consolidation_result
        assert 'consolidated_size' in consolidation_result
        assert 'consolidation_ratio' in consolidation_result
        
        # Memory size should be reduced after consolidation
        final_memory_size = len(self.toolkit.knowledge_memory)
        assert final_memory_size <= initial_memory_size


class TestIntegration:
    """Integration tests for Phase 4 features working together."""
    
    def setup_method(self):
        """Set up toolkit with all Phase 4 features enabled."""
        self.toolkit = MetaLearningToolkit()
        
        # Enable all Phase 4 features
        self.toolkit.enable_failure_prediction(True, True)
        self.toolkit.enable_automatic_algorithm_selection(True, "maml")
        self.toolkit.enable_realtime_optimization(True, 25)
        self.toolkit.enable_cross_task_knowledge_transfer(True, 50)
    
    def test_full_workflow_integration(self):
        """Test full workflow with all Phase 4 features working together."""
        # Create sample episode
        episode = Episode(
            support_x=torch.randn(25, 3, 84, 84),
            support_y=torch.randint(0, 5, (25,)),
            query_x=torch.randn(75, 3, 84, 84),
            query_y=torch.randint(0, 5, (75,))
        )
        
        task_id = "integration_test_task"
        
        # Step 1: Analyze data characteristics
        characteristics = self.toolkit.analyze_data_characteristics(episode)
        assert isinstance(characteristics, dict)
        
        # Step 2: Get algorithm recommendation
        algorithm_rec = self.toolkit.recommend_algorithm(characteristics)
        assert isinstance(algorithm_rec, dict)
        recommended_algo = algorithm_rec['algorithm']
        
        # Step 3: Get transfer learning initialization
        transfer_init = self.toolkit.get_transfer_initialization(episode, task_id)
        assert isinstance(transfer_init, dict)
        
        # Step 4: Get optimized parameters from A/B testing
        optimized_params = self.toolkit.get_current_best_params()
        assert isinstance(optimized_params, dict)
        
        # Step 5: Predict failure probability
        failure_features = self.toolkit.collect_failure_features(episode, optimized_params)
        failure_prob = self.toolkit.predict_failure_probability(failure_features)
        assert 0.0 <= failure_prob <= 1.0
        
        # Step 6: Apply recovery strategy if needed
        if self.toolkit.should_trigger_recovery(failure_features):
            recovery_strategy = self.toolkit.get_recovery_strategy(failure_features)
            optimized_params = self.toolkit.apply_recovery_strategy(optimized_params, recovery_strategy)
        
        # Step 7: Record outcomes and update all systems
        performance_metrics = {
            'accuracy': 0.87,
            'loss': 0.42,
            'training_time': 105.0
        }
        
        # Record for failure prediction
        self.toolkit.record_failure_outcome(
            failure_features, 
            failed=False, 
            performance=performance_metrics['accuracy']
        )
        
        # Record for algorithm selection
        self.toolkit.record_algorithm_performance(
            recommended_algo,
            characteristics,
            performance_metrics['accuracy'],
            performance_metrics['training_time']
        )
        
        # Store knowledge for transfer learning
        self.toolkit.store_task_knowledge(
            episode,
            task_id,
            optimized_params,
            performance_metrics
        )
        
        # All steps should complete without errors
        assert True
    
    def test_cross_feature_interactions(self):
        """Test interactions between different Phase 4 features."""
        # Test that failure prediction influences A/B testing
        # High failure probability should bias towards safer parameter choices
        
        # Test that algorithm selection uses transfer learning knowledge
        # Similar tasks should influence algorithm recommendations
        
        # Test that A/B testing results influence failure prediction
        # Better performing parameters should have lower failure predictions
        
        # This is a placeholder for more sophisticated interaction testing
        assert hasattr(self.toolkit, 'failure_predictor')
        assert hasattr(self.toolkit, 'algorithm_selector')
        assert hasattr(self.toolkit, 'ab_test_manager')
        assert hasattr(self.toolkit, 'knowledge_memory')
    
    def test_performance_with_all_features(self):
        """Test that all features together don't significantly slow down the system."""
        import time
        
        start_time = time.time()
        
        # Run several operations that use multiple features
        for i in range(10):
            episode = Episode(
                support_x=torch.randn(25, 3, 84, 84),
                support_y=torch.randint(0, 5, (25,)),
                query_x=torch.randn(75, 3, 84, 84),
                query_y=torch.randint(0, 5, (75,))
            )
            
            # Use multiple features
            characteristics = self.toolkit.analyze_data_characteristics(episode)
            algorithm_rec = self.toolkit.recommend_algorithm(characteristics)
            transfer_init = self.toolkit.get_transfer_initialization(episode, f"perf_task_{i}")
            failure_features = self.toolkit.collect_failure_features(episode, {})
            failure_prob = self.toolkit.predict_failure_probability(failure_features)
        
        elapsed_time = time.time() - start_time
        
        # Should complete reasonably quickly (less than 10 seconds for 10 iterations)
        assert elapsed_time < 10.0


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])