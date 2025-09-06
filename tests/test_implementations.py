#!/usr/bin/env python3
"""
Test suite for new meta-learning implementations.

Tests the recently implemented features:
- MemoryAwareIterator
- BalancedTaskGenerator  
- PerformanceIterator
- Enhanced evaluation metrics
- Data utilities
"""

import pytest
import torch
import torch.nn as nn
from typing import List, Dict, Any

# Test imports
try:
    from meta_learning.core.episode import Episode
    from meta_learning.data_utils import (
        MemoryAwareIterator, 
        BalancedTaskGenerator,
        PerformanceIterator,
        create_episode_from_data,
        compute_episode_statistics,
        split_episode
    )
    from meta_learning.evaluation.metrics import (
        AccuracyCalculator,
        CalibrationCalculator, 
        UncertaintyCalculator,
        compute_comprehensive_metrics
    )
    from meta_learning.evaluation.prototype_analysis import (
        PrototypeAnalyzer,
        analyze_episode_quality
    )
except ImportError as e:
    pytest.skip(f"Skipping tests due to import error: {e}", allow_module_level=True)


class TestMemoryAwareIterator:
    """Test MemoryAwareIterator functionality."""
    
    def create_dummy_iterator(self, n_items=10):
        """Create dummy iterator for testing."""
        class DummyIterator:
            def __init__(self, items):
                self.items = items
                self.index = 0
            
            def __iter__(self):
                return self
            
            def __next__(self):
                if self.index >= len(self.items):
                    raise StopIteration
                item = self.items[self.index]
                self.index += 1
                return item
        
        return DummyIterator(list(range(n_items)))
    
    def test_memory_iterator_basic_functionality(self):
        """Test basic iteration functionality."""
        base_iterator = self.create_dummy_iterator(5)
        memory_iterator = MemoryAwareIterator(
            base_iterator, 
            memory_budget=0.8,
            enable_profiling=False  # Disable for testing
        )
        
        items = []
        for item in memory_iterator:
            items.append(item)
        
        assert items == [0, 1, 2, 3, 4]
    
    def test_memory_stats_collection(self):
        """Test memory statistics collection."""
        base_iterator = self.create_dummy_iterator(3)
        memory_iterator = MemoryAwareIterator(
            base_iterator,
            memory_budget=0.8,
            enable_profiling=True
        )
        
        # Consume iterator
        list(memory_iterator)
        
        stats = memory_iterator.get_memory_stats()
        assert 'iteration_count' in stats
        assert stats['iteration_count'] == 3
        assert 'memory_budget' in stats
        assert stats['memory_budget'] == 0.8


class TestBalancedTaskGenerator:
    """Test BalancedTaskGenerator functionality."""
    
    def create_dummy_dataset(self, n_classes=10, samples_per_class=20):
        """Create dummy dataset for testing."""
        class DummyDataset:
            def __init__(self, n_classes, samples_per_class):
                self.n_classes = n_classes
                self.samples_per_class = samples_per_class
                self.data = []
                self.labels = []
                
                for class_id in range(n_classes):
                    for sample_id in range(samples_per_class):
                        # Create dummy data
                        data = torch.randn(3, 32, 32)  # RGB image
                        self.data.append(data)
                        self.labels.append(class_id)
            
            def __len__(self):
                return len(self.data)
            
            def __getitem__(self, idx):
                return self.data[idx], self.labels[idx]
        
        return DummyDataset(n_classes, samples_per_class)
    
    def test_balanced_generator_creation(self):
        """Test basic BalancedTaskGenerator creation."""
        dataset = self.create_dummy_dataset(5, 10)
        generator = BalancedTaskGenerator(
            dataset,
            balancing_strategy="frequency",
            enable_analytics=False  # Disable for testing
        )
        
        assert generator.balancing_strategy == "frequency"
        assert generator._get_num_classes() == 5
    
    def test_episode_generation(self):
        """Test episode generation.""" 
        dataset = self.create_dummy_dataset(10, 20)
        generator = BalancedTaskGenerator(
            dataset,
            balancing_strategy="random", 
            enable_analytics=False
        )
        
        episode_data = generator.generate_episode(n_way=5, k_shot=1, k_query=15)
        
        assert 'support_x' in episode_data
        assert 'support_y' in episode_data
        assert 'query_x' in episode_data
        assert 'query_y' in episode_data
        
        # Check shapes
        assert episode_data['support_x'].shape[0] == 5  # 5 way * 1 shot
        assert episode_data['query_x'].shape[0] == 75   # 5 way * 15 query
        
        # Check class distribution
        assert len(torch.unique(episode_data['support_y'])) == 5
        assert len(torch.unique(episode_data['query_y'])) == 5


class TestPerformanceIterator:
    """Test PerformanceIterator functionality."""
    
    def create_slow_iterator(self, n_items=5, delay=0.01):
        """Create iterator with artificial delays."""
        import time
        
        class SlowIterator:
            def __init__(self, items, delay):
                self.items = items
                self.delay = delay
                self.index = 0
            
            def __iter__(self):
                return self
            
            def __next__(self):
                if self.index >= len(self.items):
                    raise StopIteration
                
                time.sleep(self.delay)  # Simulate processing time
                item = self.items[self.index]
                self.index += 1
                return item
        
        return SlowIterator(list(range(n_items)), delay)
    
    def test_performance_monitoring(self):
        """Test performance monitoring functionality."""
        base_iterator = self.create_slow_iterator(3, 0.001)  # Very small delay
        perf_iterator = PerformanceIterator(
            base_iterator,
            optimization_level="balanced",
            profiling_interval=2
        )
        
        # Consume iterator
        items = list(perf_iterator)
        
        assert items == [0, 1, 2]
        assert perf_iterator.iteration_count == 3
        
        # Check performance report
        report = perf_iterator.get_performance_report()
        assert 'iteration_count' in report
        assert 'optimization_level' in report


class TestEvaluationMetrics:
    """Test evaluation metrics functionality."""
    
    def create_dummy_predictions_and_targets(self):
        """Create dummy predictions and targets."""
        # 5-way classification, 15 query samples
        predictions = torch.randn(15, 5)  # Logits
        targets = torch.randint(0, 5, (15,))  # Random targets
        return predictions, targets
    
    def test_accuracy_calculator(self):
        """Test AccuracyCalculator functionality."""
        predictions, targets = self.create_dummy_predictions_and_targets()
        
        # Test accuracy computation
        accuracy = AccuracyCalculator.compute_accuracy(predictions, targets)
        assert 0.0 <= accuracy <= 1.0
        
        # Test per-class accuracy
        per_class_acc = AccuracyCalculator.compute_per_class_accuracy(predictions, targets)
        assert isinstance(per_class_acc, dict)
        assert all(0.0 <= acc <= 1.0 for acc in per_class_acc.values())
    
    def test_comprehensive_metrics(self):
        """Test comprehensive metrics computation."""
        predictions, targets = self.create_dummy_predictions_and_targets()
        
        metrics = compute_comprehensive_metrics(predictions, targets)
        
        assert hasattr(metrics, 'accuracy')
        assert 0.0 <= metrics.accuracy <= 1.0
        assert hasattr(metrics, 'per_class_accuracy')
        assert isinstance(metrics.per_class_accuracy, dict)


class TestPrototypeAnalysis:
    """Test prototype analysis functionality."""
    
    def create_dummy_episode_data(self):
        """Create dummy episode data for prototype analysis."""
        # 3-way, 5-shot, 15-query episode
        support_features = torch.randn(15, 64)  # 3 classes * 5 shots
        support_labels = torch.repeat_interleave(torch.arange(3), 5)
        
        query_features = torch.randn(45, 64)  # 3 classes * 15 queries  
        query_labels = torch.repeat_interleave(torch.arange(3), 15)
        
        # Create prototypes (class means)
        prototypes = []
        for class_id in range(3):
            class_mask = support_labels == class_id
            prototype = support_features[class_mask].mean(dim=0)
            prototypes.append(prototype)
        prototypes = torch.stack(prototypes)
        
        return support_features, support_labels, query_features, prototypes
    
    def test_prototype_quality_analysis(self):
        """Test prototype quality analysis."""
        support_features, support_labels, query_features, prototypes = self.create_dummy_episode_data()
        
        analyzer = PrototypeAnalyzer()
        metrics = analyzer.analyze_prototype_quality(
            support_features, support_labels, prototypes
        )
        
        assert hasattr(metrics, 'intra_class_variance')
        assert hasattr(metrics, 'inter_class_distance')
        assert hasattr(metrics, 'silhouette_score')
        assert hasattr(metrics, 'prototype_coherence')
        assert hasattr(metrics, 'class_separation_ratio')
        
        # Check metric ranges
        assert metrics.intra_class_variance >= 0.0
        assert metrics.inter_class_distance >= 0.0
        assert -1.0 <= metrics.silhouette_score <= 1.0
        assert 0.0 <= metrics.prototype_coherence <= 1.0
    
    def test_episode_quality_analysis(self):
        """Test comprehensive episode quality analysis."""
        support_features, support_labels, query_features, prototypes = self.create_dummy_episode_data()
        
        quality_metrics = analyze_episode_quality(
            support_features, support_labels, query_features, prototypes
        )
        
        assert isinstance(quality_metrics, dict)
        assert 'intra_class_variance' in quality_metrics
        assert 'difficulty_score' in quality_metrics
        assert 'max_inter_class_similarity' in quality_metrics


class TestDataUtilities:
    """Test data utility functions."""
    
    def create_dummy_episode(self):
        """Create dummy episode for testing."""
        support_x = torch.randn(10, 3, 32, 32)  # 2-way, 5-shot
        support_y = torch.repeat_interleave(torch.arange(2), 5)
        query_x = torch.randn(20, 3, 32, 32)    # 2-way, 10-query
        query_y = torch.repeat_interleave(torch.arange(2), 10)
        
        return Episode(support_x, support_y, query_x, query_y)
    
    def test_episode_creation_from_data(self):
        """Test creating episode from raw data."""
        support_x = torch.randn(5, 10)
        support_y = torch.randint(0, 2, (5,))
        query_x = torch.randn(10, 10)
        query_y = torch.randint(0, 2, (10,))
        
        episode = create_episode_from_data(support_x, support_y, query_x, query_y)
        
        assert torch.equal(episode.support_x, support_x)
        assert torch.equal(episode.support_y, support_y)
        assert torch.equal(episode.query_x, query_x)
        assert torch.equal(episode.query_y, query_y)
    
    def test_episode_statistics(self):
        """Test episode statistics computation."""
        episode = self.create_dummy_episode()
        stats = compute_episode_statistics(episode)
        
        assert stats['n_support'] == 10
        assert stats['n_query'] == 20
        assert stats['n_support_classes'] == 2
        assert stats['n_query_classes'] == 2
        assert len(stats['data_shape']) == 3  # [3, 32, 32]
    
    def test_episode_splitting(self):
        """Test episode splitting functionality."""
        episode = self.create_dummy_episode()
        
        episode1, episode2 = split_episode(episode, query_ratio=0.5)
        
        # Check that both episodes have same support set
        assert torch.equal(episode1.support_x, episode.support_x)
        assert torch.equal(episode2.support_x, episode.support_x)
        assert torch.equal(episode1.support_y, episode.support_y)
        assert torch.equal(episode2.support_y, episode.support_y)
        
        # Check that query sets are split
        assert len(episode1.query_x) + len(episode2.query_x) == len(episode.query_x)
        assert len(episode1.query_y) + len(episode2.query_y) == len(episode.query_y)


# Integration test
class TestIntegration:
    """Integration tests for multiple components."""
    
    def test_end_to_end_episode_processing(self):
        """Test end-to-end episode processing with multiple components."""
        
        # Create episode
        support_x = torch.randn(15, 64)  # 3-way, 5-shot
        support_y = torch.repeat_interleave(torch.arange(3), 5)
        query_x = torch.randn(30, 64)    # 3-way, 10-query
        query_y = torch.repeat_interleave(torch.arange(3), 10)
        
        episode = create_episode_from_data(support_x, support_y, query_x, query_y)
        
        # Analyze episode quality
        quality_metrics = analyze_episode_quality(
            episode.support_x, episode.support_y, episode.query_x
        )
        
        # Generate dummy predictions
        predictions = torch.randn(30, 3)
        
        # Compute evaluation metrics
        eval_metrics = compute_comprehensive_metrics(predictions, episode.query_y)
        
        # Check that everything works together
        assert isinstance(quality_metrics, dict)
        assert hasattr(eval_metrics, 'accuracy')
        assert 'difficulty_score' in quality_metrics
        assert 0.0 <= eval_metrics.accuracy <= 1.0


if __name__ == '__main__':
    # Run tests
    pytest.main([__file__, '-v'])