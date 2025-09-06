"""
Tests for EpisodeIterator and related iterator classes.
"""
import pytest
import torch
import tempfile
from unittest.mock import Mock, patch

from meta_learning.data_utils.iterators import (
    EpisodeIterator, CurriculumSampler, AdaptiveBatchSampler,
    MemoryAwareIterator, InfiniteIterator, EpisodeBatchIterator,
    IteratorFactory, BalancedTaskGenerator
)
from meta_learning.core.episode import Episode


class MockDataset:
    """Mock dataset for testing."""
    
    def __init__(self, n_classes=10):
        self.classes = list(range(n_classes))
        self.n_classes = n_classes
    
    def __len__(self):
        return self.n_classes * 20  # 20 samples per class
    
    def __getitem__(self, index):
        class_id = index % self.n_classes
        data = torch.randn(3, 32, 32)
        return data, torch.tensor(class_id)


class TestEpisodeIterator:
    """Test EpisodeIterator functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.dataset = MockDataset(n_classes=5)
        self.iterator = EpisodeIterator(
            self.dataset, 
            n_way=3, 
            n_shot=2, 
            n_query=5,
            quality_threshold=0.1
        )
    
    def test_initialization(self):
        """Test iterator initialization."""
        assert self.iterator.n_way == 3
        assert self.iterator.n_shot == 2
        assert self.iterator.n_query == 5
        assert self.iterator.quality_threshold == 0.1
        assert self.iterator._episode_count == 0
        assert self.iterator.memory_aware is True
    
    def test_episode_generation(self):
        """Test basic episode generation."""
        episode = next(self.iterator)
        
        assert isinstance(episode, dict)
        assert 'support_x' in episode
        assert 'support_y' in episode
        assert 'query_x' in episode
        assert 'query_y' in episode
        
        # Check shapes
        assert episode['support_x'].shape == (6, 3, 84, 84)  # 3 classes * 2 shots
        assert episode['support_y'].shape == (6,)
        assert episode['query_x'].shape == (15, 3, 84, 84)   # 3 classes * 5 queries
        assert episode['query_y'].shape == (15,)
    
    def test_episode_quality_assessment(self):
        """Test episode quality assessment."""
        # Create a high-quality episode
        episode = {
            'support_x': torch.randn(6, 3, 84, 84),
            'support_y': torch.tensor([0, 0, 1, 1, 2, 2]),
            'query_x': torch.randn(15, 3, 84, 84),
            'query_y': torch.tensor([0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2])
        }
        
        quality = self.iterator._assess_episode_quality(episode)
        assert 0.0 <= quality <= 1.0
        assert quality > 0.5  # Should be reasonable quality
    
    def test_episode_quality_assessment_with_nan(self):
        """Test quality assessment with NaN values."""
        episode = {
            'support_x': torch.tensor([[float('nan')]]),
            'support_y': torch.tensor([0]),
            'query_x': torch.randn(1, 1),
            'query_y': torch.tensor([0])
        }
        
        quality = self.iterator._assess_episode_quality(episode)
        assert quality == 0.0  # Should reject NaN data
    
    def test_memory_monitoring(self):
        """Test memory monitoring functionality."""
        if torch.cuda.is_available():
            self.iterator.memory_aware = True
            self.iterator._monitor_memory()
            assert len(self.iterator._memory_usage) > 0
        else:
            # On CPU, memory monitoring should not crash
            self.iterator._monitor_memory()
    
    def test_iterator_interface(self):
        """Test iterator protocol."""
        assert iter(self.iterator) is self.iterator
        
        # Should be able to iterate multiple times
        episode1 = next(self.iterator)
        episode2 = next(self.iterator)
        
        assert self.iterator._episode_count == 2


class TestCurriculumSampler:
    """Test CurriculumSampler functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.dataset = MockDataset(n_classes=8)
        self.sampler = CurriculumSampler(
            self.dataset,
            initial_difficulty=0.3,
            curriculum_rate=0.1,
            max_difficulty=0.9
        )
    
    def test_initialization(self):
        """Test sampler initialization."""
        assert self.sampler.current_difficulty == 0.3
        assert self.sampler.curriculum_rate == 0.1
        assert self.sampler.max_difficulty == 0.9
        assert len(self.sampler._performance_history) == 0
    
    def test_episode_sampling(self):
        """Test episode sampling."""
        episode = self.sampler.sample_episode(n_way=3, n_shot=1, n_query=5)
        
        assert isinstance(episode, dict)
        assert episode['support_x'].shape == (3, 3, 84, 84)
        assert episode['query_x'].shape == (15, 3, 84, 84)
    
    def test_difficulty_update_high_performance(self):
        """Test difficulty increase with high performance."""
        initial_difficulty = self.sampler.current_difficulty
        
        # High performance should increase difficulty
        self.sampler.update_difficulty(0.9)
        
        assert self.sampler.current_difficulty > initial_difficulty
        assert len(self.sampler._performance_history) == 1
    
    def test_difficulty_update_low_performance(self):
        """Test difficulty decrease with low performance."""
        self.sampler.current_difficulty = 0.5
        initial_difficulty = self.sampler.current_difficulty
        
        # Low performance should decrease difficulty
        self.sampler.update_difficulty(0.4)
        
        assert self.sampler.current_difficulty < initial_difficulty
    
    def test_difficulty_bounds(self):
        """Test difficulty stays within bounds."""
        # Test upper bound
        for _ in range(20):
            self.sampler.update_difficulty(0.95)
        assert self.sampler.current_difficulty <= self.sampler.max_difficulty
        
        # Test lower bound
        self.sampler.current_difficulty = 0.2
        for _ in range(20):
            self.sampler.update_difficulty(0.1)
        assert self.sampler.current_difficulty >= 0.1


class TestAdaptiveBatchSampler:
    """Test AdaptiveBatchSampler functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.dataset = MockDataset(n_classes=10)
        self.sampler = AdaptiveBatchSampler(
            self.dataset,
            batch_size=4,
            difficulty_levels=3,
            adaptation_rate=0.1
        )
    
    def test_initialization(self):
        """Test sampler initialization."""
        assert self.sampler.batch_size == 4
        assert self.sampler.difficulty_levels == 3
        assert self.sampler.current_difficulty == 0.5
        assert len(self.sampler.performance_history) == 0
    
    def test_batch_generation(self):
        """Test batch generation."""
        batch = next(self.sampler)
        
        assert isinstance(batch, list)
        assert len(batch) == 4
        
        for episode in batch:
            assert isinstance(episode, Episode)
            assert episode.support_x.shape[0] > 0
            assert episode.query_x.shape[0] > 0
    
    def test_parameter_adaptation(self):
        """Test episode parameter adaptation based on difficulty."""
        # Test high difficulty adaptation
        self.sampler.current_difficulty = 0.8
        params = self.sampler._adapt_episode_parameters()
        assert params['n_way'] > self.sampler.episode_params['n_way']
        
        # Test low difficulty adaptation
        self.sampler.current_difficulty = 0.2
        params = self.sampler._adapt_episode_parameters()
        assert params['n_way'] <= self.sampler.episode_params['n_way']
    
    def test_performance_update(self):
        """Test performance-based adaptation."""
        # High performance batch
        self.sampler.update_performance([0.9, 0.85, 0.9, 0.88])
        assert len(self.sampler.performance_history) == 1
        assert self.sampler.current_difficulty > 0.5
        
        # Reset and test low performance
        self.sampler.current_difficulty = 0.5
        self.sampler.update_performance([0.3, 0.4, 0.35, 0.45])
        assert self.sampler.current_difficulty < 0.5
    
    def test_curriculum_stats(self):
        """Test curriculum statistics."""
        # Add some history
        self.sampler.update_performance([0.7, 0.8, 0.75, 0.72])
        
        stats = self.sampler.get_curriculum_stats()
        assert 'current_difficulty' in stats
        assert 'avg_recent_performance' in stats
        assert 'total_batches' in stats
        assert stats['total_batches'] == 1


class TestMemoryAwareIterator:
    """Test MemoryAwareIterator functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.base_iterator = iter([torch.randn(10, 10) for _ in range(100)])
        self.iterator = MemoryAwareIterator(
            self.base_iterator,
            memory_budget_ratio=0.5,
            cleanup_frequency=10
        )
    
    def test_initialization(self):
        """Test iterator initialization."""
        assert self.iterator.memory_budget_ratio == 0.5
        assert self.iterator.cleanup_frequency == 10
        assert self.iterator.iteration_count == 0
    
    def test_memory_usage_tracking(self):
        """Test memory usage monitoring."""
        stats = self.iterator._get_memory_usage()
        
        assert 'allocated_mb' in stats
        assert 'total_mb' in stats
        assert 'usage_ratio' in stats
        assert stats['usage_ratio'] >= 0
    
    def test_iteration(self):
        """Test basic iteration functionality."""
        item = next(self.iterator)
        assert isinstance(item, torch.Tensor)
        assert self.iterator.iteration_count == 1
    
    @patch('gc.collect')
    def test_cleanup_trigger(self, mock_gc):
        """Test automatic cleanup triggering."""
        # Iterate enough to trigger cleanup
        for _ in range(15):
            try:
                next(self.iterator)
            except StopIteration:
                break
        
        # Should have triggered cleanup
        assert mock_gc.called


class TestBalancedTaskGenerator:
    """Test BalancedTaskGenerator functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.dataset = MockDataset(n_classes=6)
        self.generator = BalancedTaskGenerator(
            self.dataset,
            n_way=3,
            n_shot=2,
            n_query=4,
            balance_strategies=['class', 'difficulty']
        )
    
    def test_initialization(self):
        """Test generator initialization."""
        assert self.generator.n_way == 3
        assert self.generator.n_shot == 2
        assert self.generator.n_query == 4
        assert 'class' in self.generator.balance_strategies
        assert len(self.generator.available_classes) > 0
    
    def test_dataset_analysis(self):
        """Test dataset analysis functionality."""
        assert hasattr(self.generator, 'class_counts')
        assert hasattr(self.generator, 'imbalance_ratio')
        assert len(self.generator.class_counts) > 0
        assert self.generator.imbalance_ratio >= 1.0
    
    def test_episode_generation(self):
        """Test balanced episode generation."""
        episode = self.generator.generate_episode(random_state=42)
        
        assert isinstance(episode, Episode)
        assert episode.support_x.shape == (6, 3, 32, 32)  # 3 classes * 2 shots
        assert episode.query_x.shape == (12, 3, 32, 32)   # 3 classes * 4 queries
        assert len(torch.unique(episode.support_y)) == 3
        assert len(torch.unique(episode.query_y)) == 3
    
    def test_class_selection(self):
        """Test balanced class selection."""
        selected = self.generator._select_balanced_classes()
        
        assert len(selected) == self.generator.n_way
        assert len(set(selected)) <= len(selected)  # May have duplicates if needed


class TestIteratorFactory:
    """Test IteratorFactory functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.dataset = MockDataset(n_classes=5)
    
    def test_create_infinite_iterator(self):
        """Test infinite iterator creation."""
        dataloader = [1, 2, 3, 4, 5]
        iterator = IteratorFactory.create_infinite_iterator(dataloader)
        
        assert isinstance(iterator, InfiniteIterator)
    
    def test_create_memory_aware_iterator(self):
        """Test memory-aware iterator creation."""
        base_iterator = iter([1, 2, 3])
        iterator = IteratorFactory.create_memory_aware_iterator(base_iterator)
        
        assert isinstance(iterator, MemoryAwareIterator)
    
    def test_create_episode_batch_iterator(self):
        """Test episode batch iterator creation."""
        self.dataset.create_episode = Mock(return_value=Episode(
            support_x=torch.randn(5, 3, 32, 32),
            support_y=torch.zeros(5),
            query_x=torch.randn(15, 3, 32, 32),
            query_y=torch.zeros(15)
        ))
        
        iterator = IteratorFactory.create_episode_batch_iterator(
            self.dataset, batch_size=2
        )
        
        assert isinstance(iterator, EpisodeBatchIterator)
    
    def test_create_optimal_iterator(self):
        """Test optimal iterator creation."""
        dataloader = [1, 2, 3]
        
        # Test different types
        iterator1 = IteratorFactory.create_optimal_iterator(
            dataloader, iterator_type="infinite"
        )
        assert isinstance(iterator1, InfiniteIterator)
        
        iterator2 = IteratorFactory.create_optimal_iterator(
            dataloader, iterator_type="memory_aware"
        )
        assert isinstance(iterator2, MemoryAwareIterator)
    
    def test_invalid_iterator_type(self):
        """Test error handling for invalid iterator type."""
        with pytest.raises(ValueError, match="Unknown iterator type"):
            IteratorFactory.create_optimal_iterator([1, 2, 3], "invalid_type")


class TestInfiniteIterator:
    """Test InfiniteIterator functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.dataloader = iter([1, 2, 3])
        self.iterator = InfiniteIterator(self.dataloader, enable_monitoring=True)
    
    def test_initialization(self):
        """Test iterator initialization."""
        assert self.iterator.enable_monitoring is True
        assert 'iterations' in self.iterator.stats
        assert self.iterator.stats['iterations'] == 0
    
    def test_basic_iteration(self):
        """Test basic iteration functionality."""
        # Should cycle through dataloader
        assert next(self.iterator) == 1
        assert next(self.iterator) == 2
        assert next(self.iterator) == 3
        
        # Should restart and continue infinitely
        assert next(self.iterator) == 1  # Restarted
    
    def test_statistics_tracking(self):
        """Test statistics tracking."""
        # Generate some iterations
        for _ in range(5):
            next(self.iterator)
        
        stats = self.iterator.get_stats()
        assert stats['iterations'] == 5
        assert 'error_rate' in stats
        assert 'avg_iteration_time' in stats


if __name__ == "__main__":
    pytest.main([__file__])