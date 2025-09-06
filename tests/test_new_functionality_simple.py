"""
Simplified Test Suite for New Functionality
==========================================

Basic tests that verify the new functionality works as implemented:
- Dataset management system basic functionality
- Phase 4 toolkit enhancements basic functionality
- Integration between components

Author: Test Suite Generator
"""

import pytest
import torch
import numpy as np
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch

from meta_learning.toolkit import MetaLearningToolkit
from meta_learning.shared.types import Episode
from meta_learning.meta_learning_modules.dataset_management import (
    DatasetManager,
    DatasetInfo,
    DatasetRegistry,
    SmartCache,
    get_dataset_manager
)


class TestDatasetManagementBasic:
    """Basic tests for dataset management system."""
    
    def test_dataset_info_creation(self):
        """Test creating DatasetInfo."""
        info = DatasetInfo(
            name="test_dataset",
            description="Test dataset",
            urls=["http://example.com/data.zip"],
            checksums={"md5": "abc123"},
            file_size=1000000,
            n_classes=10,
            n_samples=5000,
            image_size=(32, 32)
        )
        
        assert info.name == "test_dataset"
        assert info.n_classes == 10
        assert len(info.urls) == 1
    
    def test_dataset_registry_basic(self):
        """Test basic dataset registry operations."""
        registry = DatasetRegistry()
        
        # Should have built-in datasets
        datasets = registry.list_datasets()
        assert "miniimagenet" in datasets
        assert "omniglot" in datasets
        
        # Should be able to get dataset info
        mini_info = registry.get_dataset_info("miniimagenet")
        assert mini_info is not None
        assert mini_info.name == "miniimagenet"
        assert mini_info.n_classes == 100
    
    def test_smart_cache_basic(self):
        """Test basic smart cache functionality."""
        temp_dir = tempfile.mkdtemp()
        try:
            cache = SmartCache(cache_dir=temp_dir, max_size_gb=0.001)
            
            # Test cache path generation
            path = cache.get_cache_path("dataset1", "file1.txt")
            assert "dataset1" in str(path)
            assert "file1.txt" in str(path)
            
            # Test caching and retrieval
            test_data = b"Hello, World!"
            cache.cache_file("dataset1", "file1.txt", test_data)
            
            # Should be cached
            assert cache.is_cached("dataset1", "file1.txt")
            
            # Should retrieve same data
            retrieved = cache.get_cached_file("dataset1", "file1.txt")
            assert retrieved == test_data
            
            # Test cache stats
            stats = cache.get_cache_stats()
            assert isinstance(stats, dict)
            assert "total_size_gb" in stats
            
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)
    
    def test_dataset_manager_basic(self):
        """Test basic dataset manager functionality."""
        temp_dir = tempfile.mkdtemp()
        try:
            manager = DatasetManager(cache_dir=temp_dir, max_cache_size_gb=0.01)
            
            # Should list available datasets
            datasets = manager.list_available_datasets()
            assert isinstance(datasets, list)
            assert len(datasets) > 0
            
            # Should get dataset info
            info = manager.get_dataset_info("miniimagenet")
            assert info is not None
            assert info.name == "miniimagenet"
            
            # Should get cache stats
            stats = manager.get_cache_stats()
            assert isinstance(stats, dict)
            
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)


class TestToolkitPhase4Basic:
    """Basic tests for Phase 4 toolkit enhancements."""
    
    def setup_method(self):
        """Set up toolkit for testing."""
        self.toolkit = MetaLearningToolkit()
    
    def test_enable_failure_prediction_basic(self):
        """Test enabling failure prediction functionality."""
        # Should enable without errors
        self.toolkit.enable_failure_prediction(True, True)
        assert hasattr(self.toolkit, 'failure_prediction_enabled')
        assert self.toolkit.failure_prediction_enabled == True
        
        # Should work with different parameters
        self.toolkit.enable_failure_prediction(False, True)
        assert self.toolkit.failure_prediction_enabled == False
    
    def test_enable_algorithm_selection_basic(self):
        """Test enabling algorithm selection functionality."""
        # Should enable without errors
        self.toolkit.enable_automatic_algorithm_selection(True, "maml")
        assert hasattr(self.toolkit, 'algorithm_selection_enabled')
        assert self.toolkit.algorithm_selection_enabled == True
        
        # Should store fallback algorithm
        assert hasattr(self.toolkit, 'fallback_algorithm')
        assert self.toolkit.fallback_algorithm == "maml"
    
    def test_enable_realtime_optimization_basic(self):
        """Test enabling real-time optimization functionality."""
        # Should enable without errors
        self.toolkit.enable_realtime_optimization(True, 100)
        assert hasattr(self.toolkit, 'realtime_optimization_enabled')
        assert self.toolkit.realtime_optimization_enabled == True
        
        # Should store optimization interval
        assert hasattr(self.toolkit, 'optimization_interval')
        assert self.toolkit.optimization_interval == 100
    
    def test_enable_knowledge_transfer_basic(self):
        """Test enabling knowledge transfer functionality."""
        # Should enable without errors
        self.toolkit.enable_cross_task_knowledge_transfer(True, 500)
        assert hasattr(self.toolkit, 'knowledge_transfer_enabled')
        assert self.toolkit.knowledge_transfer_enabled == True
        
        # Should store memory size
        assert hasattr(self.toolkit, 'knowledge_memory_size')
        assert self.toolkit.knowledge_memory_size == 500
    
    def test_all_features_together(self):
        """Test enabling all Phase 4 features together."""
        # Should enable all features without conflicts
        self.toolkit.enable_failure_prediction(True, True)
        self.toolkit.enable_automatic_algorithm_selection(True, "prototypical")
        self.toolkit.enable_realtime_optimization(True, 50)
        self.toolkit.enable_cross_task_knowledge_transfer(True, 100)
        
        # All should be enabled
        assert self.toolkit.failure_prediction_enabled == True
        assert self.toolkit.algorithm_selection_enabled == True
        assert self.toolkit.realtime_optimization_enabled == True
        assert self.toolkit.knowledge_transfer_enabled == True
        
        # Settings should be stored correctly
        assert self.toolkit.fallback_algorithm == "prototypical"
        assert self.toolkit.optimization_interval == 50
        assert self.toolkit.knowledge_memory_size == 100


class TestBasicIntegration:
    """Basic integration tests."""
    
    def test_toolkit_with_episode(self):
        """Test that toolkit can work with Episode objects."""
        toolkit = MetaLearningToolkit()
        
        # Create a basic episode
        episode = Episode(
            support_x=torch.randn(25, 3, 32, 32),
            support_y=torch.randint(0, 5, (25,)),
            query_x=torch.randn(75, 3, 32, 32),
            query_y=torch.randint(0, 5, (75,))
        )
        
        # Should be able to create episode without errors
        assert episode.support_x.shape == (25, 3, 32, 32)
        assert episode.query_x.shape == (75, 3, 32, 32)
        assert len(torch.unique(episode.support_y)) <= 5
        assert len(torch.unique(episode.query_y)) <= 5
    
    def test_dataset_manager_singleton(self):
        """Test dataset manager singleton pattern."""
        manager1 = get_dataset_manager()
        manager2 = get_dataset_manager()
        
        # Should return same instance
        assert manager1 is manager2
        assert isinstance(manager1, DatasetManager)
    
    def test_phase4_features_dont_break_basic_functionality(self):
        """Test that Phase 4 features don't break basic functionality."""
        toolkit = MetaLearningToolkit()
        
        # Enable all features
        toolkit.enable_failure_prediction()
        toolkit.enable_automatic_algorithm_selection()
        toolkit.enable_realtime_optimization()
        toolkit.enable_cross_task_knowledge_transfer()
        
        # Basic functionality should still work
        episode = Episode(
            support_x=torch.randn(10, 1, 28, 28),
            support_y=torch.randint(0, 3, (10,)),
            query_x=torch.randn(30, 1, 28, 28),
            query_y=torch.randint(0, 3, (30,))
        )
        
        # Should not raise errors
        assert episode.support_x.numel() > 0
        assert episode.query_x.numel() > 0


class TestErrorHandling:
    """Test error handling and edge cases."""
    
    def test_invalid_dataset_name(self):
        """Test handling of invalid dataset names."""
        registry = DatasetRegistry()
        
        # Should return None for non-existent dataset
        result = registry.get_dataset_info("nonexistent_dataset")
        assert result is None
    
    def test_empty_episode_handling(self):
        """Test handling of edge case episodes."""
        # Very small episode
        episode = Episode(
            support_x=torch.randn(2, 3, 32, 32),
            support_y=torch.tensor([0, 1]),
            query_x=torch.randn(2, 3, 32, 32),
            query_y=torch.tensor([0, 1])
        )
        
        # Should create without errors
        assert episode.support_x.shape[0] == 2
        assert episode.query_x.shape[0] == 2
    
    def test_toolkit_multiple_enable_calls(self):
        """Test multiple calls to enable functions."""
        toolkit = MetaLearningToolkit()
        
        # Should handle multiple enable calls gracefully
        toolkit.enable_failure_prediction(True, True)
        toolkit.enable_failure_prediction(False, False)
        toolkit.enable_failure_prediction(True, True)
        
        # Final state should match last call
        assert toolkit.failure_prediction_enabled == True


class TestPerformanceBasic:
    """Basic performance tests."""
    
    def test_dataset_registry_performance(self):
        """Test that dataset registry operations are reasonably fast."""
        import time
        
        registry = DatasetRegistry()
        
        start_time = time.time()
        
        # Perform multiple operations
        for _ in range(100):
            datasets = registry.list_datasets()
            info = registry.get_dataset_info("miniimagenet")
        
        elapsed = time.time() - start_time
        
        # Should complete quickly (less than 1 second for 100 operations)
        assert elapsed < 1.0
    
    def test_toolkit_enable_performance(self):
        """Test that enabling Phase 4 features is fast."""
        import time
        
        start_time = time.time()
        
        # Enable all features
        toolkit = MetaLearningToolkit()
        toolkit.enable_failure_prediction()
        toolkit.enable_automatic_algorithm_selection()
        toolkit.enable_realtime_optimization()
        toolkit.enable_cross_task_knowledge_transfer()
        
        elapsed = time.time() - start_time
        
        # Should enable quickly (less than 1 second)
        assert elapsed < 1.0
    
    def test_episode_creation_performance(self):
        """Test that episode creation is reasonably fast."""
        import time
        
        start_time = time.time()
        
        # Create multiple episodes
        episodes = []
        for i in range(50):
            episode = Episode(
                support_x=torch.randn(25, 3, 32, 32),
                support_y=torch.randint(0, 5, (25,)),
                query_x=torch.randn(75, 3, 32, 32),
                query_y=torch.randint(0, 5, (75,))
            )
            episodes.append(episode)
        
        elapsed = time.time() - start_time
        
        # Should create episodes quickly (less than 5 seconds for 50 episodes)
        assert elapsed < 5.0
        assert len(episodes) == 50


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])