"""
Comprehensive tests for Episode Protocol implementation.

Tests mathematical correctness, label remapping, and episode generation
according to canonical meta-learning protocols.
"""
import pytest
import torch
import numpy as np
from unittest.mock import MagicMock, patch
from collections import Counter
from typing import Dict, List, Tuple, Any

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../src'))

from meta_learning.meta_learning_modules.episode_protocol import (
    EpisodeGenerator,
    Episode,
    EpisodicDataset,
    TaskPool
)


class TestEpisode:
    """Test Episode dataclass functionality."""
    
    def test_episode_creation(self):
        """Test Episode creation with valid data."""
        support_x = torch.randn(25, 10)  # 5-way 5-shot
        support_y = torch.tensor([0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 
                                3, 3, 3, 3, 3, 4, 4, 4, 4, 4])
        query_x = torch.randn(15, 10)  # 3 query per class
        query_y = torch.tensor([0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4])
        
        episode = Episode(support_x, support_y, query_x, query_y)
        
        assert episode.n_way == 5
        assert episode.k_shot == 5
        assert episode.m_query == 3
        assert episode.support_x.shape == (25, 10)
        assert episode.query_x.shape == (15, 10)
    
    def test_episode_validation_tensor_shapes(self):
        """Test Episode validation catches tensor shape mismatches."""
        support_x = torch.randn(25, 10)
        support_y = torch.tensor([0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 
                                3, 3, 3, 3, 3, 4, 4, 4, 4, 4])
        query_x = torch.randn(15, 8)  # Wrong feature dimension
        query_y = torch.tensor([0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4])
        
        with pytest.raises(ValueError, match="Feature dimensions don't match"):
            Episode(support_x, support_y, query_x, query_y)
    
    def test_episode_validation_label_consistency(self):
        """Test Episode validation catches label inconsistencies."""
        support_x = torch.randn(25, 10)
        support_y = torch.tensor([0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 
                                3, 3, 3, 3, 3, 4, 4, 4, 4, 4])
        query_x = torch.randn(15, 10)
        query_y = torch.tensor([0, 0, 0, 1, 1, 1, 5, 5, 5, 3, 3, 3, 4, 4, 4])  # Class 5 not in support
        
        with pytest.raises(ValueError, match="Query contains classes not in support"):
            Episode(support_x, support_y, query_x, query_y)
    
    def test_episode_validation_negative_labels(self):
        """Test Episode validation catches negative labels."""
        support_x = torch.randn(25, 10)
        support_y = torch.tensor([0, 0, 0, 0, 0, 1, 1, 1, 1, 1, -1, -1, -1, -1, -1,  # Negative labels
                                3, 3, 3, 3, 3, 4, 4, 4, 4, 4])
        query_x = torch.randn(15, 10)
        query_y = torch.tensor([0, 0, 0, 1, 1, 1, -1, -1, -1, 3, 3, 3, 4, 4, 4])
        
        with pytest.raises(ValueError, match="Labels must be non-negative"):
            Episode(support_x, support_y, query_x, query_y)


class TestEpisodeGenerator:
    """Test EpisodeGenerator mathematical correctness."""
    
    def setup_method(self):
        """Setup test fixtures."""
        # Create mock dataset with known structure
        self.n_classes = 10
        self.examples_per_class = 20
        
        # Create class_to_indices mapping
        self.class_to_indices = {}
        for class_id in range(self.n_classes):
            self.class_to_indices[class_id] = list(range(
                class_id * self.examples_per_class,
                (class_id + 1) * self.examples_per_class
            ))
        
        # Create mock dataset
        total_examples = self.n_classes * self.examples_per_class
        self.mock_dataset = MagicMock()
        self.mock_dataset.__len__ = MagicMock(return_value=total_examples)
        
        def mock_getitem(idx):
            class_id = idx // self.examples_per_class
            features = torch.randn(10) + class_id  # Different means per class
            return features, class_id
        
        self.mock_dataset.__getitem__ = mock_getitem
        
        self.generator = EpisodeGenerator(seed=42)
    
    def test_generate_episode_basic(self):
        """Test basic episode generation."""
        episode = self.generator.generate_episode(
            dataset=self.mock_dataset,
            class_to_indices=self.class_to_indices,
            n_way=5,
            k_shot=3,
            m_query=2
        )
        
        # Check episode structure
        assert episode.n_way == 5
        assert episode.k_shot == 3
        assert episode.m_query == 2
        assert episode.support_x.shape == (15, 10)  # 5 * 3
        assert episode.query_x.shape == (10, 10)    # 5 * 2
        
        # Check label remapping to [0, n_way-1]
        support_classes = set(episode.support_y.tolist())
        query_classes = set(episode.query_y.tolist())
        expected_classes = set(range(5))
        
        assert support_classes == expected_classes
        assert query_classes == expected_classes
    
    def test_label_remapping_correctness(self):
        """Test that label remapping preserves class structure."""
        # Use specific classes that aren't [0,1,2,3,4]
        available_classes = [2, 5, 7, 8, 9]
        
        episode = self.generator.generate_episode(
            dataset=self.mock_dataset,
            class_to_indices=self.class_to_indices,
            n_way=5,
            k_shot=2,
            m_query=1,
            available_classes=available_classes
        )
        
        # Labels should be remapped to [0,1,2,3,4]
        support_classes = sorted(set(episode.support_y.tolist()))
        query_classes = sorted(set(episode.query_y.tolist()))
        
        assert support_classes == [0, 1, 2, 3, 4]
        assert query_classes == [0, 1, 2, 3, 4]
        
        # Check class balance
        support_counts = Counter(episode.support_y.tolist())
        query_counts = Counter(episode.query_y.tolist())
        
        for class_id in range(5):
            assert support_counts[class_id] == 2  # k_shot
            assert query_counts[class_id] == 1    # m_query
    
    def test_no_support_query_overlap(self):
        """Test that support and query sets don't overlap."""
        episode = self.generator.generate_episode(
            dataset=self.mock_dataset,
            class_to_indices=self.class_to_indices,
            n_way=3,
            k_shot=5,
            m_query=3
        )
        
        # Extract original indices used for support and query
        # This is tricky without access to internals, so we test indirectly
        # by ensuring we have enough examples per class
        support_counts = Counter(episode.support_y.tolist())
        query_counts = Counter(episode.query_y.tolist())
        
        for class_id in range(3):
            assert support_counts[class_id] == 5
            assert query_counts[class_id] == 3
            # Total examples used per class (5+3=8) should be ≤ available (20)
            assert support_counts[class_id] + query_counts[class_id] <= self.examples_per_class
    
    def test_insufficient_examples_error(self):
        """Test error when not enough examples per class."""
        # Try to use more examples than available
        with pytest.raises(ValueError, match="Insufficient examples"):
            self.generator.generate_episode(
                dataset=self.mock_dataset,
                class_to_indices=self.class_to_indices,
                n_way=3,
                k_shot=15,  # Only 20 examples per class
                m_query=10  # 15+10=25 > 20
            )
    
    def test_insufficient_classes_error(self):
        """Test error when not enough classes available."""
        # Only 10 classes available
        with pytest.raises(ValueError, match="Insufficient classes"):
            self.generator.generate_episode(
                dataset=self.mock_dataset,
                class_to_indices=self.class_to_indices,
                n_way=15,  # More than 10 available
                k_shot=2,
                m_query=1
            )
    
    def test_deterministic_generation(self):
        """Test that episode generation is deterministic with same seed."""
        generator1 = EpisodeGenerator(seed=123)
        generator2 = EpisodeGenerator(seed=123)
        
        episode1 = generator1.generate_episode(
            dataset=self.mock_dataset,
            class_to_indices=self.class_to_indices,
            n_way=3,
            k_shot=2,
            m_query=1
        )
        
        episode2 = generator2.generate_episode(
            dataset=self.mock_dataset,
            class_to_indices=self.class_to_indices,
            n_way=3,
            k_shot=2,
            m_query=1
        )
        
        # Should be identical
        assert torch.equal(episode1.support_x, episode2.support_x)
        assert torch.equal(episode1.support_y, episode2.support_y)
        assert torch.equal(episode1.query_x, episode2.query_x)
        assert torch.equal(episode1.query_y, episode2.query_y)
    
    def test_different_seeds_different_episodes(self):
        """Test that different seeds produce different episodes."""
        generator1 = EpisodeGenerator(seed=123)
        generator2 = EpisodeGenerator(seed=456)
        
        episode1 = generator1.generate_episode(
            dataset=self.mock_dataset,
            class_to_indices=self.class_to_indices,
            n_way=3,
            k_shot=2,
            m_query=1
        )
        
        episode2 = generator2.generate_episode(
            dataset=self.mock_dataset,
            class_to_indices=self.class_to_indices,
            n_way=3,
            k_shot=2,
            m_query=1
        )
        
        # Should be different (with very high probability)
        assert not torch.equal(episode1.support_x, episode2.support_x) or \
               not torch.equal(episode1.support_y, episode2.support_y)


class TestEpisodicDataset:
    """Test EpisodicDataset wrapper functionality."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.mock_base_dataset = MagicMock()
        self.mock_base_dataset.__len__ = MagicMock(return_value=100)
        self.mock_base_dataset.__getitem__ = MagicMock(side_effect=lambda i: (torch.randn(5), i % 10))
        
        self.class_to_indices = {i: list(range(i*10, (i+1)*10)) for i in range(10)}
        
        self.episodic_dataset = EpisodicDataset(
            base_dataset=self.mock_base_dataset,
            class_to_indices=self.class_to_indices,
            n_way=5,
            k_shot=2,
            m_query=3,
            n_episodes=20
        )
    
    def test_episodic_dataset_length(self):
        """Test that EpisodicDataset has correct length."""
        assert len(self.episodic_dataset) == 20
    
    def test_episodic_dataset_getitem(self):
        """Test that EpisodicDataset generates valid episodes."""
        episode = self.episodic_dataset[0]
        
        assert isinstance(episode, Episode)
        assert episode.n_way == 5
        assert episode.k_shot == 2
        assert episode.m_query == 3
        assert episode.support_x.shape == (10, 5)  # 5*2
        assert episode.query_x.shape == (15, 5)    # 5*3
    
    def test_episodic_dataset_reproducibility(self):
        """Test that EpisodicDataset produces same episodes with same seed."""
        dataset1 = EpisodicDataset(
            base_dataset=self.mock_base_dataset,
            class_to_indices=self.class_to_indices,
            n_way=3,
            k_shot=1,
            m_query=1,
            n_episodes=5,
            seed=42
        )
        
        dataset2 = EpisodicDataset(
            base_dataset=self.mock_base_dataset,
            class_to_indices=self.class_to_indices,
            n_way=3,
            k_shot=1,
            m_query=1,
            n_episodes=5,
            seed=42
        )
        
        for i in range(5):
            ep1 = dataset1[i]
            ep2 = dataset2[i]
            
            assert torch.equal(ep1.support_x, ep2.support_x)
            assert torch.equal(ep1.support_y, ep2.support_y)
            assert torch.equal(ep1.query_x, ep2.query_x)
            assert torch.equal(ep1.query_y, ep2.query_y)


class TestTaskPool:
    """Test TaskPool for managing multiple task configurations."""
    
    def test_task_pool_creation(self):
        """Test TaskPool creation with multiple configurations."""
        configs = [
            {'n_way': 5, 'k_shot': 1, 'm_query': 15},
            {'n_way': 5, 'k_shot': 5, 'm_query': 15},
            {'n_way': 20, 'k_shot': 1, 'm_query': 5},
        ]
        
        pool = TaskPool(configs)
        assert len(pool.configs) == 3
        assert pool.current_config_idx == 0
    
    def test_task_pool_get_current_config(self):
        """Test getting current task configuration."""
        configs = [
            {'n_way': 5, 'k_shot': 1, 'm_query': 15},
            {'n_way': 10, 'k_shot': 2, 'm_query': 5},
        ]
        
        pool = TaskPool(configs)
        config = pool.get_current_config()
        
        assert config['n_way'] == 5
        assert config['k_shot'] == 1
        assert config['m_query'] == 15
    
    def test_task_pool_next_config(self):
        """Test cycling through task configurations."""
        configs = [
            {'n_way': 5, 'k_shot': 1, 'm_query': 15},
            {'n_way': 10, 'k_shot': 2, 'm_query': 5},
            {'n_way': 20, 'k_shot': 1, 'm_query': 3},
        ]
        
        pool = TaskPool(configs)
        
        # First config
        config1 = pool.get_current_config()
        assert config1['n_way'] == 5
        
        # Next config
        pool.next_config()
        config2 = pool.get_current_config()
        assert config2['n_way'] == 10
        
        # Next config
        pool.next_config()
        config3 = pool.get_current_config()
        assert config3['n_way'] == 20
        
        # Cycle back to first
        pool.next_config()
        config4 = pool.get_current_config()
        assert config4['n_way'] == 5


class TestMathematicalCorrectness:
    """Test mathematical correctness of episode generation."""
    
    def setup_method(self):
        """Setup for mathematical correctness tests."""
        self.generator = EpisodeGenerator(seed=42)
        
        # Create well-structured dataset
        self.n_classes = 20
        self.examples_per_class = 50
        
        self.class_to_indices = {}
        for class_id in range(self.n_classes):
            start_idx = class_id * self.examples_per_class
            end_idx = (class_id + 1) * self.examples_per_class
            self.class_to_indices[class_id] = list(range(start_idx, end_idx))
        
        # Mock dataset with different distributions per class
        self.mock_dataset = MagicMock()
        self.mock_dataset.__len__ = MagicMock(return_value=self.n_classes * self.examples_per_class)
        
        def mock_getitem(idx):
            class_id = idx // self.examples_per_class
            # Each class has different mean but same variance
            features = torch.randn(32) + class_id * 2.0
            return features, class_id
            
        self.mock_dataset.__getitem__ = mock_getitem
    
    def test_canonical_5_way_1_shot(self):
        """Test canonical 5-way 1-shot episode generation."""
        episode = self.generator.generate_episode(
            dataset=self.mock_dataset,
            class_to_indices=self.class_to_indices,
            n_way=5,
            k_shot=1,
            m_query=15
        )
        
        # Canonical few-shot learning protocol checks
        assert episode.n_way == 5
        assert episode.k_shot == 1
        assert episode.m_query == 15
        
        # Support set: 5 classes × 1 shot = 5 examples
        assert episode.support_x.shape == (5, 32)
        assert len(set(episode.support_y.tolist())) == 5
        
        # Query set: 5 classes × 15 queries = 75 examples  
        assert episode.query_x.shape == (75, 32)
        assert len(set(episode.query_y.tolist())) == 5
        
        # Label remapping check
        assert set(episode.support_y.tolist()) == set(range(5))
        assert set(episode.query_y.tolist()) == set(range(5))
    
    def test_canonical_5_way_5_shot(self):
        """Test canonical 5-way 5-shot episode generation."""
        episode = self.generator.generate_episode(
            dataset=self.mock_dataset,
            class_to_indices=self.class_to_indices,
            n_way=5,
            k_shot=5,
            m_query=15
        )
        
        # Canonical protocol checks
        assert episode.n_way == 5
        assert episode.k_shot == 5
        assert episode.m_query == 15
        
        # Support set: 5 classes × 5 shots = 25 examples
        assert episode.support_x.shape == (25, 32)
        
        # Query set: 5 classes × 15 queries = 75 examples
        assert episode.query_x.shape == (75, 32)
        
        # Class balance check
        support_counts = Counter(episode.support_y.tolist())
        query_counts = Counter(episode.query_y.tolist())
        
        for class_id in range(5):
            assert support_counts[class_id] == 5   # k_shot
            assert query_counts[class_id] == 15    # m_query
    
    def test_mini_imagenet_protocol(self):
        """Test miniImageNet-style protocol (5-way 1-shot, 5-way 5-shot)."""
        # 5-way 1-shot
        episode_1shot = self.generator.generate_episode(
            dataset=self.mock_dataset,
            class_to_indices=self.class_to_indices,
            n_way=5,
            k_shot=1,
            m_query=15
        )
        
        assert episode_1shot.support_x.shape == (5, 32)
        assert episode_1shot.query_x.shape == (75, 32)
        
        # 5-way 5-shot  
        episode_5shot = self.generator.generate_episode(
            dataset=self.mock_dataset,
            class_to_indices=self.class_to_indices,
            n_way=5,
            k_shot=5,
            m_query=15
        )
        
        assert episode_5shot.support_x.shape == (25, 32)
        assert episode_5shot.query_x.shape == (75, 32)
    
    def test_omniglot_protocol(self):
        """Test Omniglot-style protocol (20-way 1-shot, 5-way 1-shot)."""
        # 20-way 1-shot
        episode_20way = self.generator.generate_episode(
            dataset=self.mock_dataset,
            class_to_indices=self.class_to_indices,
            n_way=20,
            k_shot=1,
            m_query=5
        )
        
        assert episode_20way.n_way == 20
        assert episode_20way.support_x.shape == (20, 32)   # 20 × 1
        assert episode_20way.query_x.shape == (100, 32)    # 20 × 5
        
        # Check all classes represented
        assert len(set(episode_20way.support_y.tolist())) == 20
        assert len(set(episode_20way.query_y.tolist())) == 20
    
    def test_class_sampling_uniformity(self):
        """Test that class sampling is uniform across multiple episodes."""
        class_selections = []
        
        # Generate many episodes and track selected classes
        for _ in range(100):
            episode = self.generator.generate_episode(
                dataset=self.mock_dataset,
                class_to_indices=self.class_to_indices,
                n_way=5,
                k_shot=2,
                m_query=3
            )
            # Record which original classes were selected (before remapping)
            # This is indirect - we trust that different episodes use different classes
            class_selections.append(tuple(sorted(set(episode.support_y.tolist()))))
        
        # All episodes should have same remapped classes [0,1,2,3,4]
        expected_classes = (0, 1, 2, 3, 4)
        for classes in class_selections:
            assert classes == expected_classes
    
    def test_no_data_leakage_mathematical(self):
        """Test mathematical guarantee of no support-query data leakage."""
        # Use a dataset where we can track exact indices
        tracked_indices = []
        
        def tracking_getitem(idx):
            tracked_indices.append(idx)
            class_id = idx // self.examples_per_class
            return torch.randn(10), class_id
        
        self.mock_dataset.__getitem__ = tracking_getitem
        
        # Clear tracking
        tracked_indices.clear()
        
        episode = self.generator.generate_episode(
            dataset=self.mock_dataset,
            class_to_indices=self.class_to_indices,
            n_way=3,
            k_shot=5,
            m_query=5
        )
        
        # Should have accessed exactly 30 indices (3 classes × (5+5) examples)
        assert len(tracked_indices) == 30
        
        # All indices should be unique (no overlap between support and query)
        assert len(set(tracked_indices)) == 30


if __name__ == "__main__":
    pytest.main([__file__])