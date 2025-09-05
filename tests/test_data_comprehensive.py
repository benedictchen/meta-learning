"""Comprehensive Data Pipeline Tests for maximum data.py coverage."""

import pytest
import torch
import tempfile
import os
import json
import meta_learning as ml
from meta_learning.data import SyntheticFewShotDataset, make_episodes

class TestSyntheticDatasetComprehensive:
    """Comprehensive tests for SyntheticFewShotDataset targeting 100% coverage."""
    
    def test_basic_functionality(self):
        """Test basic dataset functionality."""
        dataset = SyntheticFewShotDataset(n_classes=10, dim=64, noise=0.1)
        support_x, support_y, query_x, query_y = dataset.sample_support_query(5, 3, 4, seed=42)
        
        assert support_x.shape == (15, 64)
        assert support_y.shape == (15,)
        assert query_x.shape == (20, 64)
        assert query_y.shape == (20,)
        assert torch.equal(torch.sort(torch.unique(support_y))[0], torch.arange(5))
        assert torch.equal(torch.sort(torch.unique(query_y))[0], torch.arange(5))
    
    def test_image_mode_comprehensive(self):
        """Test image mode with various dimensions."""
        # Small dimension requiring padding
        small_dataset = SyntheticFewShotDataset(n_classes=5, dim=32, noise=0.1, image_mode=True)
        support_x, support_y, query_x, query_y = small_dataset.sample_support_query(3, 2, 3, seed=123)
        
        assert support_x.dim() == 4
        assert query_x.dim() == 4
        assert support_x.shape[1] == 3
        assert support_x.shape[2] >= 16
        assert support_x.shape[3] >= 16
        
        # Large dimension needing truncation
        large_dataset = SyntheticFewShotDataset(n_classes=3, dim=5000, noise=0.2, image_mode=True)
        support_x, support_y, query_x, query_y = large_dataset.sample_support_query(2, 1, 1, seed=789)
        
        assert support_x.dim() == 4
        assert query_x.dim() == 4

class TestMakeEpisodesComprehensive:
    """Comprehensive tests for make_episodes function."""
    
    def test_basic_episode_generation(self):
        """Test basic episode generation."""
        dataset = SyntheticFewShotDataset(n_classes=10, dim=32, noise=0.1)
        episodes = list(make_episodes(dataset, n_way=5, k_shot=3, m_query=4, episodes=10))
        
        assert len(episodes) == 10
        for episode in episodes:
            assert episode.support_x.shape == (15, 32)
            assert episode.query_x.shape == (20, 32)
            episode.validate(expect_n_classes=5)
    
    def test_episode_seeding(self):
        """Test that episodes use incremental seeding."""
        dataset = SyntheticFewShotDataset(n_classes=8, dim=16, noise=0.1)
        episodes1 = list(make_episodes(dataset, n_way=3, k_shot=2, m_query=3, episodes=5))
        episodes2 = list(make_episodes(dataset, n_way=3, k_shot=2, m_query=3, episodes=5))
        
        for ep1, ep2 in zip(episodes1, episodes2):
            assert torch.allclose(ep1.support_x, ep2.support_x)
            assert torch.equal(ep1.support_y, ep2.support_y)
            assert torch.allclose(ep1.query_x, ep2.query_x)
            assert torch.equal(ep1.query_y, ep2.query_y)

class TestEpisodeClass:
    """Tests for Episode class and remap_labels function."""
    
    def test_episode_properties(self):
        """Test episode properties and validation."""
        support_x = torch.randn(6, 16)
        support_y = torch.tensor([0, 0, 1, 1, 2, 2])
        query_x = torch.randn(9, 16)  
        query_y = torch.tensor([0, 0, 0, 1, 1, 1, 2, 2, 2])
        
        episode = ml.Episode(support_x, support_y, query_x, query_y)
        episode.validate()
        episode.validate(expect_n_classes=3)
    
    def test_remap_labels_function(self):
        """Test remap_labels function directly."""
        from meta_learning.core.episode import remap_labels
        
        y_support = torch.tensor([5, 5, 10, 10, 15, 15])
        y_query = torch.tensor([5, 10, 15])
        
        ys_remapped, yq_remapped = remap_labels(y_support, y_query)
        
        assert torch.equal(ys_remapped, torch.tensor([0, 0, 1, 1, 2, 2]))
        assert torch.equal(yq_remapped, torch.tensor([0, 1, 2]))