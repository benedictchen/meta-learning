#!/usr/bin/env python3
"""
Test AdaptiveEpisodeSampler
==========================

Tests for the AdaptiveEpisodeSampler class that adjusts episode difficulty
based on model performance to maintain optimal learning challenge.
"""

import pytest
import numpy as np
from typing import Dict, Any

from meta_learning.data_utils.iterators import AdaptiveEpisodeSampler


class TestAdaptiveEpisodeSampler:
    """Test AdaptiveEpisodeSampler functionality."""

    def test_initialization(self):
        """Test sampler initialization with default parameters."""
        sampler = AdaptiveEpisodeSampler()
        
        assert sampler.easy_threshold == 0.3
        assert sampler.hard_threshold == 0.8
        assert sampler.adjustment_rate == 0.1
        assert sampler.current_difficulty == 0.5
        assert sampler.performance_history == []

    def test_initialization_custom_params(self):
        """Test sampler initialization with custom parameters."""
        sampler = AdaptiveEpisodeSampler(
            easy_threshold=0.2,
            hard_threshold=0.9,
            adjustment_rate=0.05
        )
        
        assert sampler.easy_threshold == 0.2
        assert sampler.hard_threshold == 0.9
        assert sampler.adjustment_rate == 0.05
        assert sampler.current_difficulty == 0.5

    def test_performance_history_tracking(self):
        """Test that performance history is tracked correctly."""
        sampler = AdaptiveEpisodeSampler()
        
        # Add some performance values
        performances = [0.1, 0.5, 0.8, 0.6, 0.9]
        for perf in performances:
            sampler.update_performance(perf)
        
        assert len(sampler.performance_history) == len(performances)
        assert sampler.performance_history == performances

    def test_performance_history_sliding_window(self):
        """Test that performance history maintains sliding window of 20."""
        sampler = AdaptiveEpisodeSampler()
        
        # Add more than 20 performance values
        for i in range(25):
            sampler.update_performance(i / 25.0)  # 0.0 to 0.96
        
        # Should keep only last 20
        assert len(sampler.performance_history) == 20
        assert sampler.performance_history == [i / 25.0 for i in range(5, 25)]

    def test_difficulty_decrease_poor_performance(self):
        """Test that difficulty decreases with consistently poor performance."""
        sampler = AdaptiveEpisodeSampler()
        initial_difficulty = sampler.current_difficulty
        
        # Simulate consistently poor performance (below easy_threshold=0.3)
        for _ in range(6):  # Need at least 5 for adjustment
            sampler.update_performance(0.1)
        
        assert sampler.current_difficulty < initial_difficulty
        # Should decrease by adjustment_rate=0.1
        expected_difficulty = max(0.1, initial_difficulty - 0.1)
        assert abs(sampler.current_difficulty - expected_difficulty) < 1e-6

    def test_difficulty_increase_good_performance(self):
        """Test that difficulty increases with consistently good performance."""
        sampler = AdaptiveEpisodeSampler()
        initial_difficulty = sampler.current_difficulty
        
        # Simulate consistently good performance (above hard_threshold=0.8)
        for _ in range(6):  # Need at least 5 for adjustment
            sampler.update_performance(0.9)
        
        assert sampler.current_difficulty > initial_difficulty
        # Should increase by adjustment_rate=0.1
        expected_difficulty = min(0.9, initial_difficulty + 0.1)
        assert abs(sampler.current_difficulty - expected_difficulty) < 1e-6

    def test_difficulty_stable_moderate_performance(self):
        """Test that difficulty stays stable with moderate performance."""
        sampler = AdaptiveEpisodeSampler()
        initial_difficulty = sampler.current_difficulty
        
        # Simulate moderate performance (between thresholds)
        for _ in range(6):
            sampler.update_performance(0.5)  # Between 0.3 and 0.8
        
        assert sampler.current_difficulty == initial_difficulty

    def test_difficulty_bounds(self):
        """Test that difficulty stays within bounds [0.1, 0.9]."""
        sampler = AdaptiveEpisodeSampler()
        
        # Drive difficulty to minimum
        sampler.current_difficulty = 0.15
        sampler.update_performance(0.1)
        for _ in range(5):
            sampler.update_performance(0.1)
        
        assert sampler.current_difficulty >= 0.1
        
        # Drive difficulty to maximum  
        sampler.current_difficulty = 0.85
        sampler.update_performance(0.9)
        for _ in range(5):
            sampler.update_performance(0.9)
        
        assert sampler.current_difficulty <= 0.9

    def test_recent_performance_calculation(self):
        """Test that adjustment uses last 5 performances only."""
        sampler = AdaptiveEpisodeSampler()
        initial_difficulty = sampler.current_difficulty
        
        # Add some good performances first
        for _ in range(10):
            sampler.update_performance(0.9)
        
        # Then add poor performances (last 5)
        for _ in range(5):
            sampler.update_performance(0.1)
        
        # Should adjust based on recent poor performance
        assert sampler.current_difficulty < initial_difficulty

    def test_sample_episode_params_structure(self):
        """Test that sampled episode parameters have correct structure."""
        sampler = AdaptiveEpisodeSampler()
        
        params = sampler.sample_episode_params()
        
        assert isinstance(params, dict)
        assert 'n_way' in params
        assert 'n_shot' in params
        assert 'n_query' in params
        assert 'difficulty' in params
        
        # Check types and ranges
        assert isinstance(params['n_way'], int)
        assert isinstance(params['n_shot'], int)
        assert isinstance(params['n_query'], int)
        assert isinstance(params['difficulty'], float)
        
        assert 3 <= params['n_way'] <= 7
        assert 1 <= params['n_shot'] <= 5
        assert 10 <= params['n_query'] <= 20
        assert 0.1 <= params['difficulty'] <= 0.9

    def test_sample_episode_params_easy_difficulty(self):
        """Test episode parameters for easy difficulty level."""
        sampler = AdaptiveEpisodeSampler()
        sampler.current_difficulty = 0.2  # Easy level
        
        # Sample multiple times to check consistency
        for _ in range(10):
            params = sampler.sample_episode_params()
            
            # Easy: fewer ways, more shots, fewer queries
            assert params['n_way'] in [3, 4]
            assert params['n_shot'] in [3, 4, 5]
            assert params['n_query'] in [10, 12]
            assert params['difficulty'] == 0.2

    def test_sample_episode_params_medium_difficulty(self):
        """Test episode parameters for medium difficulty level."""
        sampler = AdaptiveEpisodeSampler()
        sampler.current_difficulty = 0.5  # Medium level
        
        for _ in range(10):
            params = sampler.sample_episode_params()
            
            # Medium: moderate ways, moderate shots, moderate queries
            assert params['n_way'] in [4, 5]
            assert params['n_shot'] in [1, 2, 3]
            assert params['n_query'] in [12, 15]
            assert params['difficulty'] == 0.5

    def test_sample_episode_params_hard_difficulty(self):
        """Test episode parameters for hard difficulty level."""
        sampler = AdaptiveEpisodeSampler()
        sampler.current_difficulty = 0.8  # Hard level
        
        for _ in range(10):
            params = sampler.sample_episode_params()
            
            # Hard: more ways, fewer shots, more queries
            assert params['n_way'] in [5, 6, 7]
            assert params['n_shot'] in [1, 2]
            assert params['n_query'] in [15, 18, 20]
            assert params['difficulty'] == 0.8

    def test_adaptive_adjustment_sequence(self):
        """Test complete adaptive adjustment sequence."""
        sampler = AdaptiveEpisodeSampler()
        difficulties = [sampler.current_difficulty]
        
        # Phase 1: Poor performance -> difficulty should decrease
        for _ in range(6):
            sampler.update_performance(0.1)
        difficulties.append(sampler.current_difficulty)
        
        # Phase 2: Good performance -> difficulty should increase
        for _ in range(6):
            sampler.update_performance(0.9)
        difficulties.append(sampler.current_difficulty)
        
        # Phase 3: Moderate performance -> difficulty should stabilize
        for _ in range(6):
            sampler.update_performance(0.5)
        difficulties.append(sampler.current_difficulty)
        
        # Verify progression
        assert difficulties[1] < difficulties[0]  # Decreased after poor performance
        assert difficulties[2] > difficulties[1]  # Increased after good performance  
        assert difficulties[3] == difficulties[2]  # Stable after moderate performance

    def test_custom_thresholds_and_rates(self):
        """Test sampler with custom thresholds and adjustment rates."""
        sampler = AdaptiveEpisodeSampler(
            easy_threshold=0.4,
            hard_threshold=0.7,
            adjustment_rate=0.2
        )
        
        initial_difficulty = sampler.current_difficulty
        
        # Test with performance at 0.6 (between new thresholds)
        for _ in range(6):
            sampler.update_performance(0.6)
        
        assert sampler.current_difficulty == initial_difficulty  # Should be stable
        
        # Test with performance at 0.3 (below new easy_threshold)
        for _ in range(6):
            sampler.update_performance(0.3)
        
        expected_difficulty = max(0.1, initial_difficulty - 0.2)  # Custom rate
        assert abs(sampler.current_difficulty - expected_difficulty) < 1e-6

    def test_insufficient_history_no_adjustment(self):
        """Test that adjustment only happens with sufficient performance history."""
        sampler = AdaptiveEpisodeSampler()
        initial_difficulty = sampler.current_difficulty
        
        # Add only 4 performance values (need 5 for adjustment)
        for _ in range(4):
            sampler.update_performance(0.1)
        
        # Difficulty should not change yet
        assert sampler.current_difficulty == initial_difficulty
        
        # Add 5th value - now adjustment should happen
        sampler.update_performance(0.1)
        assert sampler.current_difficulty < initial_difficulty


if __name__ == "__main__":
    pytest.main([__file__])