#!/usr/bin/env python3
"""
Test CurriculumSampler
=====================

Tests for the CurriculumSampler class that implements progressive difficulty
increase following a curriculum schedule for meta-learning.
"""

import pytest
import numpy as np
from typing import Dict, Any

from meta_learning.data_utils.iterators import CurriculumSampler


class TestCurriculumSampler:
    """Test CurriculumSampler functionality."""

    def test_initialization(self):
        """Test sampler initialization with default parameters."""
        sampler = CurriculumSampler()
        
        assert sampler.initial_difficulty == 0.1
        assert sampler.target_difficulty == 0.9
        assert sampler.progression_rate == 0.05
        assert sampler.milestone_threshold == 0.75
        assert sampler.current_difficulty == 0.1
        assert sampler.performance_history == []
        assert sampler.milestones_reached == 0

    def test_initialization_custom_params(self):
        """Test sampler initialization with custom parameters."""
        sampler = CurriculumSampler(
            initial_difficulty=0.2,
            target_difficulty=0.8,
            progression_rate=0.1,
            milestone_threshold=0.8
        )
        
        assert sampler.initial_difficulty == 0.2
        assert sampler.target_difficulty == 0.8
        assert sampler.progression_rate == 0.1
        assert sampler.milestone_threshold == 0.8
        assert sampler.current_difficulty == 0.2

    def test_performance_history_tracking(self):
        """Test that performance history is tracked correctly."""
        sampler = CurriculumSampler()
        
        performances = [0.1, 0.5, 0.8, 0.6, 0.9]
        for perf in performances:
            sampler.update_performance(perf)
        
        assert len(sampler.performance_history) == len(performances)
        assert sampler.performance_history == performances

    def test_performance_history_sliding_window(self):
        """Test that performance history maintains sliding window of 50."""
        sampler = CurriculumSampler()
        
        # Add more than 50 performance values
        for i in range(55):
            sampler.update_performance(i / 55.0)
        
        # Should keep only last 50
        assert len(sampler.performance_history) == 50
        assert sampler.performance_history == [i / 55.0 for i in range(5, 55)]

    def test_milestone_advancement_insufficient_history(self):
        """Test that milestone advancement requires sufficient performance history."""
        sampler = CurriculumSampler()
        initial_difficulty = sampler.current_difficulty
        
        # Add only 9 performance values (need 10 for milestone evaluation)
        for _ in range(9):
            sampler.update_performance(0.8)  # Above threshold
        
        # Should not advance yet
        assert sampler.current_difficulty == initial_difficulty
        assert sampler.milestones_reached == 0

    def test_milestone_advancement_sufficient_performance(self):
        """Test milestone advancement with sufficient good performance."""
        sampler = CurriculumSampler()
        initial_difficulty = sampler.current_difficulty
        
        # Add 10 performance values above threshold
        for _ in range(10):
            sampler.update_performance(0.8)  # Above default threshold 0.75
        
        # Should advance difficulty
        assert sampler.current_difficulty > initial_difficulty
        expected_difficulty = min(0.9, initial_difficulty + 0.05)  # progression_rate
        assert abs(sampler.current_difficulty - expected_difficulty) < 1e-6
        assert sampler.milestones_reached == 1

    def test_milestone_advancement_insufficient_performance(self):
        """Test that poor performance doesn't trigger advancement."""
        sampler = CurriculumSampler()
        initial_difficulty = sampler.current_difficulty
        
        # Add 10 performance values below threshold
        for _ in range(10):
            sampler.update_performance(0.6)  # Below default threshold 0.75
        
        # Should not advance
        assert sampler.current_difficulty == initial_difficulty
        assert sampler.milestones_reached == 0

    def test_target_difficulty_limit(self):
        """Test that difficulty doesn't exceed target difficulty."""
        sampler = CurriculumSampler(target_difficulty=0.3)  # Low target
        
        # Try to advance beyond target
        for _ in range(10):
            sampler.update_performance(0.9)  # High performance
        
        # Should not exceed target
        assert sampler.current_difficulty <= 0.3
        # Should reach exactly the target
        assert abs(sampler.current_difficulty - 0.3) < 1e-6

    def test_multiple_milestone_advancement(self):
        """Test multiple sequential milestone advancements."""
        sampler = CurriculumSampler(progression_rate=0.1)
        milestones = []
        difficulties = []
        
        milestones.append(sampler.milestones_reached)
        difficulties.append(sampler.current_difficulty)
        
        # First milestone
        for _ in range(10):
            sampler.update_performance(0.8)
        milestones.append(sampler.milestones_reached)
        difficulties.append(sampler.current_difficulty)
        
        # Second milestone (history cleared after advancement)
        for _ in range(10):
            sampler.update_performance(0.8)
        milestones.append(sampler.milestones_reached)
        difficulties.append(sampler.current_difficulty)
        
        # Verify progression
        assert milestones == [0, 1, 2]
        assert difficulties[1] > difficulties[0]
        assert difficulties[2] > difficulties[1]
        assert abs(difficulties[1] - (difficulties[0] + 0.1)) < 1e-6
        assert abs(difficulties[2] - (difficulties[1] + 0.1)) < 1e-6

    def test_history_clearing_after_advancement(self):
        """Test that performance history is cleared after milestone advancement."""
        sampler = CurriculumSampler()
        
        # Build up performance history
        for _ in range(10):
            sampler.update_performance(0.8)
        
        # Should advance and clear history
        assert sampler.milestones_reached == 1
        assert len(sampler.performance_history) == 0

    def test_recent_performance_calculation(self):
        """Test that milestone evaluation uses last 10 performances."""
        sampler = CurriculumSampler()
        initial_difficulty = sampler.current_difficulty
        
        # Add poor performance first
        for _ in range(20):
            sampler.update_performance(0.5)  # Below threshold
        
        # Add good performance (last 10)
        for _ in range(10):
            sampler.update_performance(0.8)  # Above threshold
        
        # Should advance based on recent good performance
        assert sampler.current_difficulty > initial_difficulty

    def test_sample_episode_params_structure(self):
        """Test that sampled episode parameters have correct structure."""
        sampler = CurriculumSampler()
        
        params = sampler.sample_episode_params()
        
        assert isinstance(params, dict)
        assert 'n_way' in params
        assert 'n_shot' in params
        assert 'n_query' in params
        assert 'difficulty' in params
        assert 'curriculum_phase' in params
        
        # Check types
        assert isinstance(params['n_way'], int)
        assert isinstance(params['n_shot'], int)
        assert isinstance(params['n_query'], int)
        assert isinstance(params['difficulty'], float)
        assert isinstance(params['curriculum_phase'], str)

    def test_curriculum_phases(self):
        """Test different curriculum phases based on difficulty."""
        sampler = CurriculumSampler()
        
        # Easy phase (difficulty <= 0.3)
        sampler.current_difficulty = 0.2
        params = sampler.sample_episode_params()
        assert params['curriculum_phase'] == 'easy'
        assert params['n_way'] in [3, 4]
        assert params['n_shot'] in [4, 5, 6]
        assert params['n_query'] in [8, 10]
        
        # Medium phase (0.3 < difficulty <= 0.6)
        sampler.current_difficulty = 0.5
        params = sampler.sample_episode_params()
        assert params['curriculum_phase'] == 'medium'
        assert params['n_way'] in [4, 5]
        assert params['n_shot'] in [2, 3, 4]
        assert params['n_query'] in [10, 12, 15]
        
        # Hard phase (difficulty > 0.6)
        sampler.current_difficulty = 0.8
        params = sampler.sample_episode_params()
        assert params['curriculum_phase'] == 'hard'
        assert params['n_way'] in [5, 6, 7]
        assert params['n_shot'] in [1, 2]
        assert params['n_query'] in [15, 18, 20]

    def test_curriculum_phase_names(self):
        """Test curriculum phase name generation."""
        sampler = CurriculumSampler()
        
        # Test boundary conditions
        sampler.current_difficulty = 0.3
        assert sampler._get_curriculum_phase() == 'easy'
        
        sampler.current_difficulty = 0.31
        assert sampler._get_curriculum_phase() == 'medium'
        
        sampler.current_difficulty = 0.6
        assert sampler._get_curriculum_phase() == 'medium'
        
        sampler.current_difficulty = 0.61
        assert sampler._get_curriculum_phase() == 'hard'

    def test_complete_curriculum_progression(self):
        """Test complete curriculum progression from easy to hard."""
        sampler = CurriculumSampler(
            initial_difficulty=0.1,
            target_difficulty=0.7,
            progression_rate=0.2,
            milestone_threshold=0.75
        )
        
        progression = []
        phases = []
        
        # Track initial state
        progression.append(sampler.current_difficulty)
        phases.append(sampler._get_curriculum_phase())
        
        # Progress through curriculum
        while sampler.current_difficulty < sampler.target_difficulty:
            # Simulate consistent good performance
            for _ in range(10):
                sampler.update_performance(0.8)
            
            progression.append(sampler.current_difficulty)
            phases.append(sampler._get_curriculum_phase())
            
            # Safety break
            if len(progression) > 10:
                break
        
        # Verify progression
        assert len(progression) >= 3  # At least initial + 2 advances
        assert all(progression[i] <= progression[i+1] for i in range(len(progression)-1))
        assert 'easy' in phases
        assert phases[-1] in ['medium', 'hard']  # Should advance beyond easy

    def test_custom_threshold_behavior(self):
        """Test curriculum with custom milestone threshold."""
        sampler = CurriculumSampler(milestone_threshold=0.9)  # Very high threshold
        initial_difficulty = sampler.current_difficulty
        
        # Performance that would normally advance (0.8 > default 0.75)
        for _ in range(10):
            sampler.update_performance(0.8)
        
        # Should not advance with high threshold
        assert sampler.current_difficulty == initial_difficulty
        
        # Performance above custom threshold
        for _ in range(10):
            sampler.update_performance(0.95)
        
        # Should now advance
        assert sampler.current_difficulty > initial_difficulty

    def test_edge_case_zero_progression_rate(self):
        """Test curriculum with zero progression rate."""
        sampler = CurriculumSampler(progression_rate=0.0)
        initial_difficulty = sampler.current_difficulty
        
        # Even with good performance
        for _ in range(10):
            sampler.update_performance(0.9)
        
        # Should not advance (zero rate)
        assert sampler.current_difficulty == initial_difficulty
        assert sampler.milestones_reached == 0  # No milestone counted

    def test_difficulty_parameter_consistency(self):
        """Test that difficulty parameter in sampled episodes matches current difficulty."""
        sampler = CurriculumSampler()
        
        # Test at different difficulty levels
        test_difficulties = [0.1, 0.4, 0.7]
        
        for difficulty in test_difficulties:
            sampler.current_difficulty = difficulty
            
            for _ in range(5):  # Multiple samples for consistency
                params = sampler.sample_episode_params()
                assert params['difficulty'] == difficulty


if __name__ == "__main__":
    pytest.main([__file__])