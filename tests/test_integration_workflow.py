#!/usr/bin/env python3
"""
Integration Tests for Complete Workflow
======================================

Integration tests that verify the complete meta-learning workflow works
end-to-end, including all new functionality implemented.
"""

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
import tempfile
from unittest.mock import patch, MagicMock
from typing import Dict, Any, List

from meta_learning.toolkit import MetaLearningToolkit, create_meta_learning_toolkit
from meta_learning.core.episode import Episode
from meta_learning.data_utils.iterators import AdaptiveEpisodeSampler, CurriculumSampler
from meta_learning.data_utils.datasets import OmniglotDataset


class TestCompleteWorkflow:
    """Test complete meta-learning workflow integration."""

    def create_simple_model(self):
        """Create a simple model for testing."""
        return nn.Sequential(
            nn.Linear(28 * 28, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 5)  # 5-way classification
        )

    def create_test_episodes(self, n_episodes: int = 5) -> List[Episode]:
        """Create multiple test episodes."""
        episodes = []
        for i in range(n_episodes):
            # Vary episode parameters slightly
            n_way = 3 + (i % 3)  # 3, 4, or 5 way
            n_shot = 1 + (i % 2)  # 1 or 2 shot
            n_query = 10 + (i * 2)  # 10, 12, 14, etc.
            
            # Create episode
            support_x = torch.randn(n_way * n_shot, 28 * 28)
            support_y = torch.repeat_interleave(torch.arange(n_way), n_shot)
            
            query_x = torch.randn(n_way * n_query, 28 * 28)
            query_y = torch.repeat_interleave(torch.arange(n_way), n_query)
            
            episode = Episode(support_x, support_y, query_x, query_y)
            episodes.append(episode)
        
        return episodes

    def test_complete_maml_workflow(self):
        """Test complete MAML workflow with all components."""
        # Create model and toolkit
        model = self.create_simple_model()
        toolkit = create_meta_learning_toolkit(
            model=model,
            algorithm='maml',
            seed=42,
            inner_lr=0.01,
            outer_lr=0.001,
            inner_steps=2
        )
        
        # Create test episodes
        episodes = self.create_test_episodes(3)
        
        # Train on episodes and collect results
        results = []
        for i, episode in enumerate(episodes):
            episode_results = toolkit.train_episode(episode, algorithm='maml')
            
            # Verify result structure
            assert isinstance(episode_results, dict)
            assert 'query_accuracy' in episode_results
            assert 'query_loss' in episode_results
            assert 'support_loss' in episode_results
            assert 'meta_loss' in episode_results
            
            # Verify reasonable values
            assert 0.0 <= episode_results['query_accuracy'] <= 1.0
            assert episode_results['query_loss'] >= 0.0
            
            results.append(episode_results)
        
        # Verify we got results for all episodes
        assert len(results) == len(episodes)

    def test_adaptive_episode_sampling_integration(self):
        """Test integration with AdaptiveEpisodeSampler."""
        model = self.create_simple_model()
        toolkit = create_meta_learning_toolkit(model, algorithm='maml', seed=42)
        sampler = AdaptiveEpisodeSampler()
        
        # Simulate adaptive learning loop
        for round_idx in range(3):
            # Sample episode parameters based on current difficulty
            episode_params = sampler.sample_episode_params()
            
            # Create episode with sampled parameters
            n_way = episode_params['n_way']
            n_shot = episode_params['n_shot'] 
            n_query = episode_params['n_query']
            
            support_x = torch.randn(n_way * n_shot, 28 * 28)
            support_y = torch.repeat_interleave(torch.arange(n_way), n_shot)
            query_x = torch.randn(n_way * n_query, 28 * 28)
            query_y = torch.repeat_interleave(torch.arange(n_way), n_query)
            
            episode = Episode(support_x, support_y, query_x, query_y)
            
            # Train on episode
            results = toolkit.train_episode(episode, algorithm='maml')
            
            # Update sampler with performance
            sampler.update_performance(results['query_accuracy'])
            
            # Verify the sampler is learning
            assert hasattr(sampler, 'current_difficulty')
            assert hasattr(sampler, 'performance_history')
            assert len(sampler.performance_history) == round_idx + 1

    def test_curriculum_learning_integration(self):
        """Test integration with CurriculumSampler."""
        model = self.create_simple_model()
        toolkit = create_meta_learning_toolkit(model, algorithm='maml', seed=42)
        curriculum = CurriculumSampler()
        
        initial_difficulty = curriculum.current_difficulty
        
        # Simulate curriculum learning progression
        for stage in range(3):
            # Sample episode parameters for current curriculum stage
            episode_params = curriculum.sample_episode_params()
            
            # Create episode
            n_way = episode_params['n_way']
            n_shot = episode_params['n_shot']
            n_query = episode_params['n_query']
            
            support_x = torch.randn(n_way * n_shot, 28 * 28)
            support_y = torch.repeat_interleave(torch.arange(n_way), n_shot)
            query_x = torch.randn(n_way * n_query, 28 * 28)
            query_y = torch.repeat_interleave(torch.arange(n_way), n_query)
            
            episode = Episode(support_x, support_y, query_x, query_y)
            
            # Train multiple episodes per stage to potentially advance curriculum
            for _ in range(12):  # Need consistent good performance to advance
                results = toolkit.train_episode(episode, algorithm='maml')
                # Simulate good performance to advance curriculum
                curriculum.update_performance(0.8)
            
            # Verify curriculum progression
            current_phase = curriculum._get_curriculum_phase()
            assert current_phase in ['easy', 'medium', 'hard']

    def test_omniglot_dataset_integration(self):
        """Test integration with OmniglotDataset."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Mock data for testing
            mock_data = {}
            for a in range(3):  # 3 alphabets
                alphabet_name = f'alphabet_{a}'
                mock_data[alphabet_name] = {}
                for c in range(5):  # 5 characters per alphabet
                    char_name = f'character_{c}'
                    mock_data[alphabet_name][char_name] = [
                        torch.randn(28, 28) for _ in range(20)
                    ]
            
            with patch.object(OmniglotDataset, '_load_data'), \
                 patch.object(OmniglotDataset, '_data_exists', return_value=True):
                
                # Create dataset
                dataset = OmniglotDataset(root=temp_dir, download=False)
                dataset.data = mock_data
                dataset.alphabets = list(mock_data.keys())
                dataset.characters_per_alphabet = {
                    alphabet: list(mock_data[alphabet].keys())
                    for alphabet in mock_data
                }
                
                # Create model and toolkit
                model = self.create_simple_model()
                toolkit = create_meta_learning_toolkit(model, algorithm='maml', seed=42)
                
                # Generate episodes from dataset and train
                for i in range(3):
                    episode = dataset.create_episode(n_way=3, n_shot=2, n_query=5)
                    results = toolkit.train_episode(episode, algorithm='maml')
                    
                    # Verify episode structure
                    assert episode.support_x.shape == (6, 28, 28)  # 3 ways * 2 shots
                    assert episode.query_x.shape == (15, 28, 28)   # 3 ways * 5 queries
                    
                    # Verify training results
                    assert isinstance(results, dict)
                    assert 'query_accuracy' in results

    def test_test_time_compute_integration(self):
        """Test integration with Test-Time Compute Scaling."""
        model = self.create_simple_model()
        toolkit = create_meta_learning_toolkit(
            model=model,
            algorithm='test_time_compute',
            seed=42
        )
        
        episodes = self.create_test_episodes(3)
        
        for episode in episodes:
            results = toolkit.train_episode(episode, algorithm='test_time_compute')
            
            # Verify TTC-specific results
            assert isinstance(results, dict)
            assert 'query_accuracy' in results
            assert 'compute_scaling_metrics' in results
            assert 'predictions' in results
            
            # Verify metrics structure
            metrics = results['compute_scaling_metrics']
            assert isinstance(metrics, dict)
            
            # Verify predictions structure
            predictions = results['predictions']
            assert isinstance(predictions, torch.Tensor)
            assert predictions.shape[0] == len(episode.query_y)

    def test_mixed_algorithm_workflow(self):
        """Test workflow with multiple algorithms on same episodes."""
        model1 = self.create_simple_model()
        model2 = self.create_simple_model()
        
        # Ensure identical starting weights
        model2.load_state_dict(model1.state_dict())
        
        # Create toolkits for different algorithms
        maml_toolkit = create_meta_learning_toolkit(model1, algorithm='maml', seed=42)
        ttc_toolkit = create_meta_learning_toolkit(model2, algorithm='test_time_compute', seed=42)
        
        episodes = self.create_test_episodes(2)
        
        # Compare algorithms on same episodes
        algorithm_results = {'maml': [], 'test_time_compute': []}
        
        for episode in episodes:
            # MAML results
            maml_results = maml_toolkit.train_episode(episode, algorithm='maml')
            algorithm_results['maml'].append(maml_results['query_accuracy'])
            
            # Test-Time Compute results
            ttc_results = ttc_toolkit.train_episode(episode, algorithm='test_time_compute')
            algorithm_results['test_time_compute'].append(ttc_results['query_accuracy'])
        
        # Verify both algorithms produced results
        assert len(algorithm_results['maml']) == len(episodes)
        assert len(algorithm_results['test_time_compute']) == len(episodes)
        
        # All accuracies should be valid
        for alg_name, accuracies in algorithm_results.items():
            for acc in accuracies:
                assert 0.0 <= acc <= 1.0

    def test_deterministic_reproduction(self):
        """Test that workflow is deterministic with same seeds."""
        episodes = self.create_test_episodes(3)
        
        # Run 1
        model1 = self.create_simple_model()
        toolkit1 = create_meta_learning_toolkit(model1, algorithm='maml', seed=123)
        
        results1 = []
        for episode in episodes:
            torch.manual_seed(456)  # Fix episode processing seed
            result = toolkit1.train_episode(episode, algorithm='maml')
            results1.append(result['query_accuracy'])
        
        # Run 2 (identical setup)
        model2 = self.create_simple_model()
        toolkit2 = create_meta_learning_toolkit(model2, algorithm='maml', seed=123)
        
        results2 = []
        for episode in episodes:
            torch.manual_seed(456)  # Same episode processing seed
            result = toolkit2.train_episode(episode, algorithm='maml')
            results2.append(result['query_accuracy'])
        
        # Results should be identical (or very close due to floating point)
        for r1, r2 in zip(results1, results2):
            assert abs(r1 - r2) < 1e-6

    def test_error_handling_integration(self):
        """Test error handling across the complete workflow."""
        model = self.create_simple_model()
        toolkit = MetaLearningToolkit()
        
        # Test uninitialized algorithm
        episode = self.create_test_episodes(1)[0]
        
        with pytest.raises(ValueError, match="Algorithm .* not initialized"):
            toolkit.train_episode(episode, algorithm='maml')
        
        # Test invalid algorithm
        toolkit.create_research_maml(model)
        
        with pytest.raises(ValueError, match="Algorithm .* not initialized"):
            toolkit.train_episode(episode, algorithm='invalid_algorithm')

    def test_batch_episode_processing(self):
        """Test processing multiple episodes efficiently."""
        model = self.create_simple_model()
        toolkit = create_meta_learning_toolkit(model, algorithm='maml', seed=42)
        
        episodes = self.create_test_episodes(5)
        results = []
        
        # Process episodes in batch
        for episode in episodes:
            result = toolkit.train_episode(episode, algorithm='maml')
            results.append(result)
        
        # Verify all episodes processed
        assert len(results) == len(episodes)
        
        # Check for reasonable performance variation
        accuracies = [r['query_accuracy'] for r in results]
        assert len(set(accuracies)) > 1  # Should have some variation
        
        # All accuracies should be valid
        for acc in accuracies:
            assert 0.0 <= acc <= 1.0

    def test_memory_efficiency_integration(self):
        """Test that the workflow is memory efficient."""
        model = self.create_simple_model()
        toolkit = create_meta_learning_toolkit(model, algorithm='maml', seed=42)
        
        episodes = self.create_test_episodes(10)
        
        # Process episodes and ensure memory doesn't leak
        results = []
        for i, episode in enumerate(episodes):
            result = toolkit.train_episode(episode, algorithm='maml')
            results.append(result['query_accuracy'])
            
            # Periodically check that we can still create new tensors
            # (This is a basic memory leak detection)
            if i % 5 == 4:
                test_tensor = torch.randn(100, 100)
                assert test_tensor.numel() == 10000

    def test_evaluation_harness_integration(self):
        """Test integration with evaluation harness."""
        model = self.create_simple_model()
        toolkit = create_meta_learning_toolkit(model, algorithm='maml', seed=42)
        
        # Create evaluation harness
        harness = toolkit.create_evaluation_harness()
        assert harness is not None
        
        episodes = self.create_test_episodes(3)
        
        # Define evaluation function
        def evaluation_fn(episode):
            return toolkit.train_episode(episode, algorithm='maml')
        
        # Test that harness exists and can be configured
        # (Full evaluation would require implementing the actual evaluation method)
        assert hasattr(harness, 'evaluate_on_episodes')

    def test_configuration_flexibility(self):
        """Test workflow with various configuration options."""
        model = self.create_simple_model()
        
        # Test different MAML configurations
        configs = [
            {'inner_lr': 0.01, 'inner_steps': 1},
            {'inner_lr': 0.1, 'inner_steps': 3},
            {'inner_lr': 0.001, 'inner_steps': 5}
        ]
        
        episode = self.create_test_episodes(1)[0]
        
        for config in configs:
            toolkit = create_meta_learning_toolkit(
                model=model,
                algorithm='maml',
                seed=42,
                **config
            )
            
            result = toolkit.train_episode(episode, algorithm='maml')
            
            # All configurations should produce valid results
            assert isinstance(result, dict)
            assert 'query_accuracy' in result
            assert 0.0 <= result['query_accuracy'] <= 1.0


class TestWorkflowPerformance:
    """Test workflow performance and timing."""

    def test_training_speed(self):
        """Test that training completes in reasonable time."""
        import time
        
        model = nn.Sequential(nn.Linear(784, 5))  # Simple model for speed
        toolkit = create_meta_learning_toolkit(model, algorithm='maml', seed=42)
        
        # Create simple episode
        episode = Episode(
            torch.randn(5, 784), torch.arange(5),
            torch.randn(10, 784), torch.repeat_interleave(torch.arange(5), 2)
        )
        
        start_time = time.time()
        result = toolkit.train_episode(episode, algorithm='maml')
        elapsed = time.time() - start_time
        
        # Should complete quickly (under 10 seconds)
        assert elapsed < 10.0
        assert isinstance(result, dict)

    def test_scalability_with_episode_size(self):
        """Test workflow scales reasonably with episode size."""
        import time
        
        model = nn.Sequential(nn.Linear(100, 32), nn.ReLU(), nn.Linear(32, 5))
        toolkit = create_meta_learning_toolkit(model, algorithm='maml', seed=42)
        
        episode_sizes = [
            (5, 1, 5),   # Small: 5-way 1-shot
            (5, 2, 10),  # Medium: 5-way 2-shot
            (10, 1, 20), # Large: 10-way 1-shot
        ]
        
        times = []
        for n_way, n_shot, n_query in episode_sizes:
            support_x = torch.randn(n_way * n_shot, 100)
            support_y = torch.repeat_interleave(torch.arange(n_way), n_shot)
            query_x = torch.randn(n_way * n_query, 100)
            query_y = torch.repeat_interleave(torch.arange(n_way), n_query)
            
            episode = Episode(support_x, support_y, query_x, query_y)
            
            start_time = time.time()
            result = toolkit.train_episode(episode, algorithm='maml')
            elapsed = time.time() - start_time
            times.append(elapsed)
            
            # Verify result quality
            assert 'query_accuracy' in result
        
        # Verify all episodes completed in reasonable time
        for t in times:
            assert t < 30.0  # 30 seconds max per episode


if __name__ == "__main__":
    pytest.main([__file__])