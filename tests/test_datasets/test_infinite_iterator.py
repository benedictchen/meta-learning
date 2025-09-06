"""
Tests for InfiniteEpisodeIterator.

Tests the thread-safe infinite iteration with adaptive sampling and prefetching.
"""

import pytest
import time
import threading
from unittest.mock import Mock, patch
import torch

from meta_learning.datasets.infinite_iterator import (
    InfiniteEpisodeIterator, AdaptiveSampler, EpisodeBuffer
)


class TestAdaptiveSampler:
    """Test adaptive sampling functionality."""
    
    def test_initialization(self):
        """Test sampler initialization."""
        sampler = AdaptiveSampler(initial_difficulty=0.5)
        
        assert sampler.difficulty == 0.5
        assert sampler.performance_history == []
        assert sampler.adaptation_rate == 0.1
    
    def test_update_difficulty_success(self):
        """Test difficulty adaptation on success."""
        sampler = AdaptiveSampler(initial_difficulty=0.5)
        
        # Simulate successful episodes
        for _ in range(5):
            sampler.update_difficulty(success=True)
        
        # Difficulty should increase
        assert sampler.difficulty > 0.5
        assert len(sampler.performance_history) == 5
    
    def test_update_difficulty_failure(self):
        """Test difficulty adaptation on failure."""
        sampler = AdaptiveSampler(initial_difficulty=0.8)
        
        # Simulate failed episodes
        for _ in range(3):
            sampler.update_difficulty(success=False)
        
        # Difficulty should decrease
        assert sampler.difficulty < 0.8
    
    def test_difficulty_bounds(self):
        """Test difficulty stays within bounds."""
        sampler = AdaptiveSampler(initial_difficulty=0.9)
        
        # Try to push above maximum
        for _ in range(20):
            sampler.update_difficulty(success=True)
        
        assert sampler.difficulty <= 1.0
        
        # Try to push below minimum
        for _ in range(20):
            sampler.update_difficulty(success=False)
        
        assert sampler.difficulty >= 0.0
    
    def test_get_sampling_parameters(self):
        """Test sampling parameter generation."""
        sampler = AdaptiveSampler(initial_difficulty=0.5)
        params = sampler.get_sampling_parameters()
        
        assert isinstance(params, dict)
        assert 'difficulty' in params
        assert 'strategy' in params
        assert params['difficulty'] == 0.5


class TestEpisodeBuffer:
    """Test episode buffer functionality."""
    
    def test_initialization(self):
        """Test buffer initialization."""
        buffer = EpisodeBuffer(capacity=100)
        
        assert buffer.capacity == 100
        assert len(buffer.buffer) == 0
        assert buffer.stats['total_added'] == 0
        assert buffer.stats['total_retrieved'] == 0
    
    def test_add_episode(self):
        """Test adding episodes to buffer."""
        buffer = EpisodeBuffer(capacity=3)
        
        # Add episodes
        episode1 = (torch.randn(5, 3), torch.randint(0, 5, (5,)))
        episode2 = (torch.randn(5, 3), torch.randint(0, 5, (5,)))
        
        buffer.add(episode1)
        buffer.add(episode2)
        
        assert len(buffer.buffer) == 2
        assert buffer.stats['total_added'] == 2
    
    def test_circular_buffer(self):
        """Test circular buffer behavior."""
        buffer = EpisodeBuffer(capacity=2)
        
        episode1 = (torch.randn(3, 2), torch.tensor([0, 1, 0]))
        episode2 = (torch.randn(3, 2), torch.tensor([1, 0, 1]))
        episode3 = (torch.randn(3, 2), torch.tensor([0, 0, 1]))
        
        buffer.add(episode1)
        buffer.add(episode2)
        buffer.add(episode3)  # Should overwrite episode1
        
        assert len(buffer.buffer) == 2
        assert buffer.stats['total_added'] == 3
    
    def test_get_episode(self):
        """Test retrieving episodes from buffer."""
        buffer = EpisodeBuffer(capacity=5)
        
        episode = (torch.randn(4, 2), torch.tensor([0, 1, 0, 1]))
        buffer.add(episode)
        
        retrieved = buffer.get()
        
        assert retrieved is not None
        assert len(retrieved) == 2  # support_x, support_y
        assert buffer.stats['total_retrieved'] == 1
    
    def test_empty_buffer(self):
        """Test getting from empty buffer."""
        buffer = EpisodeBuffer(capacity=5)
        
        episode = buffer.get()
        assert episode is None
    
    def test_thread_safety(self):
        """Test thread-safe operations."""
        buffer = EpisodeBuffer(capacity=100)
        results = []
        
        def add_episodes():
            for i in range(50):
                episode = (torch.randn(3, 2), torch.tensor([i % 3, (i+1) % 3, (i+2) % 3]))
                buffer.add(episode)
        
        def get_episodes():
            retrieved = []
            for _ in range(25):
                episode = buffer.get()
                if episode is not None:
                    retrieved.append(episode)
                time.sleep(0.001)  # Small delay
            results.extend(retrieved)
        
        # Run concurrent add and get operations
        add_thread = threading.Thread(target=add_episodes)
        get_thread = threading.Thread(target=get_episodes)
        
        add_thread.start()
        time.sleep(0.01)  # Let some episodes be added first
        get_thread.start()
        
        add_thread.join()
        get_thread.join()
        
        # Should have retrieved some episodes without errors
        assert len(results) > 0
        assert buffer.stats['total_added'] == 50


class TestInfiniteEpisodeIterator:
    """Test infinite episode iterator."""
    
    @pytest.fixture
    def episode_generator(self):
        """Create test episode generator."""
        counter = [0]
        
        def generator():
            counter[0] += 1
            return (
                torch.randn(5, 3),
                torch.randint(0, 3, (5,)),
                {'episode_id': counter[0]}
            )
        
        return generator
    
    def test_initialization(self, episode_generator):
        """Test iterator initialization."""
        iterator = InfiniteEpisodeIterator(
            episode_generator=episode_generator,
            buffer_size=50,
            prefetch_workers=1
        )
        
        assert iterator.buffer_size == 50
        assert iterator.prefetch_workers == 1
        assert iterator.adaptive_sampling is True
        assert iterator.curriculum_learning is True
    
    def test_basic_iteration(self, episode_generator):
        """Test basic iteration functionality."""
        iterator = InfiniteEpisodeIterator(
            episode_generator=episode_generator,
            buffer_size=10,
            prefetch_workers=1
        )
        
        episodes = []
        for i, episode in enumerate(iterator):
            episodes.append(episode)
            if i >= 4:  # Get 5 episodes
                break
        
        assert len(episodes) == 5
        for episode in episodes:
            assert len(episode) == 3  # support_x, support_y, metadata
            assert isinstance(episode[2], dict)
            assert 'episode_id' in episode[2]
    
    def test_prefetching(self, episode_generator):
        """Test prefetching functionality."""
        iterator = InfiniteEpisodeIterator(
            episode_generator=episode_generator,
            buffer_size=20,
            prefetch_workers=2
        )
        
        # Allow some time for prefetching
        time.sleep(0.1)
        
        # Should have episodes ready immediately
        episode = next(iter(iterator))
        assert episode is not None
        
        iterator.stop()
    
    def test_adaptive_sampling(self, episode_generator):
        """Test adaptive sampling integration."""
        iterator = InfiniteEpisodeIterator(
            episode_generator=episode_generator,
            buffer_size=10,
            adaptive_sampling=True
        )
        
        # Get some episodes and provide feedback
        episode1 = next(iter(iterator))
        iterator.provide_feedback(episode1, success=True, accuracy=0.9)
        
        episode2 = next(iter(iterator))
        iterator.provide_feedback(episode2, success=False, accuracy=0.3)
        
        # Difficulty should be adjusted
        assert hasattr(iterator, 'sampler')
        assert len(iterator.sampler.performance_history) == 2
        
        iterator.stop()
    
    def test_curriculum_learning(self, episode_generator):
        """Test curriculum learning progression."""
        iterator = InfiniteEpisodeIterator(
            episode_generator=episode_generator,
            curriculum_learning=True
        )
        
        initial_difficulty = iterator.sampler.difficulty
        
        # Simulate successful learning
        for i in range(10):
            episode = next(iter(iterator))
            iterator.provide_feedback(episode, success=True, accuracy=0.8)
        
        # Difficulty should have increased
        assert iterator.sampler.difficulty >= initial_difficulty
        
        iterator.stop()
    
    def test_statistics_tracking(self, episode_generator):
        """Test statistics collection."""
        iterator = InfiniteEpisodeIterator(
            episode_generator=episode_generator,
            buffer_size=5
        )
        
        # Get several episodes
        episodes = []
        for i, episode in enumerate(iterator):
            episodes.append(episode)
            if i >= 9:
                break
        
        stats = iterator.get_statistics()
        
        assert isinstance(stats, dict)
        assert 'episodes_generated' in stats
        assert 'buffer_stats' in stats
        assert stats['episodes_generated'] >= 10
        
        iterator.stop()
    
    def test_performance_monitoring(self, episode_generator):
        """Test performance monitoring."""
        iterator = InfiniteEpisodeIterator(
            episode_generator=episode_generator,
            buffer_size=10
        )
        
        # Generate episodes with timing
        start_time = time.time()
        episodes = []
        for i, episode in enumerate(iterator):
            episodes.append(episode)
            if i >= 19:
                break
        end_time = time.time()
        
        stats = iterator.get_statistics()
        
        # Should track generation rate
        assert 'generation_rate' in stats
        assert stats['generation_rate'] > 0
        
        # Should have reasonable performance
        total_time = end_time - start_time
        episodes_per_second = len(episodes) / total_time
        assert episodes_per_second > 1  # Should generate at least 1 episode/second
        
        iterator.stop()
    
    def test_stop_and_cleanup(self, episode_generator):
        """Test proper cleanup."""
        iterator = InfiniteEpisodeIterator(
            episode_generator=episode_generator,
            prefetch_workers=2
        )
        
        # Get a few episodes
        episode = next(iter(iterator))
        assert episode is not None
        
        # Stop iterator
        iterator.stop()
        
        # Should cleanup properly
        assert not iterator._running
    
    def test_error_handling(self):
        """Test error handling in episode generation."""
        error_count = [0]
        
        def failing_generator():
            error_count[0] += 1
            if error_count[0] <= 3:  # Fail first 3 times
                raise RuntimeError("Simulated generation error")
            return (torch.randn(3, 2), torch.tensor([0, 1, 0]))
        
        iterator = InfiniteEpisodeIterator(
            episode_generator=failing_generator,
            buffer_size=5
        )
        
        # Should recover from errors
        episode = next(iter(iterator))
        assert episode is not None
        
        iterator.stop()
    
    def test_memory_efficiency(self, episode_generator):
        """Test memory-efficient operation."""
        iterator = InfiniteEpisodeIterator(
            episode_generator=episode_generator,
            buffer_size=5,  # Small buffer
            prefetch_workers=1
        )
        
        # Generate many episodes without accumulating them
        for i, episode in enumerate(iterator):
            if i >= 99:  # Generate 100 episodes
                break
            # Don't store episodes, just iterate
        
        stats = iterator.get_statistics()
        
        # Buffer should remain small
        assert stats['buffer_stats']['current_size'] <= 5
        assert stats['episodes_generated'] >= 100
        
        iterator.stop()


class TestIntegration:
    """Integration tests for infinite episode iteration."""
    
    def test_realistic_training_loop(self):
        """Test realistic meta-learning training loop."""
        def synthetic_episode_generator():
            """Generate synthetic few-shot episodes."""
            n_way = 5
            k_shot = 1
            n_query = 15
            
            # Generate synthetic data
            support_x = torch.randn(n_way * k_shot, 128)
            support_y = torch.arange(n_way).repeat_interleave(k_shot)
            
            query_x = torch.randn(n_query, 128)
            query_y = torch.randint(0, n_way, (n_query,))
            
            return support_x, support_y, query_x, query_y
        
        iterator = InfiniteEpisodeIterator(
            episode_generator=synthetic_episode_generator,
            buffer_size=20,
            prefetch_workers=2,
            adaptive_sampling=True,
            curriculum_learning=True
        )
        
        # Simulate training loop
        total_accuracy = 0
        num_episodes = 0
        
        for i, episode in enumerate(iterator):
            if i >= 49:  # Train for 50 episodes
                break
            
            support_x, support_y, query_x, query_y = episode
            
            # Simulate model training and evaluation
            simulated_accuracy = 0.6 + 0.4 * (i / 50)  # Improving accuracy
            total_accuracy += simulated_accuracy
            num_episodes += 1
            
            # Provide feedback for curriculum learning
            success = simulated_accuracy > 0.7
            iterator.provide_feedback(episode, success=success, accuracy=simulated_accuracy)
        
        # Check training progression
        avg_accuracy = total_accuracy / num_episodes
        assert avg_accuracy > 0.6
        
        # Check curriculum progression
        final_stats = iterator.get_statistics()
        assert final_stats['episodes_generated'] >= 50
        
        iterator.stop()
    
    def test_concurrent_access(self):
        """Test concurrent access from multiple threads."""
        def simple_generator():
            return (torch.randn(10, 5), torch.randint(0, 3, (10,)))
        
        iterator = InfiniteEpisodeIterator(
            episode_generator=simple_generator,
            buffer_size=50
        )
        
        episodes_collected = []
        
        def worker_thread(worker_id, num_episodes):
            worker_episodes = []
            for i, episode in enumerate(iterator):
                worker_episodes.append((worker_id, i, episode))
                if i >= num_episodes - 1:
                    break
            episodes_collected.extend(worker_episodes)
        
        # Start multiple worker threads
        threads = []
        for worker_id in range(3):
            thread = threading.Thread(
                target=worker_thread, 
                args=(worker_id, 10)
            )
            threads.append(thread)
            thread.start()
        
        # Wait for all threads
        for thread in threads:
            thread.join()
        
        # Should have collected episodes from all workers
        assert len(episodes_collected) == 30  # 3 workers × 10 episodes each
        
        # Check worker distribution
        worker_counts = {}
        for worker_id, _, _ in episodes_collected:
            worker_counts[worker_id] = worker_counts.get(worker_id, 0) + 1
        
        assert len(worker_counts) == 3
        for count in worker_counts.values():
            assert count == 10
        
        iterator.stop()


if __name__ == "__main__":
    pytest.main([__file__])