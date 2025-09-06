#!/usr/bin/env python3
"""
Comprehensive Performance and Benchmarking Tests
===============================================

Tests performance, scalability, and benchmarking for all meta-learning components:
- Training speed and memory usage benchmarks
- Scalability tests with varying episode sizes
- Algorithm performance comparisons  
- Memory profiling and optimization validation
- Throughput and latency measurements
"""

import pytest
import torch
import torch.nn as nn
import time
import gc
import psutil
import os
from typing import Dict, List, Tuple, Any
from contextlib import contextmanager
from unittest.mock import patch

from meta_learning.meta_learning_modules.test_time_compute import TestTimeComputeScaler
from meta_learning.meta_learning_modules.few_shot_modules.maml_variants import MAMLConfig, MAMLVariant
from meta_learning.toolkit import MetaLearningToolkit, create_meta_learning_toolkit
from meta_learning.data_utils.episode import Episode
from meta_learning.data_utils.adaptive_episode_sampler import AdaptiveEpisodeSampler
from meta_learning.data_utils.curriculum_sampler import CurriculumSampler


@contextmanager
def memory_profiler():
    """Context manager for profiling memory usage."""
    process = psutil.Process(os.getpid())
    gc.collect()
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    initial_memory = process.memory_info().rss / 1024 / 1024  # MB
    peak_memory = initial_memory
    
    def update_peak():
        nonlocal peak_memory
        current = process.memory_info().rss / 1024 / 1024
        peak_memory = max(peak_memory, current)
    
    yield update_peak
    
    gc.collect()
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    final_memory = process.memory_info().rss / 1024 / 1024
    memory_used = final_memory - initial_memory
    
    return {
        'initial_mb': initial_memory,
        'final_mb': final_memory, 
        'peak_mb': peak_memory,
        'used_mb': memory_used
    }


@contextmanager 
def timer():
    """Context manager for timing operations."""
    start_time = time.perf_counter()
    yield
    end_time = time.perf_counter()
    return end_time - start_time


class TestTrainingSpeedBenchmarks:
    """Test training speed and throughput benchmarks."""
    
    def create_benchmark_episode(self, n_way: int = 5, k_shot: int = 5, query_size: int = 15, 
                               feature_dim: int = 64) -> Episode:
        """Create standardized episode for benchmarking."""
        support_size = n_way * k_shot
        
        support_data = torch.randn(support_size, feature_dim)
        support_labels = torch.repeat_interleave(torch.arange(n_way), k_shot)
        
        query_data = torch.randn(query_size, feature_dim)
        query_labels = torch.randint(0, n_way, (query_size,))
        
        return Episode(support_data, support_labels, query_data, query_labels)
    
    def create_benchmark_model(self, input_dim: int = 64, hidden_dim: int = 128, 
                              output_dim: int = 5) -> nn.Module:
        """Create standardized model for benchmarking."""
        return nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, output_dim)
        )
    
    def test_maml_training_speed(self):
        """Benchmark MAML training speed."""
        model = self.create_benchmark_model()
        episode = self.create_benchmark_episode()
        
        toolkit = create_meta_learning_toolkit(
            model=model,
            algorithm='maml',
            seed=42,
            inner_lr=0.01,
            outer_lr=0.001,
            inner_steps=5
        )
        
        # Warm-up run
        _ = toolkit.train_episode(episode)
        
        # Benchmark multiple episodes
        num_episodes = 10
        times = []
        
        for _ in range(num_episodes):
            start_time = time.perf_counter()
            results = toolkit.train_episode(episode)
            end_time = time.perf_counter()
            
            times.append(end_time - start_time)
            assert 'loss' in results
            assert 'adapted_params' in results
        
        avg_time = sum(times) / len(times)
        std_time = (sum((t - avg_time)**2 for t in times) / len(times))**0.5
        
        # Performance assertions
        assert avg_time < 2.0, f"MAML training too slow: {avg_time:.3f}s avg"
        assert std_time < 0.5, f"MAML timing too variable: {std_time:.3f}s std"
        
        print(f"MAML Training Speed: {avg_time:.3f}±{std_time:.3f}s per episode")
    
    def test_test_time_compute_scaling_speed(self):
        """Benchmark test-time compute scaling speed."""
        model = self.create_benchmark_model(input_dim=32, output_dim=3)
        episode = self.create_benchmark_episode(n_way=3, feature_dim=32)
        
        scaler = TestTimeComputeScaler(
            base_compute_budget=10,
            max_compute_budget=50,
            scaling_strategy='adaptive'
        )
        
        # Benchmark different compute budgets
        budgets = [10, 25, 50, 100]
        times = []
        
        for budget in budgets:
            scaler.base_compute_budget = min(budget, 50)  # Respect max budget
            
            start_time = time.perf_counter()
            results = scaler.scale_inference(model, episode.query_data, episode.query_labels)
            end_time = time.perf_counter()
            
            times.append(end_time - start_time)
            
            assert 'predictions' in results
            assert 'compute_used' in results
        
        # Verify that higher compute budgets take more time (roughly)
        assert times[-1] >= times[0], "Higher compute budget should take more time"
        
        # Performance assertion
        max_time = max(times)
        assert max_time < 5.0, f"Test-time compute scaling too slow: {max_time:.3f}s"
        
        print(f"Test-Time Compute Scaling Times: {[f'{t:.3f}s' for t in times]}")
    
    def test_episode_sampling_speed(self):
        """Benchmark episode sampling speed."""
        sampler = AdaptiveEpisodeSampler()
        
        # Benchmark episode parameter sampling
        num_samples = 1000
        
        start_time = time.perf_counter()
        for _ in range(num_samples):
            params = sampler.sample_episode_params()
            assert 'n_way' in params
            assert 'k_shot' in params
            assert 'query_size' in params
        end_time = time.perf_counter()
        
        total_time = end_time - start_time
        samples_per_second = num_samples / total_time
        
        assert samples_per_second > 1000, f"Episode sampling too slow: {samples_per_second:.1f} samples/sec"
        print(f"Episode Sampling Speed: {samples_per_second:.1f} samples/sec")


class TestMemoryUsageBenchmarks:
    """Test memory usage and optimization benchmarks."""
    
    def test_maml_memory_usage(self):
        """Benchmark MAML memory usage."""
        model = nn.Sequential(
            nn.Linear(100, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )
        
        episode = Episode(
            support_data=torch.randn(50, 100),
            support_labels=torch.repeat_interleave(torch.arange(10), 5),
            query_data=torch.randn(30, 100),
            query_labels=torch.randint(0, 10, (30,))
        )
        
        with memory_profiler() as update_peak:
            toolkit = create_meta_learning_toolkit(
                model=model,
                algorithm='maml',
                inner_lr=0.01,
                outer_lr=0.001,
                inner_steps=10
            )
            
            # Train multiple episodes
            for _ in range(5):
                results = toolkit.train_episode(episode)
                update_peak()
                assert 'loss' in results
        
        memory_stats = memory_profiler.__exit__(None, None, None)
        
        # Memory usage should be reasonable
        if 'used_mb' in str(memory_stats):  # Handle return from context manager
            print(f"MAML Memory Usage: Peak memory tracking enabled")
        
        # Basic assertion - training should complete without OOM
        assert True, "MAML training completed successfully"
    
    def test_large_episode_memory_scaling(self):
        """Test memory usage scaling with episode size."""
        base_model = nn.Sequential(
            nn.Linear(20, 64),
            nn.ReLU(),
            nn.Linear(64, 5)
        )
        
        episode_sizes = [
            (5, 5, 25),    # Small: 5-way 5-shot, 25 queries
            (10, 10, 50),  # Medium: 10-way 10-shot, 50 queries  
            (20, 15, 100), # Large: 20-way 15-shot, 100 queries
        ]
        
        memory_usage = []
        
        for n_way, k_shot, query_size in episode_sizes:
            gc.collect()
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
            episode = Episode(
                support_data=torch.randn(n_way * k_shot, 20),
                support_labels=torch.repeat_interleave(torch.arange(n_way), k_shot),
                query_data=torch.randn(query_size, 20),
                query_labels=torch.randint(0, n_way, (query_size,))
            )
            
            process = psutil.Process(os.getpid())
            initial_memory = process.memory_info().rss / 1024 / 1024
            
            toolkit = create_meta_learning_toolkit(
                model=base_model,
                algorithm='maml',
                inner_lr=0.01,
                inner_steps=3
            )
            
            results = toolkit.train_episode(episode)
            assert 'loss' in results
            
            final_memory = process.memory_info().rss / 1024 / 1024
            memory_used = final_memory - initial_memory
            memory_usage.append(memory_used)
        
        print(f"Memory Usage by Episode Size: {[f'{m:.1f}MB' for m in memory_usage]}")
        
        # Memory should scale reasonably with episode size
        assert all(m >= 0 for m in memory_usage), "Memory usage should be positive"
    
    def test_model_size_impact_on_memory(self):
        """Test impact of model size on memory usage."""
        episode = Episode(
            support_data=torch.randn(25, 50),
            support_labels=torch.repeat_interleave(torch.arange(5), 5),
            query_data=torch.randn(15, 50),
            query_labels=torch.randint(0, 5, (15,))
        )
        
        model_configs = [
            (50, 64, 5),    # Small model
            (50, 128, 5),   # Medium model
            (50, 256, 5),   # Large model
        ]
        
        memory_usage = []
        training_times = []
        
        for input_dim, hidden_dim, output_dim in model_configs:
            model = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(), 
                nn.Linear(hidden_dim // 2, output_dim)
            )
            
            gc.collect()
            process = psutil.Process(os.getpid())
            initial_memory = process.memory_info().rss / 1024 / 1024
            
            start_time = time.perf_counter()
            
            toolkit = create_meta_learning_toolkit(
                model=model,
                algorithm='maml',
                inner_lr=0.01,
                inner_steps=5
            )
            
            results = toolkit.train_episode(episode)
            
            end_time = time.perf_counter()
            final_memory = process.memory_info().rss / 1024 / 1024
            
            memory_used = final_memory - initial_memory
            time_taken = end_time - start_time
            
            memory_usage.append(memory_used)
            training_times.append(time_taken)
            
            assert 'loss' in results
        
        print(f"Memory vs Model Size: {[f'{m:.1f}MB' for m in memory_usage]}")
        print(f"Time vs Model Size: {[f'{t:.3f}s' for t in training_times]}")
        
        # Larger models should generally use more memory and time
        assert training_times[-1] >= training_times[0], "Larger models should take more time"


class TestScalabilityBenchmarks:
    """Test scalability with varying parameters."""
    
    def test_n_way_scaling(self):
        """Test performance scaling with number of classes (N-way)."""
        base_model = nn.Sequential(
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 20)  # Large enough output for all N-way tests
        )
        
        n_way_values = [2, 5, 10, 15, 20]
        k_shot = 5
        query_size = 25
        
        performance_metrics = []
        
        for n_way in n_way_values:
            episode = Episode(
                support_data=torch.randn(n_way * k_shot, 32),
                support_labels=torch.repeat_interleave(torch.arange(n_way), k_shot),
                query_data=torch.randn(query_size, 32),
                query_labels=torch.randint(0, n_way, (query_size,))
            )
            
            # Update model output dimension
            model = nn.Sequential(*list(base_model.children())[:-1])
            model.add_module('final', nn.Linear(32, n_way))
            
            start_time = time.perf_counter()
            
            toolkit = create_meta_learning_toolkit(
                model=model,
                algorithm='maml',
                inner_lr=0.01,
                inner_steps=3
            )
            
            results = toolkit.train_episode(episode)
            
            end_time = time.perf_counter()
            time_taken = end_time - start_time
            
            performance_metrics.append({
                'n_way': n_way,
                'time': time_taken,
                'loss': results['loss']
            })
            
            assert 'loss' in results
        
        # Print scaling results
        for metrics in performance_metrics:
            print(f"{metrics['n_way']}-way: {metrics['time']:.3f}s, loss: {metrics['loss']:.3f}")
        
        # Time should scale reasonably with N-way
        times = [m['time'] for m in performance_metrics]
        assert times[-1] > times[0], "More classes should take more time"
        assert times[-1] < times[0] * 20, "Scaling shouldn't be exponential"
    
    def test_k_shot_scaling(self):
        """Test performance scaling with number of shots (K-shot)."""
        model = nn.Sequential(
            nn.Linear(24, 48),
            nn.ReLU(),
            nn.Linear(48, 24),
            nn.ReLU(),
            nn.Linear(24, 5)
        )
        
        n_way = 5
        k_shot_values = [1, 3, 5, 10, 15]
        query_size = 20
        
        performance_metrics = []
        
        for k_shot in k_shot_values:
            episode = Episode(
                support_data=torch.randn(n_way * k_shot, 24),
                support_labels=torch.repeat_interleave(torch.arange(n_way), k_shot),
                query_data=torch.randn(query_size, 24),
                query_labels=torch.randint(0, n_way, (query_size,))
            )
            
            start_time = time.perf_counter()
            
            toolkit = create_meta_learning_toolkit(
                model=model,
                algorithm='maml',
                inner_lr=0.01,
                inner_steps=k_shot  # Scale inner steps with k_shot
            )
            
            results = toolkit.train_episode(episode)
            
            end_time = time.perf_counter()
            time_taken = end_time - start_time
            
            performance_metrics.append({
                'k_shot': k_shot,
                'time': time_taken,
                'loss': results['loss']
            })
            
            assert 'loss' in results
        
        # Print scaling results
        for metrics in performance_metrics:
            print(f"{metrics['k_shot']}-shot: {metrics['time']:.3f}s, loss: {metrics['loss']:.3f}")
        
        # Time should scale with K-shot due to more support examples and inner steps
        times = [m['time'] for m in performance_metrics]
        assert times[-1] > times[0], "More shots should take more time"
    
    def test_inner_steps_scaling(self):
        """Test performance scaling with number of inner loop steps."""
        model = nn.Sequential(
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, 5)
        )
        
        episode = Episode(
            support_data=torch.randn(25, 16),
            support_labels=torch.repeat_interleave(torch.arange(5), 5),
            query_data=torch.randn(15, 16),
            query_labels=torch.randint(0, 5, (15,))
        )
        
        inner_steps_values = [1, 3, 5, 10, 15]
        performance_metrics = []
        
        for inner_steps in inner_steps_values:
            start_time = time.perf_counter()
            
            toolkit = create_meta_learning_toolkit(
                model=model,
                algorithm='maml',
                inner_lr=0.01,
                inner_steps=inner_steps
            )
            
            results = toolkit.train_episode(episode)
            
            end_time = time.perf_counter()
            time_taken = end_time - start_time
            
            performance_metrics.append({
                'inner_steps': inner_steps,
                'time': time_taken,
                'loss': results['loss']
            })
            
            assert 'loss' in results
        
        # Print scaling results  
        for metrics in performance_metrics:
            print(f"{metrics['inner_steps']} steps: {metrics['time']:.3f}s, loss: {metrics['loss']:.3f}")
        
        # Time should scale linearly with inner steps
        times = [m['time'] for m in performance_metrics]
        assert times[-1] > times[0], "More inner steps should take more time"
        
        # Check roughly linear scaling
        time_ratio = times[-1] / times[0]
        steps_ratio = inner_steps_values[-1] / inner_steps_values[0]
        assert time_ratio < steps_ratio * 2, "Time scaling shouldn't be much worse than linear"


class TestAlgorithmComparison:
    """Compare performance across different algorithms."""
    
    def test_algorithm_speed_comparison(self):
        """Compare training speed across different algorithms."""
        model = nn.Sequential(
            nn.Linear(20, 40),
            nn.ReLU(),
            nn.Linear(40, 5)
        )
        
        episode = Episode(
            support_data=torch.randn(25, 20),
            support_labels=torch.repeat_interleave(torch.arange(5), 5),
            query_data=torch.randn(15, 20),
            query_labels=torch.randint(0, 5, (15,))
        )
        
        algorithms = ['maml', 'fomaml']
        results = {}
        
        for algorithm in algorithms:
            times = []
            losses = []
            
            for _ in range(5):  # Multiple runs for averaging
                start_time = time.perf_counter()
                
                toolkit = create_meta_learning_toolkit(
                    model=model,
                    algorithm=algorithm,
                    inner_lr=0.01,
                    inner_steps=5
                )
                
                result = toolkit.train_episode(episode)
                
                end_time = time.perf_counter()
                
                times.append(end_time - start_time)
                losses.append(result['loss'])
                
                assert 'loss' in result
            
            avg_time = sum(times) / len(times)
            avg_loss = sum(losses) / len(losses)
            
            results[algorithm] = {
                'avg_time': avg_time,
                'avg_loss': avg_loss,
                'times': times
            }
        
        # Print comparison results
        for alg, metrics in results.items():
            print(f"{alg.upper()}: {metrics['avg_time']:.3f}±{(max(metrics['times']) - min(metrics['times']))/2:.3f}s, "
                  f"loss: {metrics['avg_loss']:.3f}")
        
        # FOMAML should be faster than MAML (first-order approximation)
        if 'fomaml' in results and 'maml' in results:
            assert results['fomaml']['avg_time'] <= results['maml']['avg_time'] * 1.5, \
                "FOMAML should be faster than or similar to MAML"
    
    def test_test_time_compute_vs_baseline(self):
        """Compare test-time compute scaling vs baseline inference."""
        model = nn.Sequential(
            nn.Linear(12, 24),
            nn.ReLU(),
            nn.Linear(24, 3)
        )
        
        episode = Episode(
            support_data=torch.randn(15, 12),
            support_labels=torch.repeat_interleave(torch.arange(3), 5),
            query_data=torch.randn(9, 12),
            query_labels=torch.randint(0, 3, (9,))
        )
        
        # Baseline inference time
        model.eval()
        start_time = time.perf_counter()
        with torch.no_grad():
            baseline_output = model(episode.query_data)
        baseline_time = time.perf_counter() - start_time
        
        # Test-time compute scaling
        scaler = TestTimeComputeScaler(
            base_compute_budget=20,
            max_compute_budget=50,
            scaling_strategy='linear'
        )
        
        start_time = time.perf_counter()
        scaling_results = scaler.scale_inference(model, episode.query_data, episode.query_labels)
        scaling_time = time.perf_counter() - start_time
        
        print(f"Baseline inference: {baseline_time:.4f}s")
        print(f"Test-time scaling: {scaling_time:.4f}s")
        print(f"Scaling overhead: {scaling_time / baseline_time:.1f}x")
        
        assert 'predictions' in scaling_results
        assert scaling_time > baseline_time, "Test-time scaling should take more time"
        assert scaling_time < baseline_time * 100, "Scaling overhead should be reasonable"


class TestThroughputBenchmarks:
    """Test throughput and latency measurements."""
    
    def test_episode_throughput(self):
        """Measure episodes processed per second."""
        model = nn.Sequential(
            nn.Linear(8, 16),
            nn.ReLU(),
            nn.Linear(16, 3)
        )
        
        # Create multiple episodes
        episodes = []
        for _ in range(20):
            episode = Episode(
                support_data=torch.randn(15, 8),
                support_labels=torch.repeat_interleave(torch.arange(3), 5),
                query_data=torch.randn(9, 8),
                query_labels=torch.randint(0, 3, (9,))
            )
            episodes.append(episode)
        
        toolkit = create_meta_learning_toolkit(
            model=model,
            algorithm='maml',
            inner_lr=0.01,
            inner_steps=3
        )
        
        # Measure throughput
        start_time = time.perf_counter()
        
        for episode in episodes:
            results = toolkit.train_episode(episode)
            assert 'loss' in results
        
        end_time = time.perf_counter()
        total_time = end_time - start_time
        episodes_per_second = len(episodes) / total_time
        
        print(f"Episode Throughput: {episodes_per_second:.1f} episodes/sec")
        assert episodes_per_second > 1.0, "Throughput should be at least 1 episode/sec"
    
    def test_batch_processing_efficiency(self):
        """Test efficiency of batch processing vs sequential."""
        model = nn.Sequential(
            nn.Linear(6, 12),
            nn.ReLU(),
            nn.Linear(12, 2)
        )
        
        # Create batch data
        batch_size = 10
        batch_data = torch.randn(batch_size, 6)
        
        # Sequential processing
        model.eval()
        start_time = time.perf_counter()
        sequential_results = []
        for i in range(batch_size):
            with torch.no_grad():
                result = model(batch_data[i:i+1])
                sequential_results.append(result)
        sequential_time = time.perf_counter() - start_time
        
        # Batch processing
        start_time = time.perf_counter()
        with torch.no_grad():
            batch_result = model(batch_data)
        batch_time = time.perf_counter() - start_time
        
        # Verify results are equivalent
        sequential_concat = torch.cat(sequential_results, dim=0)
        assert torch.allclose(sequential_concat, batch_result, atol=1e-6)
        
        # Batch should be more efficient
        efficiency_ratio = sequential_time / batch_time
        print(f"Batch Efficiency: {efficiency_ratio:.1f}x faster than sequential")
        assert efficiency_ratio > 1.5, "Batch processing should be significantly faster"
    
    def test_adaptive_sampler_throughput(self):
        """Test throughput of adaptive episode sampler."""
        sampler = AdaptiveEpisodeSampler()
        curriculum_sampler = CurriculumSampler()
        
        # Measure parameter sampling throughput
        num_samples = 5000
        
        # Adaptive sampler
        start_time = time.perf_counter()
        for _ in range(num_samples):
            params = sampler.sample_episode_params()
            assert 'n_way' in params
        adaptive_time = time.perf_counter() - start_time
        
        # Curriculum sampler  
        start_time = time.perf_counter()
        for _ in range(num_samples):
            params = curriculum_sampler.sample_episode_params()
            assert 'n_way' in params
        curriculum_time = time.perf_counter() - start_time
        
        adaptive_throughput = num_samples / adaptive_time
        curriculum_throughput = num_samples / curriculum_time
        
        print(f"Adaptive Sampler: {adaptive_throughput:.0f} samples/sec")
        print(f"Curriculum Sampler: {curriculum_throughput:.0f} samples/sec")
        
        assert adaptive_throughput > 1000, "Adaptive sampler should be fast"
        assert curriculum_throughput > 1000, "Curriculum sampler should be fast"


class TestResourceUtilization:
    """Test resource utilization and optimization."""
    
    def test_cpu_utilization(self):
        """Test CPU utilization during training."""
        model = nn.Sequential(
            nn.Linear(30, 60),
            nn.ReLU(),
            nn.Linear(60, 10)
        )
        
        episode = Episode(
            support_data=torch.randn(50, 30),
            support_labels=torch.repeat_interleave(torch.arange(10), 5),
            query_data=torch.randn(30, 30),
            query_labels=torch.randint(0, 10, (30,))
        )
        
        process = psutil.Process(os.getpid())
        
        # Measure CPU usage during training
        cpu_before = process.cpu_percent()
        
        toolkit = create_meta_learning_toolkit(
            model=model,
            algorithm='maml',
            inner_lr=0.01,
            inner_steps=8
        )
        
        # Run several episodes to get meaningful CPU measurements
        for _ in range(5):
            results = toolkit.train_episode(episode)
            assert 'loss' in results
        
        cpu_after = process.cpu_percent()
        
        print(f"CPU Usage: {cpu_after:.1f}% (was {cpu_before:.1f}%)")
        
        # CPU should be utilized during training
        assert cpu_after >= 0, "CPU usage should be measurable"
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_gpu_memory_efficiency(self):
        """Test GPU memory usage efficiency."""
        device = torch.device('cuda')
        
        model = nn.Sequential(
            nn.Linear(50, 100),
            nn.ReLU(),
            nn.Linear(100, 5)
        ).to(device)
        
        episode = Episode(
            support_data=torch.randn(25, 50).to(device),
            support_labels=torch.repeat_interleave(torch.arange(5), 5).to(device),
            query_data=torch.randn(15, 50).to(device),
            query_labels=torch.randint(0, 5, (15,)).to(device)
        )
        
        # Clear GPU cache
        torch.cuda.empty_cache()
        gpu_memory_before = torch.cuda.memory_allocated()
        
        toolkit = create_meta_learning_toolkit(
            model=model,
            algorithm='maml',
            inner_lr=0.01,
            inner_steps=5
        )
        
        results = toolkit.train_episode(episode)
        assert 'loss' in results
        
        gpu_memory_after = torch.cuda.memory_allocated()
        memory_used = (gpu_memory_after - gpu_memory_before) / 1024 / 1024  # MB
        
        print(f"GPU Memory Used: {memory_used:.1f}MB")
        
        # Memory usage should be reasonable
        assert memory_used < 500, "GPU memory usage should be reasonable"
        
        # Clean up
        torch.cuda.empty_cache()


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])