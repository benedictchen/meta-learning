"""
Performance and stress tests for meta-learning modules.

Tests performance characteristics, memory usage, and behavior under load.
"""

import pytest
import torch
import time
import gc
import threading
from concurrent.futures import ThreadPoolExecutor
import psutil
import os
from unittest.mock import Mock

from meta_learning.validation import (
    validate_episode_tensors, validate_complete_config, 
    validate_few_shot_configuration
)
from meta_learning.warnings_system import (
    MetaLearningWarnings, warn_if_suboptimal_config
)
from meta_learning.error_recovery import (
    ErrorRecoveryManager, create_robust_episode, safe_evaluate,
    handle_numerical_instability, with_retry
)
from meta_learning.evaluation.metrics import Accuracy, CalibrationCalculator


class TestValidationPerformance:
    """Test performance characteristics of validation functions."""
    
    def test_episode_validation_scaling(self):
        """Test episode validation performance with different sizes."""
        sizes = [(100, 64), (1000, 128), (5000, 256)]
        times = []
        
        for n_samples, n_features in sizes:
            support_x = torch.randn(n_samples, n_features)
            support_y = torch.arange(n_samples // 10).repeat(10)
            query_x = torch.randn(n_samples * 2, n_features)
            query_y = torch.arange(n_samples // 10).repeat(20)
            
            start_time = time.time()
            validate_episode_tensors(support_x, support_y, query_x, query_y)
            end_time = time.time()
            
            elapsed = end_time - start_time
            times.append(elapsed)
            
            # Validation should be fast even for large episodes
            assert elapsed < 0.1, f"Validation took {elapsed:.3f}s for {n_samples} samples"
        
        # Should scale reasonably (not exponentially)
        assert times[1] / times[0] < 10  # 10x samples shouldn't take 10x+ time
        assert times[2] / times[1] < 5   # 5x samples shouldn't take 5x+ time
    
    def test_validation_memory_overhead(self):
        """Test validation doesn't create significant memory overhead."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available for memory testing")
        
        device = torch.device('cuda')
        torch.cuda.empty_cache()
        
        # Create large tensors on GPU
        support_x = torch.randn(10000, 512, device=device)
        support_y = torch.arange(1000, device=device).repeat(10)
        query_x = torch.randn(20000, 512, device=device)
        query_y = torch.arange(1000, device=device).repeat(20)
        
        initial_memory = torch.cuda.memory_allocated(device)
        
        # Run validation
        validate_episode_tensors(support_x, support_y, query_x, query_y)
        
        final_memory = torch.cuda.memory_allocated(device)
        memory_overhead = final_memory - initial_memory
        
        # Should not allocate significant additional memory
        assert memory_overhead < 1024 * 1024  # Less than 1MB overhead
    
    def test_concurrent_validation(self):
        """Test validation under concurrent access."""
        def validate_worker(worker_id):
            """Worker function for concurrent validation."""
            results = []
            for i in range(10):
                support_x = torch.randn(50, 64)
                support_y = torch.arange(5).repeat(10)
                query_x = torch.randn(100, 64)
                query_y = torch.arange(5).repeat(20)
                
                start_time = time.time()
                validate_episode_tensors(support_x, support_y, query_x, query_y)
                end_time = time.time()
                
                results.append(end_time - start_time)
            return results
        
        # Run multiple workers concurrently
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(validate_worker, i) for i in range(4)]
            all_results = [future.result() for future in futures]
        
        # All workers should complete successfully
        assert len(all_results) == 4
        
        # Average time per validation should be reasonable
        all_times = [time for worker_times in all_results for time in worker_times]
        average_time = sum(all_times) / len(all_times)
        assert average_time < 0.01  # Less than 10ms per validation
    
    def test_validation_cpu_usage(self):
        """Test validation doesn't consume excessive CPU."""
        process = psutil.Process(os.getpid())
        
        # Measure CPU usage during intensive validation
        cpu_before = process.cpu_percent()
        
        # Run many validations
        for _ in range(100):
            support_x = torch.randn(100, 128)
            support_y = torch.arange(10).repeat(10)
            query_x = torch.randn(200, 128)
            query_y = torch.arange(10).repeat(20)
            validate_episode_tensors(support_x, support_y, query_x, query_y)
        
        cpu_after = process.cpu_percent()
        
        # CPU usage should not spike excessively
        # Note: This is a rough check as CPU measurement can be noisy
        assert cpu_after < 90  # Less than 90% CPU usage


class TestWarningsSystemPerformance:
    """Test performance of warnings system."""
    
    def test_warning_system_overhead(self):
        """Test warning system doesn't add significant overhead."""
        warning_system = MetaLearningWarnings(enabled=True)
        
        # Test with warnings enabled
        start_time = time.time()
        for _ in range(1000):
            warning_system.warn_if_suboptimal_few_shot(5, 2, 15)  # No warnings
        enabled_time = time.time() - start_time
        
        # Test with warnings disabled
        warning_system.disable()
        start_time = time.time()
        for _ in range(1000):
            warning_system.warn_if_suboptimal_few_shot(5, 2, 15)  # No warnings
        disabled_time = time.time() - start_time
        
        # Overhead should be minimal when warnings are triggered
        assert enabled_time < 0.5   # Less than 500ms for 1000 calls
        assert disabled_time < 0.1  # Less than 100ms when disabled
        
        # Enabled should not be dramatically slower than disabled
        assert enabled_time / disabled_time < 10
    
    def test_warning_deduplication_performance(self):
        """Test warning deduplication doesn't degrade performance."""
        warning_system = MetaLearningWarnings(enabled=True)
        warning_system.reset()
        
        # Time first warning (should be emitted)
        start_time = time.time()
        warning_system.warn_if_suboptimal_few_shot(25, 1)  # Triggers warning
        first_time = time.time() - start_time
        
        # Time subsequent identical warnings (should be deduplicated)
        start_time = time.time()
        for _ in range(100):
            warning_system.warn_if_suboptimal_few_shot(25, 1)  # Same warning
        dedup_time = time.time() - start_time
        
        # Deduplication should be fast
        assert dedup_time < 0.1  # Less than 100ms for 100 duplicate warnings
        assert dedup_time / 100 < first_time * 2  # Per-call overhead should be minimal
    
    def test_warning_system_memory_usage(self):
        """Test warning system doesn't leak memory."""
        warning_system = MetaLearningWarnings(enabled=True)
        
        # Get initial memory usage
        gc.collect()
        initial_memory = psutil.Process(os.getpid()).memory_info().rss
        
        # Generate many different warnings
        for i in range(1000):
            warning_system.warn_once(f"test_key_{i}", f"Test message {i}")
        
        # Force garbage collection
        gc.collect()
        final_memory = psutil.Process(os.getpid()).memory_info().rss
        
        memory_increase = final_memory - initial_memory
        
        # Memory increase should be reasonable (less than 10MB for 1000 warnings)
        assert memory_increase < 10 * 1024 * 1024  # 10MB limit


class TestErrorRecoveryPerformance:
    """Test performance of error recovery mechanisms."""
    
    def test_robust_episode_creation_performance(self):
        """Test robust episode creation performance."""
        sizes = [(5, 1, 15, 64), (10, 5, 50, 128), (20, 10, 100, 256)]
        
        for n_way, k_shot, n_query, feature_dim in sizes:
            start_time = time.time()
            
            episode = create_robust_episode(n_way, k_shot, n_query, feature_dim)
            
            end_time = time.time()
            elapsed = end_time - start_time
            
            # Episode creation should be fast
            assert elapsed < 0.5, f"Episode creation took {elapsed:.3f}s"
            
            # Verify episode is correctly sized
            expected_support = n_way * k_shot
            expected_query = n_way * n_query
            
            assert episode["support_x"].shape == (expected_support, feature_dim)
            assert episode["query_x"].shape == (expected_query, feature_dim)
    
    def test_error_recovery_retry_performance(self):
        """Test retry decorator performance."""
        call_count = [0]
        
        @with_retry(max_attempts=3, delay=0.01)
        def flaky_function():
            call_count[0] += 1
            if call_count[0] < 3:
                raise ValueError("Temporary failure")
            return "success"
        
        start_time = time.time()
        result = flaky_function()
        end_time = time.time()
        
        elapsed = end_time - start_time
        
        # Should succeed after retries
        assert result == "success"
        assert call_count[0] == 3
        
        # Should complete reasonably quickly (2 retries with 0.01s delay each)
        assert elapsed < 0.5  # Should be much less than 500ms
        assert elapsed >= 0.02  # But at least 2 delays (20ms)
    
    def test_numerical_instability_handling_performance(self):
        """Test numerical instability handling performance."""
        # Create tensors with various numerical issues
        test_cases = [
            torch.tensor([[1.0, 2.0], [3.0, 4.0]]),  # Clean tensor
            torch.tensor([[float('nan'), 2.0], [3.0, 4.0]]),  # NaN
            torch.tensor([[float('inf'), 2.0], [3.0, 4.0]]),  # Inf
            torch.tensor([[float('nan'), float('inf')], [3.0, 4.0]]),  # Mixed
        ]
        
        times = []
        for tensor in test_cases:
            start_time = time.time()
            cleaned = handle_numerical_instability(tensor)
            end_time = time.time()
            
            elapsed = end_time - start_time
            times.append(elapsed)
            
            # Should be very fast
            assert elapsed < 0.01  # Less than 10ms
            
            # Result should be clean
            assert not torch.isnan(cleaned).any()
            assert not torch.isinf(cleaned).any()
        
        # Performance should be consistent regardless of input quality
        max_time = max(times)
        min_time = min(times)
        assert max_time / min_time < 10  # Within 10x of each other
    
    def test_safe_evaluate_performance(self):
        """Test safe evaluation performance with various failure rates."""
        def create_mock_model(failure_rate):
            """Create mock model with specified failure rate."""
            call_count = [0]
            
            def mock_forward(*args):
                call_count[0] += 1
                if call_count[0] % int(1 / failure_rate) == 0:
                    raise RuntimeError("Simulated failure")
                return torch.randn(5, 3)
            
            mock = Mock()
            mock.side_effect = mock_forward
            return mock
        
        # Create test episodes
        episodes = [
            {
                "support_x": torch.randn(6, 64),
                "support_y": torch.tensor([0, 0, 1, 1, 2, 2]),
                "query_x": torch.randn(5, 64),
                "query_y": torch.tensor([0, 1, 2, 0, 1])
            }
            for _ in range(20)
        ]
        
        failure_rates = [0.0, 0.1, 0.3, 0.5]
        
        for failure_rate in failure_rates:
            mock_model = create_mock_model(failure_rate)
            
            start_time = time.time()
            results = safe_evaluate(mock_model, episodes)
            end_time = time.time()
            
            elapsed = end_time - start_time
            
            # Should complete reasonably quickly even with failures
            assert elapsed < 5.0, f"Evaluation took {elapsed:.3f}s with {failure_rate} failure rate"
            
            # Should have some results
            assert "accuracy" in results
            assert results["total_episodes"] == 20


class TestEvaluationMetricsPerformance:
    """Test performance of evaluation metrics."""
    
    def test_accuracy_calculation_performance(self):
        """Test accuracy calculation performance with large batches."""
        batch_sizes = [100, 1000, 10000]
        n_classes = 10
        
        accuracy_calc = Accuracy()
        
        for batch_size in batch_sizes:
            # Generate random predictions and targets
            predictions = torch.randn(batch_size, n_classes)
            targets = torch.randint(0, n_classes, (batch_size,))
            
            start_time = time.time()
            result = accuracy_calc.compute(predictions, targets)
            end_time = time.time()
            
            elapsed = end_time - start_time
            
            # Should be fast even for large batches
            assert elapsed < 0.5, f"Accuracy calculation took {elapsed:.3f}s for {batch_size} samples"
            
            # Results should be valid
            assert 0.0 <= result.mean <= 1.0
            assert len(result.confidence_interval) == 2
    
    def test_calibration_calculation_performance(self):
        """Test calibration calculation performance."""
        batch_sizes = [1000, 5000, 10000]
        
        calibration_calc = CalibrationCalculator(n_bins=10)
        
        for batch_size in batch_sizes:
            # Generate realistic probabilities and targets
            probabilities = torch.softmax(torch.randn(batch_size, 5), dim=1)
            targets = torch.randint(0, 5, (batch_size,))
            
            start_time = time.time()
            result = calibration_calc.compute(probabilities, targets)
            end_time = time.time()
            
            elapsed = end_time - start_time
            
            # Should complete reasonably quickly
            assert elapsed < 2.0, f"Calibration calculation took {elapsed:.3f}s for {batch_size} samples"
            
            # Results should be valid
            assert hasattr(result, 'ece')  # Expected Calibration Error
            assert hasattr(result, 'mce')  # Maximum Calibration Error
    
    def test_metrics_memory_efficiency(self):
        """Test metrics calculations don't consume excessive memory."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available for memory testing")
        
        device = torch.device('cuda')
        torch.cuda.empty_cache()
        
        # Create large tensors on GPU
        predictions = torch.randn(50000, 20, device=device)
        targets = torch.randint(0, 20, (50000,), device=device)
        probabilities = torch.softmax(predictions, dim=1)
        
        initial_memory = torch.cuda.memory_allocated(device)
        
        # Run metrics calculations
        accuracy_calc = Accuracy()
        accuracy_result = accuracy_calc.compute(predictions, targets)
        
        calibration_calc = CalibrationCalculator(n_bins=15)
        calibration_result = calibration_calc.compute(probabilities, targets)
        
        final_memory = torch.cuda.memory_allocated(device)
        memory_overhead = final_memory - initial_memory
        
        # Should not allocate excessive additional memory
        assert memory_overhead < 50 * 1024 * 1024  # Less than 50MB overhead


class TestStressAndLoad:
    """Stress tests and load testing."""
    
    def test_concurrent_module_access(self):
        """Test concurrent access to all modules."""
        def worker_validation():
            for _ in range(20):
                config = {"n_way": 5, "k_shot": 2, "n_query": 15}
                validate_complete_config(config)
        
        def worker_warnings():
            warning_system = MetaLearningWarnings(enabled=True)
            for _ in range(20):
                warning_system.warn_if_suboptimal_few_shot(5, 2, 15)
        
        def worker_recovery():
            for _ in range(10):
                episode = create_robust_episode(3, 1, 5, 32)
                assert episode["support_x"].shape[0] == 3
        
        def worker_metrics():
            accuracy_calc = Accuracy()
            for _ in range(20):
                preds = torch.randn(10, 3)
                targets = torch.randint(0, 3, (10,))
                result = accuracy_calc.compute(preds, targets)
                assert 0.0 <= result.mean <= 1.0
        
        # Run all workers concurrently
        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = []
            for worker in [worker_validation, worker_warnings, worker_recovery, worker_metrics]:
                for _ in range(2):  # 2 instances of each worker
                    futures.append(executor.submit(worker))
            
            # Wait for all workers to complete
            start_time = time.time()
            for future in futures:
                future.result()  # Will raise exception if worker failed
            end_time = time.time()
            
            elapsed = end_time - start_time
            
            # All workers should complete within reasonable time
            assert elapsed < 10.0, f"Concurrent execution took {elapsed:.3f}s"
    
    def test_memory_stress(self):
        """Test behavior under memory stress."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available for memory stress testing")
        
        device = torch.device('cuda')
        torch.cuda.empty_cache()
        
        # Create many large episodes to stress memory
        episodes = []
        try:
            for i in range(10):
                episode = create_robust_episode(
                    n_way=10, k_shot=5, n_query=50, 
                    feature_dim=1024,
                    device=device
                )
                episodes.append(episode)
                
                # Validate each episode
                validate_episode_tensors(
                    episode["support_x"], episode["support_y"],
                    episode["query_x"], episode["query_y"]
                )
        
        except RuntimeError as e:
            if "out of memory" in str(e):
                pytest.skip("GPU out of memory - expected for stress test")
            else:
                raise
        
        # If we got here, all episodes were created successfully
        assert len(episodes) == 10
        
        # Clean up
        del episodes
        torch.cuda.empty_cache()
    
    def test_long_running_stability(self):
        """Test stability over long-running operations."""
        start_time = time.time()
        iterations = 0
        max_runtime = 5.0  # 5 seconds
        
        warning_system = MetaLearningWarnings(enabled=True)
        recovery_manager = ErrorRecoveryManager(enable_logging=False)
        
        # Run operations continuously for specified time
        while time.time() - start_time < max_runtime:
            # Validation
            config = {"n_way": 5, "k_shot": 2}
            validate_complete_config(config)
            
            # Warnings (with some that trigger warnings)
            if iterations % 10 == 0:
                warning_system.warn_if_suboptimal_few_shot(25, 1)  # Triggers warning
            else:
                warning_system.warn_if_suboptimal_few_shot(5, 2)  # No warning
            
            # Error recovery
            episode = create_robust_episode(3, 1, 5, 16)
            
            # Metrics
            predictions = torch.randn(5, 3)
            targets = torch.randint(0, 3, (5,))
            accuracy_calc = Accuracy()
            accuracy_calc.compute(predictions, targets)
            
            iterations += 1
            
            # Periodic cleanup
            if iterations % 100 == 0:
                gc.collect()
        
        end_time = time.time()
        actual_runtime = end_time - start_time
        
        # Should have completed many iterations
        assert iterations > 100, f"Only completed {iterations} iterations in {actual_runtime:.3f}s"
        
        # Should maintain reasonable performance
        avg_time_per_iteration = actual_runtime / iterations
        assert avg_time_per_iteration < 0.01, f"Average {avg_time_per_iteration:.6f}s per iteration too slow"