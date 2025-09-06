"""
💰 SUPPORT THIS RESEARCH - PLEASE DONATE! 💰

🙏 If this reproducibility library helps your research, please donate:
💳 PayPal: https://www.paypal.com/cgi-bin/webscr?cmd=_s-xclick&hosted_button_id=WXQKYYKPHWXHS
⭐ GitHub Sponsors: https://github.com/sponsors/benedictchen

👨‍💻 Author: Benedict Chen
💰 Donations: Help support this work! Buy me a coffee ☕, beer 🍺, lamborghini 🏎️, or private island 🏝️
💖 Please consider recurring donations to fully support continued research

Your support enables cutting-edge AI research for everyone! 🚀

Reproducibility and Seeding Utilities
====================================

🎯 **ELI5 Explanation**:
Ever play a video game where you could save your progress and reload it later?
This does the same thing for AI experiments - it makes sure your experiments
give the same results every time you run them!
"""
from __future__ import annotations
import os, random
import numpy as np
import torch


def seed_all(seed: int) -> None:
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    try:
        import torch.backends.cudnn as cudnn
        cudnn.deterministic = True; cudnn.benchmark = False
    except Exception:
        pass

class ReproducibilityManager:
    """
    Complete reproducibility management for meta-learning experiments.
    
    Features:
    - Multi-GPU deterministic seeding with device affinity
    - Seed verification and validation across runs  
    - Performance impact monitoring for deterministic operations
    - Automatic fallback for unsupported deterministic operations
    - Distributed training seed synchronization
    """
    
    def __init__(self, base_seed: int = 42, enable_performance_monitoring: bool = False):
        """Initialize reproducibility manager."""
        self.base_seed = base_seed
        self.enable_performance_monitoring = enable_performance_monitoring
        self.device_seeds = {}
        self.performance_logs = []
        self.fallback_operations = set()
        
    def setup_deterministic_environment(self, strict_mode: bool = False) -> dict:
        """Setup comprehensive deterministic environment."""
        import time
        
        start_time = time.time()
        
        # Basic seeding
        seed_all(self.base_seed)
        
        # Device-specific seeding
        if torch.cuda.is_available():
            for device_id in range(torch.cuda.device_count()):
                device_seed = self.base_seed + device_id
                self.device_seeds[device_id] = device_seed
                torch.cuda.manual_seed(device_seed)
        
        # Strict deterministic settings
        if strict_mode:
            self._enable_strict_determinism()
        
        setup_time = time.time() - start_time
        
        if self.enable_performance_monitoring:
            self.performance_logs.append({
                'operation': 'setup_deterministic_environment',
                'duration': setup_time,
                'strict_mode': strict_mode
            })
        
        return {
            'base_seed': self.base_seed,
            'device_seeds': self.device_seeds,
            'setup_time': setup_time,
            'strict_mode': strict_mode
        }
    
    def _enable_strict_determinism(self):
        """Enable strict deterministic algorithms with fallbacks."""
        try:
            torch.use_deterministic_algorithms(True)
        except RuntimeError as e:
            self.fallback_operations.add('deterministic_algorithms')
            torch.use_deterministic_algorithms(True, warn_only=True)
        
        # Additional strict settings
        try:
            import torch.backends.cudnn as cudnn
            cudnn.deterministic = True
            cudnn.benchmark = False
        except Exception:
            self.fallback_operations.add('cudnn_deterministic')
    
    def validate_seed_effectiveness(self, model: torch.nn.Module, 
                                  test_input: torch.Tensor, num_runs: int = 3) -> dict:
        """Validate that seeding produces identical results."""
        import time
        
        results = []
        validation_start = time.time()
        
        for run in range(num_runs):
            seed_all(self.base_seed)
            
            model.train()
            with torch.no_grad():
                output = model(test_input)
            
            # Store results for comparison
            if isinstance(output, (list, tuple)):
                results.append([tensor.detach().cpu() for tensor in output])
            else:
                results.append(output.detach().cpu())
        
        # Check consistency across runs
        is_consistent = self._check_result_consistency(results)
        validation_time = time.time() - validation_start
        
        if self.enable_performance_monitoring:
            self.performance_logs.append({
                'operation': 'validate_seed_effectiveness',
                'duration': validation_time,
                'num_runs': num_runs,
                'consistent': is_consistent
            })
        
        return {
            'is_consistent': is_consistent,
            'num_runs': num_runs,
            'validation_time': validation_time,
            'fallback_operations_used': list(self.fallback_operations)
        }
    
    def _check_result_consistency(self, results: list) -> bool:
        """Check if all results are identical."""
        if len(results) < 2:
            return True
        
        first_result = results[0]
        
        for result in results[1:]:
            if isinstance(first_result, list):
                if len(first_result) != len(result):
                    return False
                for t1, t2 in zip(first_result, result):
                    if not torch.allclose(t1, t2, atol=1e-6):
                        return False
            else:
                if not torch.allclose(first_result, result, atol=1e-6):
                    return False
        
        return True
    
    def benchmark_reproducibility_overhead(self, benchmark_func, num_trials: int = 5) -> dict:
        """Measure performance impact of deterministic operations."""
        import time
        import statistics
        
        # Benchmark without deterministic algorithms
        non_det_times = []
        for _ in range(num_trials):
            torch.use_deterministic_algorithms(False)
            start_time = time.time()
            benchmark_func()
            non_det_times.append(time.time() - start_time)
        
        # Benchmark with deterministic algorithms
        det_times = []
        for _ in range(num_trials):
            torch.use_deterministic_algorithms(True, warn_only=True)
            start_time = time.time()
            benchmark_func()
            det_times.append(time.time() - start_time)
        
        # Restore original setting
        self.setup_deterministic_environment()
        
        return {
            'non_deterministic_mean': statistics.mean(non_det_times),
            'non_deterministic_std': statistics.stdev(non_det_times) if len(non_det_times) > 1 else 0,
            'deterministic_mean': statistics.mean(det_times),
            'deterministic_std': statistics.stdev(det_times) if len(det_times) > 1 else 0,
            'overhead_ratio': statistics.mean(det_times) / statistics.mean(non_det_times),
            'overhead_absolute': statistics.mean(det_times) - statistics.mean(non_det_times),
            'num_trials': num_trials
        }


def distributed_seed_sync(seed: int, world_size: int, rank: int) -> int:
    """
    Distributed training seed management with per-process diversity.
    
    Args:
        seed: Base seed
        world_size: Total number of processes
        rank: Current process rank
        
    Returns:
        Process-specific seed
    """
    # Derive process-specific seed while maintaining reproducibility
    process_seed = seed + rank * 1000007  # Large prime to avoid overlap
    
    # Ensure all processes have consistent base seeding
    seed_all(seed)
    
    # Set process-specific random seeds for data loading diversity
    np.random.seed(process_seed)
    random.seed(process_seed)
    
    return process_seed


def benchmark_reproducibility_overhead(model, test_input: torch.Tensor, 
                                     num_trials: int = 5) -> dict:
    """
    Benchmark reproducibility overhead for a given model and input.
    
    Args:
        model: PyTorch model to benchmark
        test_input: Input tensor for benchmarking
        num_trials: Number of trials for statistical significance
        
    Returns:
        Performance comparison results
    """
    manager = ReproducibilityManager(enable_performance_monitoring=True)
    
    def benchmark_func():
        with torch.no_grad():
            _ = model(test_input)
    
    return manager.benchmark_reproducibility_overhead(benchmark_func, num_trials)


def validate_seed_effectiveness(seed: int, model: torch.nn.Module, 
                              test_input: torch.Tensor, num_runs: int = 3) -> bool:
    """
    Validate that seeding produces identical results across multiple runs.
    
    Args:
        seed: Seed to validate
        model: Model to test
        test_input: Test input tensor
        num_runs: Number of runs to compare
        
    Returns:
        True if results are consistent, False otherwise
    """
    manager = ReproducibilityManager(base_seed=seed)
    result = manager.validate_seed_effectiveness(model, test_input, num_runs)
    return result['is_consistent']
