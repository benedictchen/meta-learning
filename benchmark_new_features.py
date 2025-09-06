"""
💰 SUPPORT THIS RESEARCH - PLEASE DONATE! 💰

Performance Benchmarking Suite for New Meta-Learning Features
============================================================

Comprehensive benchmarking and optimization analysis for all newly implemented
features to ensure optimal performance and identify bottlenecks.

🎯 **Benchmarking Coverage:**
- Core utilities (clone/update/detach) performance analysis
- Ridge regression vs other algorithms speed comparison
- Enhanced math utilities optimization assessment
- Algorithm selection performance with hardness analysis
- Data download and processing speed metrics
- Memory usage profiling across all new features

💰 Please donate if this accelerates your research!
"""

import time
import psutil
import gc
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

# Import all new features for benchmarking
from src.meta_learning.shared.types import Episode
from src.meta_learning.core.utils import clone_module, update_module, detach_module
from src.meta_learning.core.math_utils import (
    magic_box, pairwise_cosine_similarity, matching_loss,
    pairwise_sqeuclidean, cosine_logits, batched_prototype_computation
)
from src.meta_learning.algorithms.ridge_regression import RidgeRegression
from src.meta_learning.evaluation.enhanced_learnability import EnhancedLearnabilityAnalyzer
from src.meta_learning.evaluation.task_analysis import hardness_metric
from src.meta_learning.ml_enhancements.algorithm_registry import algorithm_registry
from src.meta_learning.ml_enhancements.algorithm_selection import AlgorithmSelector
from src.meta_learning.ml_enhancements.hardness_aware_selector import HardnessAwareSelector
from src.meta_learning.optimization.learnable_optimizer import LearnableOptimizer


@dataclass
class BenchmarkResult:
    """Container for benchmark results."""
    feature_name: str
    operation: str
    mean_time: float
    std_time: float
    memory_peak: float
    memory_delta: float
    throughput: Optional[float] = None
    additional_metrics: Dict[str, Any] = None


class PerformanceBenchmark:
    """
    Comprehensive performance benchmarking suite for new meta-learning features.
    
    Measures:
    - Execution time statistics
    - Memory usage patterns  
    - Throughput metrics
    - Scalability analysis
    - Optimization opportunities
    """
    
    def __init__(self, num_runs: int = 100, warmup_runs: int = 10):
        """
        Initialize benchmark suite.
        
        Args:
            num_runs: Number of benchmark runs for averaging
            warmup_runs: Number of warmup runs to exclude from timing
        """
        self.num_runs = num_runs
        self.warmup_runs = warmup_runs
        self.results: List[BenchmarkResult] = []
        
        # Set device for consistent benchmarking
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"🔧 Benchmarking on device: {self.device}")
        
        # Prepare test data of various sizes
        self.test_data = self._prepare_test_data()
        
    def _prepare_test_data(self) -> Dict[str, Any]:
        """Prepare test data of various sizes for benchmarking."""
        return {
            'small_episode': self._create_episode(5, 3, 2, 32),    # 5-way 3-shot, 32-dim
            'medium_episode': self._create_episode(10, 5, 5, 64),  # 10-way 5-shot, 64-dim
            'large_episode': self._create_episode(20, 10, 10, 128), # 20-way 10-shot, 128-dim
            'small_model': self._create_test_model(32, 16, 5),
            'medium_model': self._create_test_model(64, 32, 10),
            'large_model': self._create_test_model(128, 64, 20),
            'batch_sizes': [16, 32, 64, 128, 256],
            'feature_dims': [32, 64, 128, 256, 512]
        }
    
    def _create_episode(self, n_way: int, k_shot: int, n_query: int, dim: int) -> Episode:
        """Create synthetic episode for benchmarking."""
        support_x = torch.randn(n_way * k_shot, dim, device=self.device)
        support_y = torch.arange(n_way, device=self.device).repeat_interleave(k_shot)
        query_x = torch.randn(n_way * n_query, dim, device=self.device)
        query_y = torch.arange(n_way, device=self.device).repeat_interleave(n_query)
        return Episode(support_x, support_y, query_x, query_y)
    
    def _create_test_model(self, input_dim: int, hidden_dim: int, output_dim: int) -> nn.Module:
        """Create test model for benchmarking."""
        return nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, output_dim)
        ).to(self.device)
    
    def _measure_time_and_memory(self, func, *args, **kwargs) -> Tuple[float, float, float]:
        """
        Measure execution time and memory usage of a function.
        
        Returns:
            Tuple of (execution_time, memory_peak, memory_delta)
        """
        # Clear GPU cache if using CUDA
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
        
        # Garbage collection for consistent memory measurement
        gc.collect()
        
        # Measure initial memory
        process = psutil.Process()
        memory_before = process.memory_info().rss / 1024 / 1024  # MB
        
        if self.device.type == 'cuda':
            torch.cuda.synchronize()
            gpu_memory_before = torch.cuda.memory_allocated() / 1024 / 1024  # MB
        
        # Execute function with timing
        start_time = time.perf_counter()
        
        if self.device.type == 'cuda':
            torch.cuda.synchronize()
        
        result = func(*args, **kwargs)
        
        if self.device.type == 'cuda':
            torch.cuda.synchronize()
        
        end_time = time.perf_counter()
        execution_time = end_time - start_time
        
        # Measure final memory
        memory_after = process.memory_info().rss / 1024 / 1024  # MB
        memory_delta = memory_after - memory_before
        
        if self.device.type == 'cuda':
            gpu_memory_after = torch.cuda.memory_allocated() / 1024 / 1024  # MB
            gpu_memory_delta = gpu_memory_after - gpu_memory_before
            memory_peak = max(memory_after, gpu_memory_after)
        else:
            memory_peak = memory_after
        
        return execution_time, memory_peak, memory_delta
    
    def _run_benchmark(self, name: str, operation: str, func, *args, **kwargs) -> BenchmarkResult:
        """Run a single benchmark with multiple iterations."""
        print(f"  🏃 Benchmarking {name} - {operation}...")
        
        times = []
        memory_peaks = []
        memory_deltas = []
        
        # Warmup runs
        for _ in range(self.warmup_runs):
            try:
                self._measure_time_and_memory(func, *args, **kwargs)
            except Exception as e:
                print(f"    ⚠️  Warmup failed: {e}")
                continue
        
        # Actual benchmark runs
        successful_runs = 0
        for i in range(self.num_runs):
            try:
                exec_time, mem_peak, mem_delta = self._measure_time_and_memory(func, *args, **kwargs)
                times.append(exec_time)
                memory_peaks.append(mem_peak)
                memory_deltas.append(mem_delta)
                successful_runs += 1
            except Exception as e:
                print(f"    ⚠️  Run {i+1} failed: {e}")
                continue
        
        if successful_runs == 0:
            print(f"    ❌ All benchmark runs failed for {name} - {operation}")
            return BenchmarkResult(
                name, operation, float('inf'), float('inf'), 
                float('inf'), float('inf')
            )
        
        # Calculate statistics
        mean_time = np.mean(times)
        std_time = np.std(times)
        mean_memory_peak = np.mean(memory_peaks)
        mean_memory_delta = np.mean(memory_deltas)
        
        result = BenchmarkResult(
            feature_name=name,
            operation=operation,
            mean_time=mean_time,
            std_time=std_time,
            memory_peak=mean_memory_peak,
            memory_delta=mean_memory_delta,
            additional_metrics={
                'successful_runs': successful_runs,
                'total_runs': self.num_runs,
                'success_rate': successful_runs / self.num_runs
            }
        )
        
        print(f"    ✅ {operation}: {mean_time*1000:.2f}±{std_time*1000:.2f}ms, "
              f"Mem: {mean_memory_delta:.1f}MB delta")
        
        return result
    
    def benchmark_core_utilities(self):
        """Benchmark core utilities performance."""
        print("🔧 Benchmarking Core Utilities...")
        
        model = self.test_data['medium_model']
        
        # Benchmark clone_module
        result = self._run_benchmark(
            "Core Utilities", "clone_module", 
            clone_module, model
        )
        self.results.append(result)
        
        # Benchmark update_module
        cloned = clone_module(model)
        updates = {name: torch.randn_like(param) * 0.01 
                  for name, param in model.named_parameters()}
        
        result = self._run_benchmark(
            "Core Utilities", "update_module",
            update_module, cloned, updates
        )
        self.results.append(result)
        
        # Benchmark detach_module
        result = self._run_benchmark(
            "Core Utilities", "detach_module",
            detach_module, model
        )
        self.results.append(result)
    
    def benchmark_ridge_regression(self):
        """Benchmark ridge regression performance."""
        print("📈 Benchmarking Ridge Regression...")
        
        # Test different data sizes
        for size_name, episode in [('small', self.test_data['small_episode']),
                                  ('medium', self.test_data['medium_episode']),
                                  ('large', self.test_data['large_episode'])]:
            
            ridge = RidgeRegression(reg_lambda=0.01, use_woodbury=True)
            
            # Benchmark fit
            X = episode.support_x
            y = episode.support_y.unsqueeze(1).float()  # Ridge expects [N, 1] output
            
            result = self._run_benchmark(
                f"Ridge Regression ({size_name})", "fit",
                ridge.fit, X, y
            )
            self.results.append(result)
            
            # Fit for prediction benchmark
            ridge.fit(X, y)
            
            # Benchmark predict
            result = self._run_benchmark(
                f"Ridge Regression ({size_name})", "predict",
                ridge.predict, episode.query_x
            )
            self.results.append(result)
    
    def benchmark_math_utilities(self):
        """Benchmark enhanced math utilities."""
        print("🔢 Benchmarking Math Utilities...")
        
        # Test different tensor sizes
        for dim in [64, 128, 256]:
            a = torch.randn(100, dim, device=self.device)
            b = torch.randn(50, dim, device=self.device)
            
            # Benchmark pairwise_cosine_similarity
            result = self._run_benchmark(
                f"Math Utilities (dim={dim})", "pairwise_cosine_similarity",
                pairwise_cosine_similarity, a, b
            )
            self.results.append(result)
            
            # Benchmark pairwise_sqeuclidean
            result = self._run_benchmark(
                f"Math Utilities (dim={dim})", "pairwise_sqeuclidean",
                pairwise_sqeuclidean, a, b
            )
            self.results.append(result)
            
            # Benchmark magic_box
            result = self._run_benchmark(
                f"Math Utilities (dim={dim})", "magic_box",
                magic_box, a
            )
            self.results.append(result)
    
    def benchmark_algorithm_selection(self):
        """Benchmark algorithm selection performance."""
        print("🤖 Benchmarking Algorithm Selection...")
        
        selector = AlgorithmSelector()
        hardness_selector = HardnessAwareSelector()
        
        for size_name, episode in [('small', self.test_data['small_episode']),
                                  ('medium', self.test_data['medium_episode'])]:
            
            # Benchmark basic algorithm selection
            result = self._run_benchmark(
                f"Algorithm Selection ({size_name})", "select_algorithm",
                selector.select_algorithm, episode
            )
            self.results.append(result)
            
            # Benchmark hardness-aware selection
            result = self._run_benchmark(
                f"Algorithm Selection ({size_name})", "hardness_aware_selection",
                hardness_selector.select_algorithm_with_hardness, episode
            )
            self.results.append(result)
    
    def benchmark_task_analysis(self):
        """Benchmark task analysis and difficulty computation."""
        print("📊 Benchmarking Task Analysis...")
        
        analyzer = EnhancedLearnabilityAnalyzer()
        
        for size_name, episode in [('small', self.test_data['small_episode']),
                                  ('medium', self.test_data['medium_episode']),
                                  ('large', self.test_data['large_episode'])]:
            
            # Benchmark hardness metric
            n_classes = len(torch.unique(episode.support_y))
            result = self._run_benchmark(
                f"Task Analysis ({size_name})", "hardness_metric",
                hardness_metric, episode, num_classes=n_classes
            )
            self.results.append(result)
            
            # Benchmark enhanced learnability analysis
            result = self._run_benchmark(
                f"Task Analysis ({size_name})", "enhanced_learnability",
                analyzer.compute_enhanced_task_difficulty, episode
            )
            self.results.append(result)
    
    def run_comprehensive_benchmark(self):
        """Run comprehensive benchmark of all new features."""
        print("🚀 Starting Comprehensive Performance Benchmark...")
        print("=" * 60)
        
        start_time = time.time()
        
        try:
            self.benchmark_core_utilities()
            self.benchmark_ridge_regression()
            self.benchmark_math_utilities()
            self.benchmark_algorithm_selection()
            self.benchmark_task_analysis()
        except Exception as e:
            print(f"❌ Benchmark failed with error: {e}")
            import traceback
            traceback.print_exc()
        
        total_time = time.time() - start_time
        print("=" * 60)
        print(f"🎯 Benchmark completed in {total_time:.2f}s")
        print(f"📊 Total results: {len(self.results)}")
        
        return self.results
    
    def generate_performance_report(self) -> str:
        """Generate comprehensive performance report."""
        if not self.results:
            return "❌ No benchmark results available. Run benchmark first."
        
        report = []
        report.append("🏆 META-LEARNING PERFORMANCE BENCHMARK REPORT")
        report.append("=" * 50)
        report.append("")
        
        # Group results by feature
        by_feature = {}
        for result in self.results:
            if result.feature_name not in by_feature:
                by_feature[result.feature_name] = []
            by_feature[result.feature_name].append(result)
        
        # Performance summary
        report.append("📊 PERFORMANCE SUMMARY:")
        report.append("-" * 30)
        
        for feature_name, results in by_feature.items():
            report.append(f"\n🔧 {feature_name}:")
            
            for result in results:
                if result.mean_time == float('inf'):
                    report.append(f"   ❌ {result.operation}: FAILED")
                else:
                    report.append(f"   ✅ {result.operation}: {result.mean_time*1000:.2f}ms "
                                f"(±{result.std_time*1000:.2f}ms)")
                    
                    if result.memory_delta > 0:
                        report.append(f"      💾 Memory: +{result.memory_delta:.1f}MB")
        
        # Find fastest and slowest operations
        valid_results = [r for r in self.results if r.mean_time != float('inf')]
        if valid_results:
            fastest = min(valid_results, key=lambda x: x.mean_time)
            slowest = max(valid_results, key=lambda x: x.mean_time)
            
            report.append(f"\n⚡ FASTEST: {fastest.feature_name} - {fastest.operation} "
                         f"({fastest.mean_time*1000:.2f}ms)")
            report.append(f"🐌 SLOWEST: {slowest.feature_name} - {slowest.operation} "
                         f"({slowest.mean_time*1000:.2f}ms)")
        
        # Memory usage analysis
        memory_intensive = [r for r in valid_results if r.memory_delta > 10]  # >10MB
        if memory_intensive:
            report.append(f"\n💾 HIGH MEMORY USAGE:")
            for result in sorted(memory_intensive, key=lambda x: x.memory_delta, reverse=True):
                report.append(f"   📈 {result.feature_name} - {result.operation}: "
                             f"+{result.memory_delta:.1f}MB")
        
        # Optimization recommendations
        report.append(f"\n🔧 OPTIMIZATION RECOMMENDATIONS:")
        
        slow_operations = [r for r in valid_results if r.mean_time > 0.1]  # >100ms
        if slow_operations:
            report.append("   ⚠️  Consider optimizing slow operations (>100ms):")
            for result in sorted(slow_operations, key=lambda x: x.mean_time, reverse=True):
                report.append(f"     - {result.feature_name} - {result.operation}: "
                             f"{result.mean_time*1000:.0f}ms")
        
        high_variance = [r for r in valid_results if r.std_time/r.mean_time > 0.3]  # >30% CV
        if high_variance:
            report.append("   📊 High variance operations (inconsistent performance):")
            for result in high_variance:
                cv = (result.std_time / result.mean_time) * 100
                report.append(f"     - {result.feature_name} - {result.operation}: "
                             f"{cv:.1f}% CV")
        
        report.append(f"\n✅ Benchmark completed successfully!")
        report.append(f"🎯 Device: {self.device}")
        report.append(f"📊 Total operations benchmarked: {len(valid_results)}")
        
        return "\n".join(report)


def main():
    """Run the complete performance benchmark."""
    print("🚀 Meta-Learning New Features Performance Benchmark")
    print("=" * 60)
    
    # Create benchmark suite
    benchmark = PerformanceBenchmark(num_runs=50, warmup_runs=5)
    
    # Run comprehensive benchmarks
    results = benchmark.run_comprehensive_benchmark()
    
    # Generate and display report
    report = benchmark.generate_performance_report()
    print("\n" + report)
    
    # Save report to file
    report_path = Path("performance_benchmark_report.md")
    with open(report_path, 'w') as f:
        f.write(report)
    
    print(f"\n📝 Report saved to: {report_path}")
    print("🎯 Performance benchmarking complete!")


if __name__ == "__main__":
    main()