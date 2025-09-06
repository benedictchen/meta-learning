"""
Fast Performance Benchmark for New Meta-Learning Features
=========================================================

Quick performance analysis of key new features with optimized benchmarking.
"""

import time
import torch
import numpy as np
from typing import Dict, List, Any

# Import key new features
from src.meta_learning.shared.types import Episode
from src.meta_learning.core.utils import clone_module, update_module, detach_module
from src.meta_learning.core.math_utils import pairwise_cosine_similarity, magic_box
from src.meta_learning.algorithms.ridge_regression import RidgeRegression
from src.meta_learning.ml_enhancements.algorithm_selection import AlgorithmSelector


def benchmark_function(func, *args, num_runs=10, **kwargs):
    """Quick benchmark of a function."""
    times = []
    
    # Warmup
    for _ in range(2):
        try:
            func(*args, **kwargs)
        except:
            pass
    
    # Benchmark
    for _ in range(num_runs):
        start = time.perf_counter()
        try:
            result = func(*args, **kwargs)
            end = time.perf_counter()
            times.append(end - start)
        except Exception as e:
            print(f"Error in {func.__name__}: {e}")
            return float('inf'), float('inf')
    
    if not times:
        return float('inf'), float('inf')
    
    return np.mean(times), np.std(times)


def main():
    """Run fast benchmark."""
    print("🚀 Fast Performance Benchmark - New Meta-Learning Features")
    print("=" * 60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Create test data
    support_x = torch.randn(15, 64, device=device)  # 5-way 3-shot
    support_y = torch.arange(5, device=device).repeat_interleave(3)
    query_x = torch.randn(20, 64, device=device)    # 5-way 4-query
    query_y = torch.arange(5, device=device).repeat_interleave(4)
    episode = Episode(support_x, support_y, query_x, query_y)
    
    model = torch.nn.Sequential(
        torch.nn.Linear(64, 32),
        torch.nn.ReLU(),
        torch.nn.Linear(32, 5)
    ).to(device)
    
    results = {}
    
    # Benchmark core utilities
    print("🔧 Core Utilities:")
    mean_time, std_time = benchmark_function(clone_module, model)
    results['clone_module'] = (mean_time, std_time)
    print(f"  clone_module: {mean_time*1000:.2f}±{std_time*1000:.2f}ms")
    
    mean_time, std_time = benchmark_function(detach_module, model)
    results['detach_module'] = (mean_time, std_time)
    print(f"  detach_module: {mean_time*1000:.2f}±{std_time*1000:.2f}ms")
    
    # Benchmark math utilities
    print("🔢 Math Utilities:")
    mean_time, std_time = benchmark_function(pairwise_cosine_similarity, support_x, query_x)
    results['cosine_similarity'] = (mean_time, std_time)
    print(f"  cosine_similarity: {mean_time*1000:.2f}±{std_time*1000:.2f}ms")
    
    mean_time, std_time = benchmark_function(magic_box, support_x)
    results['magic_box'] = (mean_time, std_time)
    print(f"  magic_box: {mean_time*1000:.2f}±{std_time*1000:.2f}ms")
    
    # Benchmark ridge regression
    print("📈 Ridge Regression:")
    ridge = RidgeRegression(reg_lambda=0.01, use_woodbury=True)
    y_targets = support_y.unsqueeze(1).float()
    
    mean_time, std_time = benchmark_function(ridge.fit, support_x, y_targets)
    results['ridge_fit'] = (mean_time, std_time)
    print(f"  ridge_fit: {mean_time*1000:.2f}±{std_time*1000:.2f}ms")
    
    # Fit for prediction benchmark
    ridge.fit(support_x, y_targets)
    mean_time, std_time = benchmark_function(ridge.predict, query_x)
    results['ridge_predict'] = (mean_time, std_time)
    print(f"  ridge_predict: {mean_time*1000:.2f}±{std_time*1000:.2f}ms")
    
    # Benchmark algorithm selection
    print("🤖 Algorithm Selection:")
    selector = AlgorithmSelector()
    mean_time, std_time = benchmark_function(selector.select_algorithm, episode)
    results['algorithm_selection'] = (mean_time, std_time)
    print(f"  algorithm_selection: {mean_time*1000:.2f}±{std_time*1000:.2f}ms")
    
    # Summary
    print("\n📊 PERFORMANCE SUMMARY:")
    print("-" * 30)
    
    valid_results = {k: v for k, v in results.items() if v[0] != float('inf')}
    
    if valid_results:
        fastest = min(valid_results.items(), key=lambda x: x[1][0])
        slowest = max(valid_results.items(), key=lambda x: x[1][0])
        
        print(f"⚡ FASTEST: {fastest[0]} ({fastest[1][0]*1000:.2f}ms)")
        print(f"🐌 SLOWEST: {slowest[0]} ({slowest[1][0]*1000:.2f}ms)")
        
        avg_time = np.mean([v[0] for v in valid_results.values()])
        print(f"📊 AVERAGE: {avg_time*1000:.2f}ms")
        
        slow_ops = {k: v for k, v in valid_results.items() if v[0] > 0.01}  # >10ms
        if slow_ops:
            print("⚠️  SLOW OPERATIONS (>10ms):")
            for name, (mean_t, std_t) in sorted(slow_ops.items(), key=lambda x: x[1][0], reverse=True):
                print(f"   {name}: {mean_t*1000:.1f}ms")
    
    print("\n✅ Fast benchmark completed!")
    return results


if __name__ == "__main__":
    main()