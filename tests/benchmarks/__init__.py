"""
🚀 Performance Benchmarking Suite for Meta-Learning Package
===========================================================

This module contains comprehensive performance benchmarks that evaluate
meta-learning algorithms across different configurations and hardware setups.

Benchmark Categories:
- Algorithm speed and accuracy comparisons
- Memory usage profiling and optimization
- Scalability testing across problem sizes
- Hardware acceleration performance analysis
- Cross-component integration benchmarks

Usage:
    pytest tests/benchmarks/ -v -m benchmark       # Run all benchmarks
    pytest tests/benchmarks/ -v -m "benchmark and not slow"  # Skip slow benchmarks
    pytest tests/benchmarks/ -v --benchmark-json=results.json  # Save results
"""