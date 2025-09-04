"""
💪 Stress Testing Suite for Meta-Learning Package
================================================

This module contains comprehensive stress tests and edge case validation
to ensure robustness under extreme conditions and potential failure scenarios.

Test Categories:
- Extreme parameter configurations and edge cases
- Numerical stability under unusual conditions  
- Resource exhaustion and memory pressure testing
- Invalid input handling and error recovery
- Concurrent execution stress testing
- Long-running stability validation

These tests ensure production readiness by validating behavior under:
- Memory constraints and resource limits
- Extreme hyperparameter values
- Unusual data distributions
- Concurrent multi-threaded execution
- Extended training scenarios

Usage:
    pytest tests/stress/ -v -m stress           # Run all stress tests
    pytest tests/stress/ -v -m "stress and not slow"  # Skip long tests
    pytest tests/stress/ -v --maxfail=1         # Stop on first failure
"""