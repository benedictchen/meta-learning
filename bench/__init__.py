"""
Professional benchmarking suite for few-shot learning evaluation.

This module provides enterprise-grade benchmarking utilities with proper
statistical evaluation following established few-shot learning protocols.
"""

from .benchmark import run_benchmark, BenchResult, mean_ci95

__all__ = ['run_benchmark', 'BenchResult', 'mean_ci95']