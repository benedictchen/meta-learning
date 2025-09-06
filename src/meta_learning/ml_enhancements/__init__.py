"""
ML-powered enhancements for meta-learning.

This package provides advanced ML-powered features that enhance the core
meta-learning toolkit with intelligent automation and optimization capabilities.

Key components:
- FailurePredictionModel: ML-based failure prediction
- AlgorithmSelector: Automatic algorithm selection  
- ABTestingFramework: A/B testing for hyperparameters
- CrossTaskKnowledgeTransfer: Knowledge transfer between tasks
- PerformanceMonitor: Real-time performance monitoring

These components work together to create an intelligent meta-learning system
that learns from experience and automatically optimizes performance.
"""

from .failure_prediction import FailurePredictionModel
from .algorithm_selection import AlgorithmSelector
from .ab_testing import ABTestingFramework
from .knowledge_transfer import CrossTaskKnowledgeTransfer
from .performance_monitoring import PerformanceMonitor

__all__ = [
    'FailurePredictionModel',
    'AlgorithmSelector', 
    'ABTestingFramework',
    'CrossTaskKnowledgeTransfer',
    'PerformanceMonitor',
]