"""
Visualization Package for Meta-Learning.

This package provides comprehensive visualization tools for meta-learning
experiments including learning curves, statistical comparisons, and task analysis.

Key components:
- VisualizationConfig: Configuration settings for all plots
- LearningCurveAnalyzer: Learning curve analysis and convergence plots
- StatisticalComparison: Statistical comparison plots with significance testing
- TaskAnalysisPlots: Task-specific analysis including difficulty and embeddings

These components work together to provide publication-ready visualizations
for meta-learning research with proper statistical analysis.
"""

from .config import VisualizationConfig
from .learning_curves import LearningCurveAnalyzer
from .statistical_plots import StatisticalComparison
from .task_analysis import TaskAnalysisPlots

__all__ = [
    'VisualizationConfig',
    'LearningCurveAnalyzer',
    'StatisticalComparison', 
    'TaskAnalysisPlots',
]