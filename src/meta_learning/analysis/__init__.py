"""
Analysis tools for meta-learning research.

This package provides comprehensive analysis capabilities including:
- Task difficulty assessment with multiple complexity measures
- Prototype quality analysis for few-shot learning
- Learning dynamics analysis
- Statistical complexity measures

Key modules:
- task_difficulty: Comprehensive task difficulty assessment
- prototype_quality: Prototype quality analysis tools
"""

from .task_difficulty import TaskDifficultyAssessor
from .prototype_quality import PrototypeQualityAnalyzer

__all__ = [
    'TaskDifficultyAssessor',
    'PrototypeQualityAnalyzer',
]