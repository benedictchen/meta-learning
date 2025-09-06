"""
Comprehensive meta-learning evaluation infrastructure.

Features:
- Statistical significance testing with confidence intervals
- Stratified episode sampling for robust evaluation
- Cross-validation support for meta-learning
- Model calibration and uncertainty analysis
- Prototype quality assessment
- Task difficulty analysis tools
"""

from .few_shot_evaluation_harness import FewShotEvaluationHarness
from .statistical_testing import StatisticalTestSuite
from .cross_validation import MetaLearningCrossValidator
from .calibration_analysis import CalibrationAnalyzer
from .uncertainty_metrics import UncertaintyQuantifier
from .prototype_analysis import PrototypeAnalyzer
from .task_difficulty import TaskDifficultyAssessor
from .performance_visualization import PerformanceVisualizer

__all__ = [
    "FewShotEvaluationHarness",
    "StatisticalTestSuite", 
    "MetaLearningCrossValidator",
    "CalibrationAnalyzer",
    "UncertaintyQuantifier",
    "PrototypeAnalyzer", 
    "TaskDifficultyAssessor",
    "PerformanceVisualizer"
]