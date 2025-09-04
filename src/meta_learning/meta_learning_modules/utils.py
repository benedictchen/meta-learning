"""
Meta-Learning Utilities - Modular Implementation 🛠️📚
=====================================================

🎯 **ELI5 Explanation**:
Think of this like the ultimate Swiss Army knife for AI researchers!
Just like a Swiss Army knife has different tools (knife, scissors, screwdriver) for different jobs,
this utilities module has different tools for different parts of meta-learning research:

- 📊 **Statistical Tools**: Like calculators that tell you if your results are actually good
- 🎲 **Dataset Tools**: Like smart organizers that prepare your training data perfectly  
- 📈 **Evaluation Tools**: Like report card generators that measure how well your AI learned
- 🎨 **Visualization Tools**: Like artists that turn your boring numbers into pretty charts
- ⚙️ **Configuration Tools**: Like instruction manuals that set up everything correctly

📊 **Utility Categories Visualization**:
```
Raw Research Needs:      Meta-Learning Utils:      Research Results:
┌─────────────────┐     ┌─────────────────────┐    ┌──────────────────┐
│ • Need datasets │     │ 📊 Statistical      │    │ ✅ Published     │
│ • Run experiments│ ──→ │ 🎲 Dataset Sampling │ ──→│    Papers        │
│ • Analyze results│     │ 📈 Evaluation       │    │ ✅ Conferences   │
│ • Make charts   │     │ 🎨 Visualization    │    │ ✅ Reproducible  │
└─────────────────┘     └─────────────────────┘    └──────────────────┘
```

🔬 **Research-Accurate Implementation**:
Fills critical gaps in existing meta-learning libraries (learn2learn, torchmeta, higher)
with statistically rigorous functionality based on best practices from:
- **Statistical Evaluation**: Proper confidence intervals and significance testing
- **Dataset Protocols**: Standard few-shot learning evaluation procedures  
- **Reproducible Science**: Consistent random seeds and deterministic results

💡 **Modular Architecture**:
The original 1632-line utility file has been intelligently split into focused modules,
making it easier to find and use the tools you need for your specific research.

Author: Benedict Chen (benedict@benedictchen.com)

This module provides research-accurate utilities for meta-learning that fill
critical gaps in existing libraries and provide statistically rigorous functionality
for proper scientific evaluation.
"""

# Import all components from modular structure for backward compatibility
from .utils_modules.configurations import (
    TaskConfiguration,
    EvaluationConfig,
    DatasetConfig,
    MetricsConfig,
    StatsConfig,
    CurriculumConfig,
    DiversityConfig
)

from .utils_modules.dataset_sampling import (
    MetaLearningDataset,
    TaskSampler
)

from .utils_modules.statistical_evaluation import (
    few_shot_accuracy,
    adaptation_speed,
    compute_confidence_interval,
    compute_confidence_interval_research_accurate,
    compute_t_confidence_interval,
    compute_meta_learning_ci,
    compute_bca_bootstrap_ci,
    basic_confidence_interval,
    estimate_difficulty,
    EvaluationMetrics,
    StatisticalAnalysis
)

from .utils_modules.analysis_visualization import (
    visualize_meta_learning_results,
    save_meta_learning_results,
    load_meta_learning_results
)

from .utils_modules.factory_functions import (
    create_basic_task_config,
    create_research_accurate_task_config,
    create_basic_evaluation_config,
    create_research_accurate_evaluation_config,
    create_meta_learning_standard_evaluation_config,
    create_dataset,
    create_metrics_evaluator,
    create_curriculum_scheduler,
    track_task_diversity,
    evaluate_meta_learning_algorithm,
    CurriculumLearning,
    TaskDiversityTracker
)

# Export all for backward compatibility
__all__ = [
    # Configuration Classes
    'TaskConfiguration',
    'EvaluationConfig',
    'DatasetConfig',
    'MetricsConfig',
    'StatsConfig',
    'CurriculumConfig',
    'DiversityConfig',
    
    # Dataset & Sampling
    'MetaLearningDataset',
    'TaskSampler',
    
    # Statistical Evaluation
    'few_shot_accuracy',
    'adaptation_speed',
    'compute_confidence_interval',
    'compute_confidence_interval_research_accurate',
    'compute_t_confidence_interval',
    'compute_meta_learning_ci',
    'compute_bca_bootstrap_ci',
    'basic_confidence_interval',
    'estimate_difficulty',
    'EvaluationMetrics',
    'StatisticalAnalysis',
    
    # Analysis & Visualization
    'visualize_meta_learning_results',
    'save_meta_learning_results',
    'load_meta_learning_results',
    
    # Factory Functions & Helpers
    'create_basic_task_config',
    'create_research_accurate_task_config',
    'create_basic_evaluation_config',
    'create_research_accurate_evaluation_config',
    'create_meta_learning_standard_evaluation_config',
    'create_dataset',
    'create_metrics_evaluator',
    'create_curriculum_scheduler',
    'track_task_diversity',
    'evaluate_meta_learning_algorithm',
    'CurriculumLearning',
    'TaskDiversityTracker'
]

# Modularization Summary:
# ======================
# Original utils.py (1632 lines) split into:
# 1. configurations.py (125 lines) - Configuration dataclasses
# 2. dataset_sampling.py (523 lines) - Dataset and sampling classes
# 3. statistical_evaluation.py (421 lines) - Statistical functions and metrics
# 4. analysis_visualization.py (134 lines) - Visualization and I/O
# 5. factory_functions.py (278 lines) - Factory functions and helpers
# 
# Total modular lines: ~1481 lines (9% reduction through cleanup)
# Benefits: Better organization, easier maintenance, focused responsibilities