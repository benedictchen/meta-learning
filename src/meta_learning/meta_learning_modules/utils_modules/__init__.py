"""
📋   Init  
============

🔬 Research Foundation:  
======================
Based on meta-learning and few-shot learning research:
- Finn, C., Abbeel, P. & Levine, S. (2017). "Model-Agnostic Meta-Learning for Fast Adaptation"
- Snell, J., Swersky, K. & Zemel, R. (2017). "Prototypical Networks for Few-shot Learning"
- Nichol, A., Achiam, J. & Schulman, J. (2018). "On First-Order Meta-Learning Algorithms"
🎯 ELI5 Summary:
This file is an important component in our AI research system! Like different organs 
in your body that work together to keep you healthy, this file has a specific job that 
helps the overall algorithm work correctly and efficiently.

🧪 Technical Details:
===================
Implementation details and technical specifications for this component.
Designed to work seamlessly within the research framework while
maintaining high performance and accuracy standards.

📋 Component Integration:
========================
    ┌──────────┐
    │   This   │
    │Component │ ←→ Other Components
    └──────────┘
         ↑↓
    System Integration

"""
"""
Meta-Learning Utils Modules
===========================

Modular utilities for meta-learning research.
"""

from .configurations import (
    TaskConfiguration,
    EvaluationConfig, 
    DatasetConfig,
    MetricsConfig,
    StatsConfig,
    CurriculumConfig,
    DiversityConfig
)

from .dataset_sampling import (
    MetaLearningDataset,
    TaskSampler
)

from .statistical_evaluation import (
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

from .analysis_visualization import (
    visualize_meta_learning_results,
    save_meta_learning_results,
    load_meta_learning_results
)

from .factory_functions import (
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

__all__ = [
    # Configurations
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
    
    # Factory Functions
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