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