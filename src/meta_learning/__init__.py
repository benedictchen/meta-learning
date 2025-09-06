"""
💰 SUPPORT THIS RESEARCH - PLEASE DONATE! 💰

🙏 If this meta-learning library helps your research, please donate:
💳 PayPal: https://www.paypal.com/cgi-bin/webscr?cmd=_s-xclick&hosted_button_id=WXQKYYKPHWXHS
⭐ GitHub Sponsors: https://github.com/sponsors/benedictchen

👨‍💻 Author: Benedict Chen
💰 Donations: Help support this work! Buy me a coffee ☕, beer 🍺, lamborghini 🏎️, or private island 🏝️
💖 Please consider recurring donations to fully support continued research

Your support enables cutting-edge AI research for everyone! 🚀

META-LEARNING PACKAGE - 2024-2025 breakthrough algorithms
========================================================

🎯 **ELI5 Explanation**:
Imagine you're learning to learn! Just like how you get better at solving puzzles
the more types you practice, this package helps AI models get better at learning
new tasks by practicing on lots of similar tasks first.

Features:
- Test-Time Compute Scaling (FIRST public implementation!)
- Advanced MAML variants (MAML++, iMAML, MetaSGD)
- Enhanced Few-Shot Learning with multi-scale features  
- Online Meta-Learning with continual learning
- Research-grade evaluation and curriculum learning tools
- Professional package structure with comprehensive testing

💰 Please donate if this accelerates your research!
"""

from ._version import __version__

# Core components - always available
from .shared.types import Episode
from .core.utils import partition_task, remap_labels
from .core.seed import seed_all
from .core.bn_policy import freeze_batchnorm_running_stats
from .core.math_utils import pairwise_sqeuclidean, cosine_logits

# Data handling - essential utilities  
from .data_utils import InfiniteIterator, OnDeviceDataset
from .data import SyntheticFewShotDataset, CIFARFSDataset, MiniImageNetDataset, make_episodes

# Models - import conditionally to prevent crashes
try:
    from .models.conv4 import Conv4
    CONV4_AVAILABLE = True
except ImportError:
    Conv4 = None
    CONV4_AVAILABLE = False

# Algorithms - now with integrated advanced features
from .algos.protonet import ProtoHead  # Now includes uncertainty estimation
from .algos.maml import inner_adapt_and_eval, meta_outer_step, ContinualMAML

# Evaluation
from .eval import evaluate

# Benchmarking
from .bench import run_benchmark


# Hardware acceleration and research integrity (imported separately for CLI)
try:
    from .hardware_utils import (
        HardwareConfig, HardwareDetector, MemoryManager, ModelOptimizer,
        HardwareProfiler, create_hardware_config, setup_optimal_hardware
    )
    from .leakage_guard import (
        LeakageGuard, LeakageType, LeakageViolation, create_leakage_guard
    )
    INTEGRATED_ADVANCED_AVAILABLE = True
except ImportError:
    INTEGRATED_ADVANCED_AVAILABLE = False

# Legacy standalone modules (for backward compatibility)
try:
    from .continual_meta_learning import (
        OnlineMetaLearner, ContinualMetaConfig, FisherInformationMatrix,
        EpisodicMemoryBank, create_continual_meta_learner
    )
    from .few_shot_modules.uncertainty_components import (
        UncertaintyAwareDistance, MonteCarloDropout, DeepEnsemble,
        EvidentialLearning, UncertaintyConfig, create_uncertainty_aware_distance
    )
    STANDALONE_MODULES_AVAILABLE = True
except ImportError:
    STANDALONE_MODULES_AVAILABLE = False

try:
    # Basic  
    from .algorithms.ttc_scaler import TestTimeComputeScaler
    from .algorithms.ttc_config import TestTimeComputeConfig
    from .algorithms.maml_research_accurate import ResearchMAML, MAMLConfig, MAMLVariant
    
    # Import research patches  
    from .research_patches.batch_norm_policy import apply_episodic_bn_policy, EpisodicBatchNormPolicy
    from .research_patches.determinism_hooks import setup_deterministic_environment, DeterminismManager
    
    # Import evaluation harness
    from .evaluation.few_shot_evaluation_harness import FewShotEvaluationHarness
    
    # External research features available
    EXTERNAL_RESEARCH_AVAILABLE = True
    
except ImportError as e:
    # External research modules not available - core restored functionality still works!
    import warnings
    warnings.warn(f"External research modules not available: {e}. Core restored functionality still available!")
    
    TestTimeComputeScaler = None
    TestTimeComputeConfig = None
    ResearchMAML = None
    MAMLConfig = None
    MAMLVariant = None
    apply_episodic_bn_policy = None
    EpisodicBatchNormPolicy = None
    setup_deterministic_environment = None
    DeterminismManager = None
    FewShotEvaluationHarness = None
    EXTERNAL_RESEARCH_AVAILABLE = False

# Core restored functionality is ALWAYS available now
RESEARCH_AVAILABLE = True

# Input validation and error handling (newly implemented)
from .validation import (
    ValidationError, ConfigurationWarning, validate_episode_tensors,
    validate_few_shot_configuration, validate_distance_metric, 
    validate_temperature_parameter, validate_learning_rate,
    validate_model_parameters, validate_maml_config, validate_uncertainty_config,
    validate_optimizer_config, validate_episodic_config, validate_regularization_config,
    validate_complete_config, ValidationContext, check_episode_quality
)

# Error recovery and fault tolerance (newly implemented)
from .error_recovery import (
    RecoveryError, ErrorRecoveryManager, with_retry, safe_tensor_operation,
    handle_numerical_instability, recover_from_dimension_mismatch,
    RobustPrototypeNetwork, create_robust_episode, FaultTolerantTrainer, safe_evaluate
)
from .core.seed import ReproducibilityManager, distributed_seed_sync, benchmark_reproducibility_overhead, validate_seed_effectiveness
from .core.bn_policy import apply_episodic_bn_policy, validate_bn_compatibility, EpisodicBatchNormPolicy
from .algos.ttcs import TTCSWarningSystem, ttcs_with_fallback, ttcs_for_learn2learn_models, TTCSProfiler

# High-level toolkit API - core component, should always work
from .toolkit import MetaLearningToolkit, create_meta_learning_toolkit, quick_evaluation

# Dataset ecosystem - newly implemented
try:
    from .data_utils.datasets import (
        BenchmarkDatasetManager, OnDeviceDataset, InfiniteEpisodeIterator,
        MiniImageNetDataset, SyntheticFewShotDataset, DatasetRegistry,
        BaseMetaLearningDataset
    )
    DATASET_ECOSYSTEM_AVAILABLE = True
except ImportError:
    BenchmarkDatasetManager = None
    OnDeviceDataset = None
    InfiniteEpisodeIterator = None
    MiniImageNetDataset = None
    SyntheticFewShotDataset = None
    DatasetRegistry = None
    BaseMetaLearningDataset = None
    DATASET_ECOSYSTEM_AVAILABLE = False

# Error handling and monitoring - newly implemented
try:
    from .error_handling import (
        IntelligentErrorRecovery, PerformanceMonitor, WarningManager,
        ErrorType, WarningCategory, with_error_recovery, monitor_performance
    )
    ERROR_HANDLING_AVAILABLE = True
except ImportError:
    IntelligentErrorRecovery = None
    PerformanceMonitor = None
    WarningManager = None
    ErrorType = None
    WarningCategory = None
    with_error_recovery = None
    monitor_performance = None
    ERROR_HANDLING_AVAILABLE = False

__all__ = [
    "Episode",
    "remap_labels",
    "__version__",
    "partition_task",
    "InfiniteIterator",
    "OnDeviceDataset",
    "evaluate",
    "validate_episode_tensors",
    "validate_few_shot_configuration",
    "validate_distance_metric",
    "validate_temperature_parameter",
    "validate_learning_rate",
    "validate_model_parameters",
    "validate_maml_config",
    "validate_uncertainty_config",
    "validate_optimizer_config",
    "validate_episodic_config",
    "validate_regularization_config",
    "validate_complete_config",
    "validate_seed_effectiveness",
    "validate_bn_compatibility",
    "ValidationError",
    "RecoveryError",
    "ErrorRecoveryManager",
    "IntelligentErrorRecovery",
    "ErrorType",
    "with_error_recovery",
    "TestTimeComputeScaler",
    "TestTimeComputeConfig",
    "ResearchMAML",
    "MAMLConfig",
    "MAMLVariant",
    "setup_deterministic_environment",
    "DeterminismManager",
    "FewShotEvaluationHarness",
    "MetaLearningToolkit",
    "create_meta_learning_toolkit",
    "quick_evaluation",
    "RESEARCH_AVAILABLE",
    "EXTERNAL_RESEARCH_AVAILABLE",
    "DATASET_ECOSYSTEM_AVAILABLE",
    "ERROR_HANDLING_AVAILABLE",
    "ConfigurationWarning",
    "ValidationContext",
    "check_episode_quality",
    "with_retry",
    "safe_tensor_operation",
    "handle_numerical_instability",
    "recover_from_dimension_mismatch",
    "RobustPrototypeNetwork",
    "create_robust_episode",
    "FaultTolerantTrainer",
    "safe_evaluate",
    "ReproducibilityManager",
    "distributed_seed_sync",
    "benchmark_reproducibility_overhead",
    "apply_episodic_bn_policy",
    "EpisodicBatchNormPolicy",
    "TTCSWarningSystem",
    "ttcs_with_fallback",
    "ttcs_for_learn2learn_models",
    "TTCSProfiler",
    "HardwareConfig",
    "BenchmarkDatasetManager",
    "InfiniteEpisodeIterator",
    "MiniImageNetDataset",
    "SyntheticFewShotDataset",
    "DatasetRegistry",
    "BaseMetaLearningDataset",
    "PerformanceMonitor",
    "WarningManager",
    "WarningCategory",
    "monitor_performance"
]
