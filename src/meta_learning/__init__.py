"""
💰 SUPPORT THIS RESEARCH - PLEASE DONATE! 💰

🙏 If this library helps your research or project, please consider donating:
💳 PayPal: https://www.paypal.com/cgi-bin/webscr?cmd=_s-xclick&hosted_button_id=WXQKYYKPHWXHS
⭐ GitHub Sponsors: https://github.com/sponsors/benedictchen

👨‍💻 Author: Benedict Chen
💰 Donations: Help support this work! Buy me a coffee ☕, beer 🍺, lamborghini 🏎️, or private island 🏝️
💖 Please consider recurring donations to fully support continued research

Your support enables cutting-edge AI research for everyone! 🚀

Author: Benedict Chen (benedict@benedictchen.com)
License: Custom Non-Commercial License with Donation Requirements
"""
from ._version import __version__
from .core.episode import Episode, remap_labels

# Core components - always available
from .core.episode import Episode, remap_labels
from .core.seed import seed_all
from .core.bn_policy import freeze_batchnorm_running_stats
from .core.math_utils import pairwise_sqeuclidean, cosine_logits

# Data handling
from .data import SyntheticFewShotDataset, make_episodes

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
    from algorithms.ttc_scaler import TestTimeComputeScaler
    from algorithms.ttc_config import TestTimeComputeConfig
    from algorithms.maml_research_accurate import ResearchMAML, MAMLConfig, MAMLVariant
    
    # Import research patches  
    from research_patches.batch_norm_policy import apply_episodic_bn_policy, EpisodicBatchNormPolicy
    from research_patches.determinism_hooks import setup_deterministic_environment, DeterminismManager
    
    # Import evaluation harness
    from evaluation.few_shot_evaluation_harness import FewShotEvaluationHarness
    
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

# High-level toolkit API - core component, should always work
from .toolkit import MetaLearningToolkit, create_meta_learning_toolkit, quick_evaluation

__all__ = [
    # Core functionality
    "Episode", "remap_labels", "__version__",
    
    # Continual Meta-Learning
    "OnlineMetaLearner", "ContinualMetaConfig", "FisherInformationMatrix",
    "EpisodicMemoryBank", "create_continual_meta_learner",
    # Hardware Acceleration  
    "HardwareConfig", "HardwareDetector", "MemoryManager", "ModelOptimizer",
    "HardwareProfiler", "create_hardware_config", "setup_optimal_hardware",
    # Research Integrity
    "LeakageGuard", "LeakageType", "LeakageViolation", "create_leakage_guard", 
    # Advanced Few-Shot Learning
    "UncertaintyAwareDistance", "MonteCarloDropout", "DeepEnsemble",
    "EvidentialLearning", "UncertaintyConfig", "create_uncertainty_aware_distance",
    
    # External research algorithms (may be None if not available)
    "TestTimeComputeScaler", "TestTimeComputeConfig",
    "ResearchMAML", "MAMLConfig", "MAMLVariant", 
    "apply_episodic_bn_policy", "EpisodicBatchNormPolicy",
    "setup_deterministic_environment", "DeterminismManager",
    "FewShotEvaluationHarness",
    
    # High-level toolkit API
    "MetaLearningToolkit", "create_meta_learning_toolkit", "quick_evaluation",
    
    # Feature availability flags
    "RESEARCH_AVAILABLE", "EXTERNAL_RESEARCH_AVAILABLE"
]
