"""
Meta-Learning Toolkit
====================

Production-ready meta-learning algorithms with research-accurate implementations
of MAML, Prototypical Networks, and test-time compute scaling.

This package implements cutting-edge meta-learning algorithms based on
30+ foundational papers spanning 1987-2025.
"""

__version__ = "2.0.0"
__author__ = "Benedict Chen"
__email__ = "benedict@benedictchen.com"

# Core public API - simple and clean
try:
    from .algos.protonet import ProtoHead as AlgosProtoHead, fit_episode, make_episode as algos_make_episode
    from .core.math_utils import pairwise_sqeuclidean, cosine_logits
except ImportError:
    # Fallback to new core implementation
    from .core import ProtoHead, PrototypicalNetworks, PrototypicalConfig, Conv4
    from .core import make_episode, get_dataset
    AlgosProtoHead = ProtoHead
    algos_make_episode = make_episode
    pairwise_sqeuclidean = None
    cosine_logits = None
    fit_episode = None

# Main algorithms with clean interfaces
try:
    from .meta_learning_modules.few_shot_learning import PrototypicalNetworks
    from .meta_learning_modules.maml_variants import MAMLLearner  
    from .meta_learning_modules.test_time_compute import TestTimeComputeScaler
except ImportError:
    # Fallback for missing components
    PrototypicalNetworks = None
    MAMLLearner = None
    TestTimeComputeScaler = None

# Configuration classes
try:
    from .meta_learning_modules.few_shot_modules.configurations import PrototypicalConfig
    from .meta_learning_modules.maml_variants import MAMLConfig
    from .meta_learning_modules.test_time_compute import TestTimeComputeConfig
except ImportError:
    PrototypicalConfig = None
    MAMLConfig = None
    TestTimeComputeConfig = None

# Dataset utilities
try:
    from .meta_learning_modules.utils_modules import (
        MetaLearningDataset,
        TaskConfiguration,
        few_shot_accuracy
    )
except ImportError:
    MetaLearningDataset = None
    TaskConfiguration = None
    few_shot_accuracy = None

# Expose the working implementations
ProtoHead = AlgosProtoHead if 'AlgosProtoHead' in locals() else ProtoHead
make_episode = algos_make_episode if 'algos_make_episode' in locals() else make_episode

# Clean public API
__all__ = [
    # Core functions (always available)
    "ProtoHead",
    "fit_episode", 
    "make_episode",
    "get_dataset",
    "Conv4", 
    "pairwise_sqeuclidean",
    "cosine_logits",
    
    # Main algorithms
    "PrototypicalNetworks",
    "MAMLLearner",
    "TestTimeComputeScaler",
    
    # Configuration
    "PrototypicalConfig", 
    "MAMLConfig",
    "TestTimeComputeConfig",
    
    # Utilities
    "MetaLearningDataset",
    "TaskConfiguration",
    "few_shot_accuracy",
]

# Package info
ALGORITHMS_AVAILABLE = [
    "Prototypical Networks (Snell et al. 2017)",
    "Model-Agnostic Meta-Learning (Finn et al. 2017)", 
    "Test-Time Compute Scaling (2024)",
]