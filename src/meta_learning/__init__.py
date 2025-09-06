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

# Core essential components that should always work
from .shared.types import Episode
from .core.utils import partition_task, remap_labels
from .core.seed import seed_all
from .core.math_utils import pairwise_sqeuclidean, cosine_logits

# Essential data utilities
from .data import InfiniteIterator, OnDeviceDataset

# Model architectures - recently implemented and tested
from .models.conv4 import Conv4, ResNet12, WideResNet, MetaModule, MetaLinear, MetaConv2d

# Basic algorithms  
from .algorithms import ProtoHead, inner_adapt_and_eval, meta_outer_step

# Evaluation
from .eval import evaluate

# Benchmarking
from .bench import run_benchmark

# Core validation functionality
from .validation import (
    ValidationError, ConfigurationWarning, validate_episode_tensors,
    validate_few_shot_configuration, validate_distance_metric, 
    validate_temperature_parameter, validate_learning_rate
)

__all__ = [
    "Episode",
    "remap_labels",
    "__version__",
    "partition_task",
    "seed_all",
    "pairwise_sqeuclidean",
    "cosine_logits",
    "InfiniteIterator",
    "OnDeviceDataset",
    "Conv4", 
    "ResNet12",
    "WideResNet", 
    "MetaModule",
    "MetaLinear",
    "MetaConv2d",
    "ProtoHead",
    "inner_adapt_and_eval",
    "meta_outer_step",
    "evaluate",
    "run_benchmark",
    "ValidationError",
    "ConfigurationWarning",
    "validate_episode_tensors",
    "validate_few_shot_configuration",
    "validate_distance_metric",
    "validate_temperature_parameter",
    "validate_learning_rate"
]