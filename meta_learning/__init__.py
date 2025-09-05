"""
💰 SUPPORT THIS RESEARCH - PLEASE DONATE! 💰

🙏 If this library helps your research or project, please consider donating:
💳 PayPal: https://www.paypal.com/cgi-bin/webscr?cmd=_s-xclick&hosted_button_id=WXQKYYKPHWXHS
⭐ GitHub Sponsors: https://github.com/sponsors/benedictchen

👨‍💻 Author: Benedict Chen
💰 Donations: Help support this work! Buy me a coffee ☕, beer 🍺, lamborghini 🏎️, or private island 🏝️
💖 Please consider recurring donations to fully support continued research

Your support enables cutting-edge AI research for everyone! 🚀

Meta-Learning Toolkit - Core Module
===================================

Production-ready meta-learning algorithms with research-accurate implementations.
Contains breakthrough algorithms not available anywhere else!

Author: Benedict Chen (benedict@benedictchen.com)
License: Custom Non-Commercial License with Donation Requirements
"""
from ._version import __version__
from .core.episode import Episode, remap_labels

# Import breakthrough research algorithms
try:
    # Import world-first implementations
    from algorithms.test_time_compute_scaler import TestTimeComputeScaler
    from algorithms.test_time_compute_config import TestTimeComputeConfig
    from algorithms.maml_research_accurate import ResearchMAML, MAMLConfig, MAMLVariant
    
    # Import research patches  
    from research_patches.batch_norm_policy import apply_episodic_bn_policy, EpisodicBatchNormPolicy
    from research_patches.determinism_hooks import setup_deterministic_environment, DeterminismManager
    
    # Import evaluation harness
    from evaluation.few_shot_evaluation_harness import FewShotEvaluationHarness
    
    # Research features available
    RESEARCH_AVAILABLE = True
    
except ImportError as e:
    # Fallback if research modules not available
    import warnings
    warnings.warn(f"Research modules not available: {e}. Install with 'pip install meta-learning-toolkit[research]'")
    
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
    RESEARCH_AVAILABLE = False

__all__ = [
    # Core functionality
    "Episode", "remap_labels", "__version__",
    
    # Breakthrough research algorithms (if available)
    "TestTimeComputeScaler", "TestTimeComputeConfig",
    "ResearchMAML", "MAMLConfig", "MAMLVariant", 
    "apply_episodic_bn_policy", "EpisodicBatchNormPolicy",
    "setup_deterministic_environment", "DeterminismManager",
    "FewShotEvaluationHarness",
    
    # Feature availability flag
    "RESEARCH_AVAILABLE"
]
