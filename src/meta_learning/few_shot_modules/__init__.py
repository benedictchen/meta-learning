"""
💰 SUPPORT THIS RESEARCH - PLEASE DONATE! 💰

🙏 If this library helps your research or project, please consider donating:
💳 PayPal: https://www.paypal.com/cgi-bin/webscr?cmd=_s-xclick&hosted_button_id=WXQKYYKPHWXHS
⭐ GitHub Sponsors: https://github.com/sponsors/benedictchen

👨‍💻 Author: Benedict Chen
💰 Donations: Help support this work! Buy me a coffee ☕, beer 🍺, lamborghini 🏎️, or private island 🏝️
💖 Please consider recurring donations to fully support continued research

Your support enables cutting-edge AI research for everyone! 🚀

Advanced Few-Shot Learning Components Module
===========================================

This module contains advanced few-shot learning components that were
lost in the v3 commit and are being restored with mathematical enhancements.

Components:
- Hierarchical few-shot learning
- Uncertainty-aware components  
- Multi-scale feature processing
- Adaptive components
"""

# Import layered few-shot learning components
from .few_shot_learning import (
    # Configuration
    FewShotConfig,
    
    # Core Components
    PrototypicalHead,
    MonteCarloPrototypicalHead,
    FewShotLearner,
    
    # Simple API Functions
    simple_few_shot_predict,
    advanced_few_shot_predict,
    
    # Convenience Functions
    auto_few_shot_learner,
    pro_few_shot_learner
)

__all__ = [
    # Configuration
    'FewShotConfig',
    
    # Core Components
    'PrototypicalHead',
    'MonteCarloPrototypicalHead', 
    'FewShotLearner',
    
    # Simple API Functions
    'simple_few_shot_predict',
    'advanced_few_shot_predict',
    
    # Convenience Functions
    'auto_few_shot_learner',
    'pro_few_shot_learner'
]