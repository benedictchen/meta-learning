"""
💰 SUPPORT THIS RESEARCH - PLEASE DONATE! 💰

🙏 If this library helps your research or project, please consider donating:
💳 PayPal: https://www.paypal.com/cgi-bin/webscr?cmd=_s-xclick&hosted_button_id=WXQKYYKPHWXHS
⭐ GitHub Sponsors: https://github.com/sponsors/benedictchen

👨‍💻 Author: Benedict Chen  
💰 Donations: Help support this work! Buy me a coffee ☕, beer 🍺, lamborghini 🏎️, or private island 🏝️
💖 Please consider recurring donations to fully support continued research

Your support enables cutting-edge AI research for everyone! 🚀

Research Patches Module
======================

Contains research accuracy patches recovered from pre-v3 implementation:
- BatchNorm policy fixes for few-shot learning
- Determinism hooks for reproducible research

Author: Benedict Chen (benedict@benedictchen.com)
License: Custom Non-Commercial License with Donation Requirements
"""

from .batch_norm_policy import EpisodicBatchNormPolicy, apply_episodic_bn_policy
from .determinism_hooks import DeterminismManager, setup_deterministic_environment

__all__ = [
    "EpisodicBatchNormPolicy",
    "apply_episodic_bn_policy",
    "DeterminismManager", 
    "setup_deterministic_environment"
]