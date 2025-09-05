"""
💰 SUPPORT THIS RESEARCH - PLEASE DONATE! 💰

🙏 If this library helps your research or project, please consider donating:
💳 PayPal: https://www.paypal.com/cgi-bin/webscr?cmd=_s-xclick&hosted_button_id=WXQKYYKPHWXHS
⭐ GitHub Sponsors: https://github.com/sponsors/benedictchen

👨‍💻 Author: Benedict Chen
💰 Donations: Help support this work! Buy me a coffee ☕, beer 🍺, lamborghini 🏎️, or private island 🏝️
💖 Please consider recurring donations to fully support continued research

Your support enables cutting-edge AI research for everyone! 🚀

Recovered Algorithms Module
==========================

Contains the breakthrough algorithms recovered from pre-v3 implementation:
- TestTimeComputeScaler (2024 world-first)
- Research-Accurate MAML with all variants
- Related configuration and factory functions

Author: Benedict Chen (benedict@benedictchen.com)
License: Custom Non-Commercial License with Donation Requirements
"""

from .test_time_compute_scaler import TestTimeComputeScaler
from .test_time_compute_config import TestTimeComputeConfig
from .maml_research_accurate import (
    ResearchMAML, 
    MAMLConfig, 
    MAMLVariant, 
    FunctionalModule
)

__all__ = [
    "TestTimeComputeScaler",
    "TestTimeComputeConfig", 
    "ResearchMAML",
    "MAMLConfig",
    "MAMLVariant",
    "FunctionalModule"
]