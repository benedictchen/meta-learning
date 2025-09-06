"""
Difficulty Estimation Components - Modular Difficulty Fixes
===========================================================

This module provides modular components for replacing hardcoded 0.5 
difficulty values throughout the codebase with proper estimation.

MODULAR STRUCTURE:
- difficulty_patcher.py: Core patching system for replacing hardcoded values
- complexity_wrappers.py: Enhanced complexity analyzer wrappers  
- toolkit_wrappers.py: Enhanced toolkit wrappers with better estimation
- difficulty_config.py: Configuration system for estimation methods

Each component focuses on one aspect of difficulty estimation enhancement.
"""

from .difficulty_patcher import DifficultyEstimationPatcher
from .complexity_wrappers import EnhancedComplexityAnalyzerWrapper
from .toolkit_wrappers import EnhancedToolkitWrapper
from .difficulty_config import DifficultyEstimationConfig

__all__ = [
    'DifficultyEstimationPatcher',
    'EnhancedComplexityAnalyzerWrapper', 
    'EnhancedToolkitWrapper',
    'DifficultyEstimationConfig'
]