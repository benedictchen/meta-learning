"""
📋 Strategies
==============

🎯 ELI5 Summary:
This file is an important component in our AI research system! Like different organs 
in your body that work together to keep you healthy, this file has a specific job that 
helps the overall algorithm work correctly and efficiently.

🧪 Technical Details:
===================
Implementation details and technical specifications for this component.
Designed to work seamlessly within the research framework while
maintaining high performance and accuracy standards.

📋 Component Integration:
========================
    ┌──────────┐
    │   This   │
    │Component │ ←→ Other Components
    └──────────┘
         ↑↓
    System Integration

"""
"""
Test-Time Compute Strategies and Enums
=======================================

Strategy definitions for test-time compute scaling based on research.

Author: Benedict Chen (benedict@benedictchen.com)
Based on: Snell et al. (2024) "Scaling LLM Test-Time Compute Optimally"
"""

from enum import Enum


class TestTimeComputeStrategy(Enum):
    """Test-time compute scaling strategies based on research."""
    BASIC = "basic"                         # Basic adaptive allocation
    PROCESS_REWARD = "process_reward"       # Process-based reward models (Snell et al. 2024)
    VERIFIER = "verifier"                   # Same as process_reward (backward compatibility)
    CONSISTENCY_VERIFICATION = "consistency" # Consistency-based verification
    GRADIENT_VERIFICATION = "gradient"      # Gradient-based verification
    ATTENTION_REASONING = "attention"       # Attention-based reasoning
    FEATURE_REASONING = "feature"           # Feature-based reasoning
    PROTOTYPE_REASONING = "prototype"       # Prototype-based reasoning
    SEARCH = "search"                       # Tree search with verification (o1 style) 
    CHAIN_OF_THOUGHT = "cot"               # Chain-of-thought reasoning
    TEST_TIME_TRAINING = "ttt"             # Test-time training (Akyürek et al. 2024)
    MANY_SHOT = "many_shot"                # Many-shot in-context learning
    ENSEMBLE = "ensemble"                  # Multi-model ensemble at test time
    HYBRID = "hybrid"                      # Combination of multiple strategies


class StateFallbackMethod(Enum):
    """Fallback methods when state encoding fails."""
    ZERO = "zero"                   # Zero tensor fallback
    RANDOM = "random"               # Random initialization
    LEARNED = "learned"             # Use learned embedding
    PREVIOUS = "previous"           # Use previous state


class StateForwardMethod(Enum):
    """Methods for forward state processing."""
    DIRECT = "direct"               # Direct tensor processing
    ATTENTION = "attention"         # Attention-based processing
    GRAPH = "graph"                 # Graph neural network
    TRANSFORMER = "transformer"     # Transformer-based encoding


class VerificationFallbackMethod(Enum):
    """Fallback methods for verification when primary fails."""
    SKIP = "skip"                   # Skip verification step
    SIMPLE = "simple"               # Use simple heuristic
    LEARNED = "learned"             # Use learned verifier
    CONSENSUS = "consensus"         # Multi-verifier consensus
    GRADIENT = "gradient"           # Gradient-based verification