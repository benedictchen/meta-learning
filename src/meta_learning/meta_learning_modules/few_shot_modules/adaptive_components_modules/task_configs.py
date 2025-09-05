"""
🧠 Task-Adaptive Configuration and Enums
========================================

This module contains configuration classes and enums for task-adaptive components.
Separated from the main implementation for better modularity and maintainability.

Author: Benedict Chen (benedict@benedictchen.com)
Based on: MAML, Prototypical Networks, and modern attention-based meta-learning
"""

from dataclasses import dataclass
from enum import Enum


class TaskContextMethod(Enum):
    """Task context encoding methods for adaptive components."""
    MAML_GRADIENT = "maml_gradient"                   # Gradient-based encoding
    TASK_STATISTICS = "task_statistics"               # Statistical features
    CROSS_CLASS_INTERACTION = "cross_class_interaction" # Inter-class patterns
    SUPPORT_QUERY_JOINT = "support_query_joint"       # Joint encoding


class AdaptiveAttentionMethod(Enum):
    """Attention mechanisms for adaptive components."""
    SELF_ATTENTION = "self_attention"                 # Standard self-attention
    TASK_CONDITIONED = "task_conditioned"             # Task-specific attention
    CROSS_ATTENTION = "cross_attention"               # Cross-modal attention
    LEARNABLE_MIXING = "learnable_mixing"             # Learnable attention mixing


@dataclass
class TaskAdaptiveConfig:
    """
    Configuration for task-adaptive prototype solutions.
    
    All solutions are based on research-accurate implementations.
    """
    # Core method selection
    method: str = "attention_based"  # attention_based, meta_learning, context_dependent, transformer_based
    
    # Task Context Encoding
    task_context_method: TaskContextMethod = TaskContextMethod.MAML_GRADIENT
    task_context_temperature: float = 1.0
    
    # Adaptive Attention
    adaptive_attention_method: AdaptiveAttentionMethod = AdaptiveAttentionMethod.SELF_ATTENTION
    attention_heads: int = 8
    attention_dropout: float = 0.1
    
    maml_method: str = "finn_2017"  # "finn_2017", "nichol_2018_reptile", "triantafillou_2019" 
    
    task_context_method: str = "ravi_2017_fisher"  # "ravi_2017_fisher", "vinyals_2015_set2set", "sung_2018_relational"
    
    # General adaptation parameters
    adaptation_layers: int = 2
    attention_heads: int = 8
    adaptation_dim: int = 128
    meta_lr: float = 0.01
    temperature: float = 1.0
    context_pooling: str = "attention"  # mean, attention, max, learned
    adaptation_dropout: float = 0.1
    use_residual_adaptation: bool = True
    adaptation_steps: int = 5
    normalization: str = "layer"  # layer, batch, instance
    
    # Task context encoding method selection
    task_context_method: str = "maml_gradient"  # maml_gradient, task_statistics, cross_class_interaction, support_query_joint
    
    # Attention mechanism selection  
    attention_mechanism: str = "self_attention"  # self_attention, task_conditioned, cross_attention, learnable_mixing
    
    # Prototype adaptation method
    prototype_adaptation: str = "standard"  # standard, task_conditioned, hierarchical, distance_weighted