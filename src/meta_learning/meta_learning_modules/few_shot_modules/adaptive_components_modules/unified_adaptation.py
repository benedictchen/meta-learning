"""
🧠 Unified Task-Adaptive Prototypes
===================================

Unified interface for all task adaptation methods with factory functions.
Provides a single entry point for different adaptation approaches.

Author: Benedict Chen (benedict@benedictchen.com)
Research Foundation: Combined work from MAML, attention mechanisms, and transformer architectures
"""

import torch
import torch.nn as nn
from typing import Dict, Optional

from .task_configs import TaskAdaptiveConfig
from .attention_adaptation import AttentionBasedTaskAdaptation
from .meta_learning_adaptation import MetaLearningTaskAdaptation
from .context_adaptation import ContextDependentTaskAdaptation
from .transformer_adaptation import TransformerBasedTaskAdaptation


class TaskAdaptivePrototypes(nn.Module):
    """
    Unified Task-Adaptive Prototypes Module.
    
    Supports multiple task adaptation methods with configurable options.
    """
    
    def __init__(self, embedding_dim: int, config: TaskAdaptiveConfig = None):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.config = config or TaskAdaptiveConfig()
        
        # Initialize adaptation method based on configuration
        if self.config.method == "attention_based":
            self.adaptation_module = AttentionBasedTaskAdaptation(
                embedding_dim, self.config.adaptation_layers, 
                self.config.attention_heads, self.config.adaptation_dropout, self.config
            )
        elif self.config.method == "meta_learning":
            self.adaptation_module = MetaLearningTaskAdaptation(
                embedding_dim, self.config.meta_lr, self.config.adaptation_steps, self.config
            )
        elif self.config.method == "context_dependent":
            self.adaptation_module = ContextDependentTaskAdaptation(
                embedding_dim, self.config.adaptation_dim, self.config.context_pooling, self.config
            )
        elif self.config.method == "transformer_based":
            self.adaptation_module = TransformerBasedTaskAdaptation(
                embedding_dim, self.config.adaptation_layers, self.config.attention_heads, config=self.config
            )
        else:
            raise ValueError(f"Unknown adaptation method: {self.config.method}")
        
        # Optional residual connection
        if self.config.use_residual_adaptation:
            self.residual_weight = nn.Parameter(torch.tensor(0.5))
        
        # Temperature scaling
        self.temperature = nn.Parameter(torch.ones(1) * self.config.temperature)
        
    def forward(self, support_features: torch.Tensor, support_labels: torch.Tensor, 
                query_features: torch.Tensor = None) -> Dict[str, torch.Tensor]:
        """
        Perform task adaptation using the configured method.
        
        Args:
            support_features: [n_support, embedding_dim]
            support_labels: [n_support]
            query_features: [n_query, embedding_dim] (optional)
            
        Returns:
            Dictionary with task-adapted prototypes and method-specific information
        """
        # Get base prototypes for residual connection
        if self.config.use_residual_adaptation:
            unique_labels = torch.unique(support_labels)
            base_prototypes = []
            for class_idx in unique_labels:
                class_mask = support_labels == class_idx
                class_features = support_features[class_mask]
                base_prototype = class_features.mean(dim=0)
                base_prototypes.append(base_prototype)
            base_prototypes = torch.stack(base_prototypes)
        
        # Apply adaptation method
        result = self.adaptation_module(support_features, support_labels, query_features)
        
        # Apply residual connection if enabled
        if self.config.use_residual_adaptation:
            adapted_prototypes = (self.residual_weight * result['prototypes'] + 
                                (1 - self.residual_weight) * base_prototypes)
            result['prototypes'] = adapted_prototypes
            result['residual_weight'] = self.residual_weight.item()
        
        # Apply temperature scaling
        result['prototypes'] = result['prototypes'] / self.temperature
        result['temperature'] = self.temperature.item()
        result['adaptation_method'] = self.config.method
        
        return result


# Factory functions for easy creation
def create_task_adaptive_prototypes(method: str = "attention_based", 
                                   embedding_dim: int = 512, 
                                   **kwargs) -> TaskAdaptivePrototypes:
    """Factory function to create task-adaptive prototype modules."""
    config = TaskAdaptiveConfig(method=method, **kwargs)
    return TaskAdaptivePrototypes(embedding_dim, config)


def create_attention_adaptive_prototypes(embedding_dim: int, adaptation_layers: int = 2, 
                                       attention_heads: int = 8) -> TaskAdaptivePrototypes:
    """Create attention-based task-adaptive prototypes."""
    config = TaskAdaptiveConfig(
        method="attention_based",
        adaptation_layers=adaptation_layers,
        attention_heads=attention_heads
    )
    return TaskAdaptivePrototypes(embedding_dim, config)


def create_meta_adaptive_prototypes(embedding_dim: int, meta_lr: float = 0.01, 
                                   adaptation_steps: int = 5) -> TaskAdaptivePrototypes:
    """Create meta-learning based task-adaptive prototypes."""
    config = TaskAdaptiveConfig(
        method="meta_learning",
        meta_lr=meta_lr,
        adaptation_steps=adaptation_steps
    )
    return TaskAdaptivePrototypes(embedding_dim, config)


def create_context_adaptive_prototypes(embedding_dim: int, adaptation_dim: int = 128, 
                                     context_pooling: str = "attention") -> TaskAdaptivePrototypes:
    """Create context-dependent task-adaptive prototypes."""
    config = TaskAdaptiveConfig(
        method="context_dependent",
        adaptation_dim=adaptation_dim,
        context_pooling=context_pooling
    )
    return TaskAdaptivePrototypes(embedding_dim, config)


def create_transformer_adaptive_prototypes(embedding_dim: int, num_layers: int = 2, 
                                         attention_heads: int = 8) -> TaskAdaptivePrototypes:
    """Create transformer-based task-adaptive prototypes."""
    config = TaskAdaptiveConfig(
        method="transformer_based",
        adaptation_layers=num_layers,
        attention_heads=attention_heads
    )
    return TaskAdaptivePrototypes(embedding_dim, config)