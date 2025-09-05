"""
🧠 Context-Dependent Task Adaptation
====================================

Context-dependent adaptation mechanisms based on TADAM and related work.
Implements global and local context encoding with feature modulation.

Author: Benedict Chen (benedict@benedictchen.com)
Research Foundation: TADAM (Oreshkin et al. 2018), Context Networks (Ren et al. 2018)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict

from .task_configs import TaskAdaptiveConfig


class ContextDependentTaskAdaptation(nn.Module):
    """
    Context-Dependent Task Adaptation based on TADAM.
    
    Based on: Oreshkin et al. (2018) "TADAM: Task dependent adaptive metric for improved few-shot learning"
    
    Implementation combines:
    - Global task context: Encodes task-level statistics from support set
    - Local class context: Encodes class-specific prototype information  
    - Feature modulation: Adapts prototypes using context-dependent transformations
    """
    
    def __init__(self, embedding_dim: int, adaptation_dim: int = 128, 
                 context_pooling: str = "attention", config: TaskAdaptiveConfig = None):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.adaptation_dim = adaptation_dim
        self.context_pooling = context_pooling
        self.config = config or TaskAdaptiveConfig()
        
        # Global context encoder
        self.global_context_encoder = nn.Sequential(
            nn.Linear(embedding_dim, adaptation_dim),
            nn.ReLU(),
            nn.Linear(adaptation_dim, adaptation_dim)
        )
        
        # Local context encoder
        self.local_context_encoder = nn.Sequential(
            nn.Linear(embedding_dim, adaptation_dim),
            nn.ReLU(), 
            nn.Linear(adaptation_dim, adaptation_dim)
        )
        
        # Context pooling mechanisms
        if context_pooling == "attention":
            self.context_attention = nn.MultiheadAttention(
                adaptation_dim, num_heads=4, batch_first=True
            )
        elif context_pooling == "learned":
            # Use Xavier uniform initialization instead of torch.randn
            self.context_pooling_weights = nn.Parameter(torch.empty(adaptation_dim))
            nn.init.xavier_uniform_(self.context_pooling_weights.unsqueeze(0))
            self.context_pooling_weights = nn.Parameter(self.context_pooling_weights.squeeze(0))
        
        # Context fusion network
        self.context_fusion = nn.Sequential(
            nn.Linear(adaptation_dim * 2, adaptation_dim),
            nn.ReLU(),
            nn.Linear(adaptation_dim, embedding_dim)
        )
        
        # Prototype adaptation network
        self.prototype_adaptation = nn.Sequential(
            nn.Linear(embedding_dim * 2, embedding_dim),  # [prototype, context]
            nn.LayerNorm(embedding_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(embedding_dim, embedding_dim)
        )
        
    def _pool_context(self, contexts: torch.Tensor) -> torch.Tensor:
        """Pool context features using the specified method."""
        if self.context_pooling == "mean":
            return contexts.mean(dim=0)
        elif self.context_pooling == "max":
            return contexts.max(dim=0)[0]
        elif self.context_pooling == "attention":
            # Self-attention pooling
            pooled, _ = self.context_attention(
                contexts.unsqueeze(0), contexts.unsqueeze(0), contexts.unsqueeze(0)
            )
            return pooled.squeeze(0).mean(dim=0)
        elif self.context_pooling == "learned":
            # Learned weighted pooling
            weights = F.softmax(torch.matmul(contexts, self.context_pooling_weights), dim=0)
            return torch.sum(contexts * weights.unsqueeze(1), dim=0)
        else:
            return contexts.mean(dim=0)
    
    def forward(self, support_features: torch.Tensor, support_labels: torch.Tensor, 
                query_features: torch.Tensor = None) -> Dict[str, torch.Tensor]:
        """
        Perform context-dependent task adaptation.
        
        Args:
            support_features: [n_support, embedding_dim]
            support_labels: [n_support]
            query_features: [n_query, embedding_dim] (optional)
            
        Returns:
            Dictionary with context-adapted prototypes
        """
        unique_labels = torch.unique(support_labels)
        n_classes = len(unique_labels)
        
        # Encode global task context
        global_contexts = self.global_context_encoder(support_features)
        global_context = self._pool_context(global_contexts)
        
        # Adapt prototypes for each class
        adapted_prototypes = []
        local_contexts = []
        
        for class_idx in unique_labels:
            class_mask = support_labels == class_idx
            class_features = support_features[class_mask]
            
            # Compute base prototype
            base_prototype = class_features.mean(dim=0)
            
            # Encode local class context
            class_local_contexts = self.local_context_encoder(class_features)
            local_context = self._pool_context(class_local_contexts)
            local_contexts.append(local_context)
            
            # Fuse global and local contexts
            fused_context = self.context_fusion(
                torch.cat([global_context, local_context])
            )
            
            # Adapt prototype using fused context
            adaptation_input = torch.cat([base_prototype, fused_context])
            adapted_prototype = self.prototype_adaptation(adaptation_input)
            
            adapted_prototypes.append(adapted_prototype)
        
        return {
            'prototypes': torch.stack(adapted_prototypes),
            'global_context': global_context,
            'local_contexts': torch.stack(local_contexts),
            'pooling_method': self.context_pooling
        }