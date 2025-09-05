"""
🧠 Transformer-Based Task Adaptation
====================================

Transformer architecture for sophisticated task adaptation with cross-attention.
Uses modern transformer components for prototype adaptation.

Author: Benedict Chen (benedict@benedictchen.com)
Research Foundation: Transformer architecture (Vaswani et al. 2017)
"""

import torch
import torch.nn as nn
from typing import Dict
import math

from .task_configs import TaskAdaptiveConfig


class TransformerBasedTaskAdaptation(nn.Module):
    """
    Transformer-based Task Adaptation using cross-attention.
    
    Uses transformer architecture for sophisticated task adaptation.
    """
    
    def __init__(self, embedding_dim: int, num_layers: int = 2, 
                 attention_heads: int = 8, feedforward_dim: int = 512, config: TaskAdaptiveConfig = None):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers
        self.config = config or TaskAdaptiveConfig()
        
        # Positional encoding with sinusoidal initialization
        self.positional_encoding = nn.Parameter(
            self._create_sinusoidal_encoding(100, embedding_dim)
        )
        
        # Transformer encoder for task context
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=attention_heads,
            dim_feedforward=feedforward_dim,
            dropout=0.1,
            batch_first=True
        )
        self.task_encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        
        # Cross-attention for prototype adaptation
        self.cross_attention_layers = nn.ModuleList([
            nn.TransformerDecoderLayer(
                d_model=embedding_dim,
                nhead=attention_heads,
                dim_feedforward=feedforward_dim,
                dropout=0.1,
                batch_first=True
            ) for _ in range(num_layers)
        ])
        
        # Final output projection
        self.output_projection = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.LayerNorm(embedding_dim)
        )
    
    def _create_sinusoidal_encoding(self, max_len: int, d_model: int) -> torch.Tensor:
        """Create sinusoidal positional encoding (Vaswani et al. 2017)"""
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe
        
    def forward(self, support_features: torch.Tensor, support_labels: torch.Tensor, 
                query_features: torch.Tensor = None) -> Dict[str, torch.Tensor]:
        """
        Perform transformer-based task adaptation.
        
        Args:
            support_features: [n_support, embedding_dim]
            support_labels: [n_support]
            query_features: [n_query, embedding_dim] (optional)
            
        Returns:
            Dictionary with transformer-adapted prototypes
        """
        unique_labels = torch.unique(support_labels)
        n_classes = len(unique_labels)
        
        # Encode task context using transformer
        task_context = self.task_encoder(support_features.unsqueeze(0))  # [1, n_support, embedding_dim]
        
        # Compute base prototypes with positional encoding
        base_prototypes = []
        for i, class_idx in enumerate(unique_labels):
            class_mask = support_labels == class_idx
            class_features = support_features[class_mask]
            base_prototype = class_features.mean(dim=0)
            
            # Add positional encoding
            base_prototype = base_prototype + self.positional_encoding[i]
            base_prototypes.append(base_prototype)
        
        prototypes = torch.stack(base_prototypes).unsqueeze(0)  # [1, n_classes, embedding_dim]
        
        # Apply cross-attention layers
        current_prototypes = prototypes
        
        for cross_attn_layer in self.cross_attention_layers:
            adapted_prototypes = cross_attn_layer(
                current_prototypes,  # target (prototypes)
                task_context         # memory (task context)
            )
            current_prototypes = adapted_prototypes
        
        # Final projection
        final_prototypes = self.output_projection(current_prototypes.squeeze(0))
        
        return {
            'prototypes': final_prototypes,
            'task_context': task_context.squeeze(0),
            'num_transformer_layers': self.num_layers
        }