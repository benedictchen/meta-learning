"""
🧠 Attention-Based Task Adaptation
==================================

Implementation of attention-based task adaptation mechanisms for few-shot learning.
Based on research from Baik et al. (2020) and related attention-based meta-learning work.

Author: Benedict Chen (benedict@benedictchen.com)
Research Foundation: Attention mechanisms in meta-learning (Vaswani et al. 2017, Baik et al. 2020)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple
import math
import warnings

from .task_configs import TaskAdaptiveConfig


class AttentionBasedTaskAdaptation(nn.Module):
    """
    Attention-based Task Adaptive Prototypes (Baik et al., 2020).
    
    Uses multi-head attention to adapt prototypes based on task context.
    """
    
    def __init__(self, embedding_dim: int, adaptation_layers: int = 2, 
                 attention_heads: int = 8, adaptation_dropout: float = 0.1, 
                 config: TaskAdaptiveConfig = None):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.adaptation_layers = adaptation_layers
        self.attention_heads = attention_heads
        self.config = config or TaskAdaptiveConfig()
        
        # Task context encoder
        self.task_encoder = nn.Sequential(*[
            nn.Sequential(
                nn.Linear(embedding_dim, embedding_dim),
                nn.LayerNorm(embedding_dim),
                nn.ReLU(),
                nn.Dropout(adaptation_dropout)
            ) for _ in range(adaptation_layers)
        ])
        
        # Multi-head attention for prototype adaptation
        self.adaptation_attention = nn.MultiheadAttention(
            embedding_dim, attention_heads, batch_first=True, dropout=adaptation_dropout
        )
        
        # Context-aware transformation
        self.context_transform = nn.Sequential(
            nn.Linear(embedding_dim * 2, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim)
        )
        
        # Adaptation gate
        self.adaptation_gate = nn.Sequential(
            nn.Linear(embedding_dim * 2, embedding_dim),
            nn.Sigmoid()
        )
        
        # Output projection
        self.output_projection = nn.Linear(embedding_dim, embedding_dim)
        
    def forward(self, support_features: torch.Tensor, support_labels: torch.Tensor, 
                query_features: torch.Tensor = None) -> Dict[str, torch.Tensor]:
        """
        Perform attention-based task adaptation.
        
        Args:
            support_features: [n_support, embedding_dim]
            support_labels: [n_support]
            query_features: [n_query, embedding_dim] (optional)
            
        Returns:
            Dictionary with adapted prototypes and attention information
        """
        unique_labels = torch.unique(support_labels)
        n_classes = len(unique_labels)
        
        # Configurable task context encoding methods
        if self.config.task_context_method == "maml_gradient":
            task_context = self._compute_maml_gradient_encoding(support_features, support_labels)
        elif self.config.task_context_method == "task_statistics":
            task_context = self._compute_task_statistics_encoding(support_features, support_labels, unique_labels)
        elif self.config.task_context_method == "cross_class_interaction":
            task_context = self._compute_cross_class_interaction_encoding(support_features, support_labels, unique_labels)
        elif self.config.task_context_method == "support_query_joint":
            task_context = self._compute_support_query_joint_encoding(support_features, support_labels, query_features)
        else:
            # Default: use task statistics method (research-based fallback)
            task_context = self._compute_task_statistics_encoding(support_features, support_labels, unique_labels)
        
        # Base prototypes: standard Snell et al. (2017) class means
        base_prototypes = []
        for class_idx in unique_labels:
            class_mask = support_labels == class_idx
            class_features = support_features[class_mask]
            base_prototype = class_features.mean(dim=0)
            base_prototypes.append(base_prototype)
        
        base_prototypes = torch.stack(base_prototypes)  # [n_classes, embedding_dim]
        
        # Configurable attention mechanisms
        if self.config.attention_mechanism == "self_attention":
            attended_prototypes, attention_weights = self._compute_self_attention(base_prototypes)
        elif self.config.attention_mechanism == "task_conditioned":
            attended_prototypes, attention_weights = self._compute_task_conditioned_adaptation(base_prototypes, task_context)
        elif self.config.attention_mechanism == "cross_attention":
            attended_prototypes, attention_weights = self._compute_cross_attention_adaptation(base_prototypes, support_features)
        elif self.config.attention_mechanism == "learnable_mixing":
            attended_prototypes, attention_weights = self._compute_learnable_mixing(base_prototypes)
        else:
            # Fallback to simple identity (no attention)
            attended_prototypes = base_prototypes
            attention_weights = torch.ones(base_prototypes.size(0), base_prototypes.size(0), device=base_prototypes.device) / base_prototypes.size(0)
        
        # Context-aware transformation
        context_input = torch.cat([
            base_prototypes, 
            task_context.repeat(n_classes, 1)
        ], dim=1)  # [n_classes, embedding_dim * 2]
        
        context_adapted = self.context_transform(context_input)  # [n_classes, embedding_dim]
        
        # Gated combination
        gate_input = torch.cat([attended_prototypes, context_adapted], dim=1)
        adaptation_gate = self.adaptation_gate(gate_input)  # [n_classes, embedding_dim]
        
        adapted_prototypes = (adaptation_gate * attended_prototypes + 
                            (1 - adaptation_gate) * context_adapted)
        
        # Final projection
        final_prototypes = self.output_projection(adapted_prototypes)
        
        return {
            'prototypes': final_prototypes,
            'base_prototypes': base_prototypes,
            'attention_weights': attention_weights.squeeze(0),
            'adaptation_gates': adaptation_gate,
            'task_context': task_context.squeeze(0)
        }
    
    def _compute_maml_gradient_encoding(self, support_features: torch.Tensor, support_labels: torch.Tensor) -> torch.Tensor:
        """SOLUTION 1: MAML-Style Task Gradient Encoding (Finn et al. 2017)"""
        try:
            # Compute task-specific loss
            unique_labels = torch.unique(support_labels)
            
            # Create temporary prototypes for loss computation
            prototypes = []
            for class_idx in unique_labels:
                class_mask = support_labels == class_idx
                class_features = support_features[class_mask]
                prototypes.append(class_features.mean(dim=0))
            
            prototypes = torch.stack(prototypes)
            
            # Compute distances and loss for gradient computation
            distances = torch.cdist(support_features, prototypes, p=2)
            targets = torch.zeros_like(distances)
            for i, label in enumerate(support_labels):
                targets[i, label] = 1.0
            
            task_loss = F.mse_loss(distances, targets)
            
            # Compute gradients with respect to task encoder
            if self.task_encoder.parameters() and any(p.requires_grad for p in self.task_encoder.parameters()):
                task_gradients = torch.autograd.grad(task_loss, self.task_encoder.parameters(), 
                                                   create_graph=True, retain_graph=True, allow_unused=True)
                
                # Filter out None gradients and flatten
                valid_gradients = [g.flatten() for g in task_gradients if g is not None]
                if valid_gradients:
                    task_context = torch.cat(valid_gradients)
                    # Project to embedding dimension
                    if hasattr(self, 'gradient_encoder'):
                        task_context = self.gradient_encoder(task_context.unsqueeze(0))
                    else:
                        # Fallback: use linear layer to project to correct size
                        if not hasattr(self, '_gradient_proj'):
                            self._gradient_proj = nn.Linear(task_context.size(0), self.embedding_dim).to(task_context.device)
                        task_context = self._gradient_proj(task_context.unsqueeze(0))
                else:
                    # No gradients available, fallback to mean
                    task_context = self.task_encoder(support_features.mean(dim=0, keepdim=True))
            else:
                # No parameters to compute gradients from, fallback to mean
                task_context = self.task_encoder(support_features.mean(dim=0, keepdim=True))
                
            return task_context
            
        except Exception as e:
            warnings.warn(f"MAML gradient encoding failed: {e}, falling back to mean")
            return self.task_encoder(support_features.mean(dim=0, keepdim=True))
    
    def _compute_task_statistics_encoding(self, support_features: torch.Tensor, support_labels: torch.Tensor, unique_labels: torch.Tensor) -> torch.Tensor:
        """SOLUTION 2: Task Statistics Encoding (Hospedales et al. 2020)"""
        try:
            # Compute class-aware task statistics
            class_means = []
            class_vars = []
            
            for class_idx in unique_labels:
                class_mask = support_labels == class_idx
                class_feats = support_features[class_mask]
                class_means.append(class_feats.mean(dim=0))
                class_vars.append(class_feats.var(dim=0))
            
            class_means = torch.stack(class_means)
            class_vars = torch.stack(class_vars)
            
            # Aggregate task statistics
            task_stats = torch.cat([
                class_means.mean(dim=0),  # Inter-class centroid
                class_vars.mean(dim=0),   # Average intra-class variance  
                class_means.var(dim=0)    # Inter-class variance
            ])
            
            task_context = self.task_encoder(task_stats.unsqueeze(0))
            return task_context
            
        except Exception as e:
            warnings.warn(f"Task statistics encoding failed: {e}, falling back to mean")
            return self.task_encoder(support_features.mean(dim=0, keepdim=True))
    
    def _compute_cross_class_interaction_encoding(self, support_features: torch.Tensor, support_labels: torch.Tensor, unique_labels: torch.Tensor) -> torch.Tensor:
        """SOLUTION 3: Cross-Class Interaction Encoding (Lee et al. 2019)"""
        try:
            # Encode relationships between classes
            prototype_embeddings = []
            for class_idx in unique_labels:
                class_mask = support_labels == class_idx
                class_feats = support_features[class_mask]
                prototype_embeddings.append(class_feats.mean(dim=0))
            
            prototypes = torch.stack(prototype_embeddings)  # [n_classes, embedding_dim]
            
            # Use Set Transformer-style encoding if available, otherwise simple attention
            if hasattr(self, 'set_transformer_encoder'):
                task_context = self.set_transformer_encoder(prototypes.unsqueeze(0))  # [1, embedding_dim]
            else:
                # Fallback: Use self-attention to encode prototype relationships
                if not hasattr(self, '_proto_attention'):
                    self._proto_attention = nn.MultiheadAttention(
                        self.embedding_dim, num_heads=self.config.attention_heads, 
                        batch_first=True
                    ).to(prototypes.device)
                
                # Self-attention on prototypes
                attended_protos, _ = self._proto_attention(
                    prototypes.unsqueeze(0), prototypes.unsqueeze(0), prototypes.unsqueeze(0)
                )
                task_context = attended_protos.mean(dim=1)  # [1, embedding_dim]
                
            return task_context
            
        except Exception as e:
            warnings.warn(f"Cross-class interaction encoding failed: {e}, falling back to mean")
            return self.task_encoder(support_features.mean(dim=0, keepdim=True))
    
    def _compute_support_query_joint_encoding(self, support_features: torch.Tensor, support_labels: torch.Tensor, query_features: torch.Tensor) -> torch.Tensor:
        """SOLUTION 4: Support-Query Joint Encoding (Sung et al. 2018)"""
        try:
            if query_features is not None and query_features.size(0) > 0:
                # Create all support-query pairs
                n_support, n_query = support_features.size(0), query_features.size(0)
                support_expanded = support_features.unsqueeze(1).expand(-1, n_query, -1)
                query_expanded = query_features.unsqueeze(0).expand(n_support, -1, -1)
                
                # Compute pairwise relations (following Sung et al. 2018)
                relations = torch.cat([
                    support_expanded, 
                    query_expanded, 
                    support_expanded - query_expanded, 
                    support_expanded * query_expanded
                ], dim=-1)
                
                # Aggregate relations as task context
                if hasattr(self, 'relation_encoder'):
                    task_context = self.relation_encoder(relations.mean(dim=(0,1))).unsqueeze(0)
                else:
                    # Fallback: use task encoder on aggregated relations
                    relation_features = relations.mean(dim=(0,1))
                    # Project to correct size if needed
                    if relation_features.size(-1) != self.embedding_dim:
                        if not hasattr(self, '_relation_proj'):
                            self._relation_proj = nn.Linear(relation_features.size(-1), self.embedding_dim).to(relation_features.device)
                        relation_features = self._relation_proj(relation_features)
                    task_context = relation_features.unsqueeze(0)
                    
            else:
                # Fallback: Use intra-support relations only
                task_context = self._encode_intra_support_relations(support_features, support_labels)
                
            return task_context
            
        except Exception as e:
            warnings.warn(f"Support-query joint encoding failed: {e}, falling back to mean")
            return self.task_encoder(support_features.mean(dim=0, keepdim=True))
    
    def _encode_intra_support_relations(self, support_features: torch.Tensor, support_labels: torch.Tensor) -> torch.Tensor:
        """Fallback method for encoding support set relations only"""
        try:
            # Compute pairwise distances within support set
            distances = torch.cdist(support_features, support_features, p=2)
            
            # Aggregate distance statistics as task context
            task_stats = torch.cat([
                distances.mean(dim=1).mean().unsqueeze(0),  # Average distance
                distances.std().unsqueeze(0),               # Distance spread
                distances.max().unsqueeze(0),               # Maximum distance
                distances.min().unsqueeze(0)                # Minimum distance
            ])
            
            # Expand to embedding dimension
            if task_stats.size(0) < self.embedding_dim:
                padding = torch.zeros(self.embedding_dim - task_stats.size(0), device=task_stats.device)
                task_stats = torch.cat([task_stats, padding])
            elif task_stats.size(0) > self.embedding_dim:
                task_stats = task_stats[:self.embedding_dim]
                
            return task_stats.unsqueeze(0)
            
        except Exception as e:
            warnings.warn(f"Intra-support relations encoding failed: {e}, falling back to mean")
            return self.task_encoder(support_features.mean(dim=0, keepdim=True))
    
    def _compute_self_attention(self, base_prototypes: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """SOLUTION 1: Self-Attention on Prototypes (Vaswani et al. 2017)"""
        try:
            # Self-attention to learn inter-prototype relationships
            if not hasattr(self, '_self_attention'):
                self._self_attention = nn.MultiheadAttention(
                    self.embedding_dim, num_heads=self.config.attention_heads, 
                    batch_first=True
                ).to(base_prototypes.device)
            
            # Apply self-attention 
            attended_prototypes, attention_weights = self._self_attention(
                base_prototypes.unsqueeze(0),  # [1, n_classes, embedding_dim]
                base_prototypes.unsqueeze(0),  # [1, n_classes, embedding_dim]  
                base_prototypes.unsqueeze(0)   # [1, n_classes, embedding_dim]
            )
            
            attended_prototypes = attended_prototypes.squeeze(0)  # [n_classes, embedding_dim]
            attention_weights = attention_weights.squeeze(0)      # [n_classes, n_classes]
            
            return attended_prototypes, attention_weights
            
        except Exception as e:
            warnings.warn(f"Self-attention failed: {e}, falling back to identity")
            identity_weights = torch.eye(base_prototypes.size(0), device=base_prototypes.device)
            return base_prototypes, identity_weights
    
    def _compute_task_conditioned_adaptation(self, base_prototypes: torch.Tensor, task_context: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """SOLUTION 2: Task-Conditioned Prototype Adaptation (Real meta-learning)"""
        try:
            # Generate adaptation parameters from task context
            if not hasattr(self, '_meta_parameter_generator'):
                self._meta_parameter_generator = nn.Linear(
                    self.embedding_dim, self.embedding_dim * 2
                ).to(task_context.device)
            
            if not hasattr(self, '_adaptation_transform'):
                self._adaptation_transform = nn.Linear(
                    self.embedding_dim * 2, self.embedding_dim
                ).to(base_prototypes.device)
            
            adaptation_params = self._meta_parameter_generator(task_context.squeeze(0))  # [param_dim]
            
            # Apply task-specific transformation to each prototype
            adapted_prototypes = []
            attention_weights = []
            
            for i, prototype in enumerate(base_prototypes):
                # Task-conditioned linear transformation
                combined_input = torch.cat([prototype, adaptation_params], dim=0)
                adapted_prototype = prototype + self._adaptation_transform(combined_input)
                adapted_prototypes.append(adapted_prototype)
                
                # Compute attention weight based on adaptation magnitude
                adaptation_magnitude = torch.norm(adapted_prototype - prototype).item()
                attention_weights.append(adaptation_magnitude)
            
            attended_prototypes = torch.stack(adapted_prototypes)
            
            # Normalize attention weights
            attention_weights = torch.tensor(attention_weights, device=base_prototypes.device)
            attention_weights = F.softmax(attention_weights, dim=0)
            
            # Create attention weight matrix (diagonal for simplicity)
            attention_weight_matrix = torch.diag(attention_weights)
            
            return attended_prototypes, attention_weight_matrix
            
        except Exception as e:
            warnings.warn(f"Task-conditioned adaptation failed: {e}, falling back to identity")
            identity_weights = torch.eye(base_prototypes.size(0), device=base_prototypes.device)
            return base_prototypes, identity_weights
    
    def _compute_cross_attention_adaptation(self, base_prototypes: torch.Tensor, support_features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """SOLUTION 3: Cross-Attention with Support Set (Relation Networks style)"""
        try:
            # Initialize cross-attention components if needed
            if not hasattr(self, '_support_key_projection'):
                self._support_key_projection = nn.Linear(self.embedding_dim, self.embedding_dim).to(support_features.device)
                self._support_value_projection = nn.Linear(self.embedding_dim, self.embedding_dim).to(support_features.device)
                self._prototype_query_projection = nn.Linear(self.embedding_dim, self.embedding_dim).to(base_prototypes.device)
            
            # Use support features as keys and values
            support_keys = self._support_key_projection(support_features)     # [n_support, d_model]
            support_values = self._support_value_projection(support_features) # [n_support, d_model]
            prototype_queries = self._prototype_query_projection(base_prototypes) # [n_classes, d_model]
            
            # Cross-attention: prototypes attend to support examples
            attention_scores = torch.matmul(prototype_queries, support_keys.transpose(-2, -1))
            attention_scores = attention_scores / math.sqrt(self.embedding_dim)
            attention_weights = F.softmax(attention_scores, dim=-1)  # [n_classes, n_support]
            
            # Aggregate support information for each prototype
            attended_prototypes = torch.matmul(attention_weights, support_values)
            
            # Convert attention weights to square matrix for consistency
            # Use prototype-to-prototype similarities based on support attention
            proto_similarities = torch.matmul(attention_weights, attention_weights.transpose(-2, -1))
            
            return attended_prototypes, proto_similarities
            
        except Exception as e:
            warnings.warn(f"Cross-attention adaptation failed: {e}, falling back to identity")
            identity_weights = torch.eye(base_prototypes.size(0), device=base_prototypes.device)
            return base_prototypes, identity_weights
    
    def _compute_learnable_mixing(self, base_prototypes: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """SOLUTION 4: Learnable Prototype Mixing (Set Transformer style)"""
        try:
            n_classes = base_prototypes.size(0)
            
            # Create learnable queries with proper initialization
            if not hasattr(self, '_learnable_queries'):
                bound = (6.0 / self.embedding_dim) ** 0.5
                self._learnable_queries = nn.Parameter(
                    torch.empty(n_classes, self.config.attention_heads, self.embedding_dim).uniform_(-bound, bound)
                ).to(base_prototypes.device)
            
            if not hasattr(self, '_set_attention'):
                self._set_attention = nn.MultiheadAttention(
                    self.embedding_dim, num_heads=self.config.attention_heads,
                    batch_first=True
                ).to(base_prototypes.device)
            
            # Expand learnable queries to match current batch
            if self._learnable_queries.size(0) != n_classes:
                # Reinitialize with proper initialization if class count changed
                bound = (6.0 / self.embedding_dim) ** 0.5
                self._learnable_queries = nn.Parameter(
                    torch.empty(n_classes, self.config.attention_heads, self.embedding_dim).uniform_(-bound, bound)
                ).to(base_prototypes.device)
            
            mixing_queries = self._learnable_queries  # [n_classes, n_heads, d_model]
            
            # Multi-head attention between mixing queries and prototypes
            # Use averaged queries for simplicity
            query_input = mixing_queries.mean(dim=1).unsqueeze(0)  # [1, n_classes, d_model]
            key_input = base_prototypes.unsqueeze(0)               # [1, n_classes, d_model]
            value_input = base_prototypes.unsqueeze(0)             # [1, n_classes, d_model]
            
            attended_prototypes, attention_weights = self._set_attention(
                query=query_input,
                key=key_input,
                value=value_input
            )
            
            attended_prototypes = attended_prototypes.squeeze(0)  # [n_classes, d_model]
            attention_weights = attention_weights.squeeze(0)      # [n_classes, n_classes]
            
            return attended_prototypes, attention_weights
            
        except Exception as e:
            warnings.warn(f"Learnable mixing failed: {e}, falling back to identity")
            identity_weights = torch.eye(base_prototypes.size(0), device=base_prototypes.device)
            return base_prototypes, identity_weights