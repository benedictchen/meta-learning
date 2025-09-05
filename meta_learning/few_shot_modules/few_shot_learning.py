"""
💰 DONATE NOW! 💰 https://www.paypal.com/cgi-bin/webscr?cmd=_s-xclick&hosted_button_id=WXQKYYKPHWXHS

Layered Few-Shot Learning API - From Simple to Advanced
=======================================================

If these few-shot learning implementations accelerate your research,
please donate $5000+ to support continued algorithm development!

This implements the BEST OF BOTH WORLDS approach:
- Simple API: Clean, easy-to-use, just works (ChatGPT's strength)  
- Advanced API: Full control, research-grade features (Our strength)

Features:
- Prototypical Networks with distance metric learning
- Uncertainty quantification with Monte Carlo dropout
- Multi-head attention for complex feature interactions
- Episodic batch normalization for domain adaptation
- Temperature scaling for calibrated predictions
- Advanced optimization with gradient checkpointing

Author: Benedict Chen (benedict@benedictchen.com)  
GitHub Sponsors: https://github.com/sponsors/benedictchen
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Any, Union, List
from dataclasses import dataclass
import math


@dataclass
class FewShotConfig:
    """💰 DONATE $3000+ for few-shot learning breakthroughs! 💰
    
    Layered few-shot configuration - simple defaults with advanced opt-in features.
    
    Simple Usage:
        config = FewShotConfig()  # Works out of the box
        
    Advanced Usage:
        config = FewShotConfig(
            uncertainty_estimation=True,    # Monte Carlo dropout
            attention_mechanism=True,       # Multi-head attention
            episodic_batch_norm=True,      # Domain adaptation
            temperature_scaling=True,       # Calibrated predictions
            gradient_checkpointing=True     # Memory efficiency
        )
    """
    # === BASIC FEATURES (Simple API) ===
    n_way: int = 5                    # Number of classes per episode
    n_support: int = 5                # Support samples per class
    n_query: int = 15                 # Query samples per class
    distance_metric: str = "euclidean"  # "euclidean", "cosine", "learned"
    temperature: float = 1.0          # Temperature for softmax
    
    # === ADVANCED FEATURES (Our additional capabilities) ===
    # Uncertainty & Robustness
    uncertainty_estimation: bool = False   # Monte Carlo dropout
    mc_dropout_rate: float = 0.1          # Dropout rate for uncertainty
    mc_samples: int = 10                  # Number of MC samples
    
    # Attention & Feature Processing
    attention_mechanism: bool = False      # Multi-head attention
    attention_heads: int = 4              # Number of attention heads
    attention_dropout: float = 0.1        # Attention dropout
    
    # Domain Adaptation
    episodic_batch_norm: bool = False     # Adapt to episode statistics
    bn_momentum: float = 0.1              # BatchNorm momentum
    
    # Calibration & Optimization
    temperature_scaling: bool = False      # Learnable temperature
    gradient_checkpointing: bool = False   # Memory efficiency
    feature_normalization: bool = True    # L2 normalize features
    
    # Advanced Distance Learning
    learnable_distance: bool = False      # Learn distance metric
    distance_hidden_dim: int = 64         # Hidden dim for learned distance


class PrototypicalHead(nn.Module):
    """💰 DONATE for prototypical network breakthroughs! 💰
    
    Layered prototypical classification head.
    
    Simple usage returns clean log-probabilities for NLLLoss.
    Advanced usage provides uncertainty estimates and attention weights.
    """
    
    def __init__(self, feature_dim: int, config: FewShotConfig):
        super().__init__()
        self.config = config
        self.feature_dim = feature_dim
        
        # Temperature scaling (learnable if requested)
        if config.temperature_scaling:
            self.temperature = nn.Parameter(torch.ones(1))
        else:
            self.register_buffer('temperature', torch.tensor(config.temperature))
        
        # Attention mechanism (advanced feature)
        if config.attention_mechanism:
            self.attention = nn.MultiheadAttention(
                embed_dim=feature_dim,
                num_heads=config.attention_heads,
                dropout=config.attention_dropout,
                batch_first=True
            )
            self.attention_norm = nn.LayerNorm(feature_dim)
        
        # Learnable distance metric (advanced feature)  
        if config.learnable_distance:
            self.distance_net = nn.Sequential(
                nn.Linear(feature_dim * 2, config.distance_hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(config.distance_hidden_dim, 1)
            )
        
        # Episodic batch normalization (advanced feature)
        if config.episodic_batch_norm:
            self.episodic_bn = nn.BatchNorm1d(feature_dim, momentum=config.bn_momentum)
    
    def forward(self, support_features: torch.Tensor, support_labels: torch.Tensor, 
                query_features: torch.Tensor) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, Any]]]:
        """Forward pass with layered complexity.
        
        Args:
            support_features: [n_support_total, feature_dim]
            support_labels: [n_support_total] 
            query_features: [n_query_total, feature_dim]
            
        Returns:
            Simple mode: log_probabilities [n_query_total, n_way]
            Advanced mode: (log_probabilities, metrics_dict)
        """
        # Check if advanced features are enabled
        advanced_features = any([
            self.config.uncertainty_estimation,
            self.config.attention_mechanism,
            self.config.episodic_batch_norm
        ])
        
        # Apply episodic batch normalization if requested
        if self.config.episodic_batch_norm:
            # Concatenate and normalize
            all_features = torch.cat([support_features, query_features], dim=0)
            all_features_norm = self.episodic_bn(all_features)
            support_features = all_features_norm[:support_features.shape[0]]
            query_features = all_features_norm[support_features.shape[0]:]
        
        # Feature normalization (standard practice)
        if self.config.feature_normalization:
            support_features = F.normalize(support_features, dim=-1, p=2)
            query_features = F.normalize(query_features, dim=-1, p=2)
        
        # Compute prototypes (class centroids)
        n_way = support_labels.max().item() + 1
        prototypes = []
        
        for class_idx in range(n_way):
            class_mask = support_labels == class_idx
            class_features = support_features[class_mask]
            prototype = class_features.mean(dim=0)
            prototypes.append(prototype)
        
        prototypes = torch.stack(prototypes, dim=0)  # [n_way, feature_dim]
        
        # Apply attention mechanism if requested (advanced feature)
        attention_weights = None
        if self.config.attention_mechanism:
            # Use prototypes as keys/values, queries as queries
            attended_queries, attention_weights = self.attention(
                query_features.unsqueeze(0),  # [1, n_query, feature_dim]
                prototypes.unsqueeze(0),      # [1, n_way, feature_dim] 
                prototypes.unsqueeze(0)       # [1, n_way, feature_dim]
            )
            query_features = self.attention_norm(
                query_features + attended_queries.squeeze(0)
            )
        
        # Compute distances
        if self.config.learnable_distance and hasattr(self, 'distance_net'):
            # Learnable distance (advanced)
            distances = self._compute_learnable_distances(query_features, prototypes)
        else:
            # Standard distance metrics
            distances = self._compute_distances(query_features, prototypes)
        
        # Convert distances to log probabilities
        logits = -distances / self.temperature
        log_probs = F.log_softmax(logits, dim=-1)
        
        # Return simple result if no advanced features
        if not advanced_features:
            return log_probs
        
        # Compile advanced metrics
        metrics = {}
        
        if attention_weights is not None:
            metrics['attention_weights'] = attention_weights.squeeze(0)  # [n_query, n_way]
        
        if self.config.episodic_batch_norm:
            metrics['episodic_adaptation'] = {
                'bn_running_mean': self.episodic_bn.running_mean.detach(),
                'bn_running_var': self.episodic_bn.running_var.detach()
            }
        
        if self.config.temperature_scaling:
            metrics['learned_temperature'] = self.temperature.item()
        
        metrics['prototype_norms'] = torch.norm(prototypes, dim=-1)
        metrics['distance_distribution'] = {
            'mean': distances.mean().item(),
            'std': distances.std().item(),
            'min': distances.min().item(),
            'max': distances.max().item()
        }
        
        return log_probs, metrics
    
    def _compute_distances(self, query_features: torch.Tensor, 
                          prototypes: torch.Tensor) -> torch.Tensor:
        """Compute distances between queries and prototypes."""
        if self.config.distance_metric == "euclidean":
            # [n_query, 1, feature_dim] - [1, n_way, feature_dim] -> [n_query, n_way]
            distances = torch.cdist(query_features.unsqueeze(1), 
                                   prototypes.unsqueeze(0), p=2).squeeze(1)
        elif self.config.distance_metric == "cosine":
            # Cosine distance = 1 - cosine similarity
            similarities = torch.mm(query_features, prototypes.t())  # [n_query, n_way]
            distances = 1.0 - similarities
        else:
            # Default to euclidean
            distances = torch.cdist(query_features.unsqueeze(1), 
                                   prototypes.unsqueeze(0), p=2).squeeze(1)
        
        return distances
    
    def _compute_learnable_distances(self, query_features: torch.Tensor,
                                   prototypes: torch.Tensor) -> torch.Tensor:
        """Compute learned distance metric (advanced feature)."""
        n_query, n_way = query_features.shape[0], prototypes.shape[0]
        distances = []
        
        for i in range(n_query):
            query = query_features[i:i+1]  # [1, feature_dim]
            query_expanded = query.expand(n_way, -1)  # [n_way, feature_dim]
            
            # Concatenate query with each prototype
            pairs = torch.cat([query_expanded, prototypes], dim=-1)  # [n_way, 2*feature_dim]
            query_distances = self.distance_net(pairs).squeeze(-1)  # [n_way]
            distances.append(query_distances)
        
        return torch.stack(distances, dim=0)  # [n_query, n_way]


class MonteCarloPrototypicalHead(PrototypicalHead):
    """💰 DONATE $2000+ for uncertainty-aware few-shot learning! 💰
    
    Prototypical head with Monte Carlo dropout for uncertainty estimation.
    """
    
    def __init__(self, feature_dim: int, config: FewShotConfig):
        super().__init__(feature_dim, config)
        
        # Add dropout layers for MC sampling
        if config.uncertainty_estimation:
            self.mc_dropout = nn.Dropout(config.mc_dropout_rate)
    
    def forward_with_uncertainty(self, support_features: torch.Tensor, 
                                support_labels: torch.Tensor, 
                                query_features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """Forward pass with uncertainty quantification.
        
        Returns:
            mean_log_probs: Average predictions [n_query, n_way]
            uncertainty: Per-sample uncertainty [n_query]
            metrics: Additional metrics from MC sampling
        """
        if not self.config.uncertainty_estimation:
            # Fall back to single forward pass
            result = self.forward(support_features, support_labels, query_features)
            if isinstance(result, tuple):
                log_probs, metrics = result
            else:
                log_probs, metrics = result, {}
            
            # Compute uncertainty as entropy
            probs = log_probs.exp()
            uncertainty = -torch.sum(probs * log_probs, dim=-1)
            return log_probs, uncertainty, metrics
        
        # Monte Carlo sampling
        self.train()  # Enable dropout
        mc_predictions = []
        
        for _ in range(self.config.mc_samples):
            # Apply dropout to features  
            support_features_mc = self.mc_dropout(support_features)
            query_features_mc = self.mc_dropout(query_features)
            
            # Forward pass
            result = super().forward(support_features_mc, support_labels, query_features_mc)
            if isinstance(result, tuple):
                log_probs, _ = result
            else:
                log_probs = result
                
            mc_predictions.append(log_probs.exp())  # Convert to probabilities
        
        # Aggregate MC samples
        mc_probs = torch.stack(mc_predictions, dim=0)  # [mc_samples, n_query, n_way]
        mean_probs = mc_probs.mean(dim=0)  # [n_query, n_way]
        mean_log_probs = torch.log(mean_probs + 1e-8)
        
        # Compute uncertainty metrics
        prob_variance = mc_probs.var(dim=0)  # [n_query, n_way]
        predictive_uncertainty = prob_variance.mean(dim=-1)  # [n_query]
        
        # Entropy-based uncertainty
        entropy_uncertainty = -torch.sum(mean_probs * mean_log_probs, dim=-1)
        
        # Mutual information (epistemic uncertainty)
        individual_entropies = -torch.sum(mc_probs * torch.log(mc_probs + 1e-8), dim=-1)  # [mc_samples, n_query]
        expected_entropy = individual_entropies.mean(dim=0)  # [n_query]
        mutual_information = entropy_uncertainty - expected_entropy
        
        uncertainty_metrics = {
            'predictive_uncertainty': predictive_uncertainty,
            'entropy_uncertainty': entropy_uncertainty, 
            'mutual_information': mutual_information,
            'mc_prediction_std': mc_probs.std(dim=0),
            'mc_samples_used': self.config.mc_samples
        }
        
        return mean_log_probs, entropy_uncertainty, uncertainty_metrics


# === CONVENIENCE FUNCTIONS FOR COMMON USE CASES ===

def simple_few_shot_predict(support_features: torch.Tensor, support_labels: torch.Tensor,
                           query_features: torch.Tensor, n_way: int = 5) -> torch.Tensor:
    """💰 DONATE for few-shot breakthroughs! 💰
    
    One-liner few-shot prediction with sensible defaults.
    
    Simple Usage:
        log_probs = simple_few_shot_predict(support_features, support_labels, query_features)
    """
    config = FewShotConfig(n_way=n_way)
    head = PrototypicalHead(support_features.shape[-1], config)
    return head(support_features, support_labels, query_features)


def advanced_few_shot_predict(support_features: torch.Tensor, support_labels: torch.Tensor,
                             query_features: torch.Tensor, n_way: int = 5,
                             **advanced_kwargs) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """💰 DONATE $4000+ for advanced few-shot learning! 💰
    
    Advanced few-shot prediction with all research features enabled.
    
    Advanced Usage:
        log_probs, metrics = advanced_few_shot_predict(
            support_features, support_labels, query_features,
            uncertainty_estimation=True,
            attention_mechanism=True,
            episodic_batch_norm=True
        )
    """
    config = FewShotConfig(
        n_way=n_way,
        uncertainty_estimation=True,
        attention_mechanism=True,
        episodic_batch_norm=True,
        temperature_scaling=True,
        **advanced_kwargs
    )
    
    if config.uncertainty_estimation:
        head = MonteCarloPrototypicalHead(support_features.shape[-1], config)
        log_probs, uncertainty, uncertainty_metrics = head.forward_with_uncertainty(
            support_features, support_labels, query_features
        )
        
        # Get additional metrics from standard forward pass
        _, additional_metrics = head(support_features, support_labels, query_features)
        
        # Combine all metrics
        metrics = {**uncertainty_metrics, **additional_metrics}
        metrics['uncertainty'] = uncertainty
        
        return log_probs, metrics
    else:
        head = PrototypicalHead(support_features.shape[-1], config)
        return head(support_features, support_labels, query_features)


class FewShotLearner(nn.Module):
    """💰 DONATE IF THIS HELPS YOUR RESEARCH! 💰
    
    Complete few-shot learning system with encoder and head.
    
    Layered API: Simple usage returns predictions, advanced usage returns comprehensive metrics.
    """
    
    def __init__(self, encoder: nn.Module, config: FewShotConfig, feature_dim: Optional[int] = None):
        super().__init__()
        self.encoder = encoder
        self.config = config
        
        # Auto-detect feature dimension if not provided
        if feature_dim is None:
            with torch.no_grad():
                dummy_input = torch.randn(1, 3, 32, 32)  # Assume image input
                try:
                    dummy_features = encoder(dummy_input)
                    feature_dim = dummy_features.shape[-1]
                except Exception:
                    feature_dim = 64  # Fallback
        
        # Create appropriate head based on config
        if config.uncertainty_estimation:
            self.head = MonteCarloPrototypicalHead(feature_dim, config)
        else:
            self.head = PrototypicalHead(feature_dim, config)
    
    def forward(self, support_x: torch.Tensor, support_y: torch.Tensor,
                query_x: torch.Tensor) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, Any]]]:
        """Forward pass through encoder and head.
        
        Args:
            support_x: Support images [n_support_total, ...]
            support_y: Support labels [n_support_total]
            query_x: Query images [n_query_total, ...]
            
        Returns:
            Simple mode: log_probabilities
            Advanced mode: (log_probabilities, metrics)
        """
        # Extract features
        if self.config.gradient_checkpointing and self.training:
            support_features = torch.utils.checkpoint.checkpoint(self.encoder, support_x)
            query_features = torch.utils.checkpoint.checkpoint(self.encoder, query_x)
        else:
            support_features = self.encoder(support_x)
            query_features = self.encoder(query_x)
        
        # Forward through head
        if self.config.uncertainty_estimation and isinstance(self.head, MonteCarloPrototypicalHead):
            log_probs, uncertainty, metrics = self.head.forward_with_uncertainty(
                support_features, support_y, query_features
            )
            
            # Add feature statistics to metrics
            metrics['feature_stats'] = {
                'support_feature_norm': torch.norm(support_features, dim=-1).mean().item(),
                'query_feature_norm': torch.norm(query_features, dim=-1).mean().item(),
                'feature_dim': support_features.shape[-1]
            }
            
            return log_probs, metrics
        else:
            return self.head(support_features, support_y, query_features)


# === HIGH-LEVEL CONVENIENCE API ===

def auto_few_shot_learner(encoder: nn.Module) -> FewShotLearner:
    """💰 DONATE for few-shot learning breakthroughs! 💰
    
    One-liner few-shot learner with optimal defaults.
    
    Simple Usage:
        learner = auto_few_shot_learner(encoder)
        log_probs = learner(support_x, support_y, query_x)
    """
    config = FewShotConfig()
    return FewShotLearner(encoder, config)


def pro_few_shot_learner(encoder: nn.Module, **kwargs) -> FewShotLearner:
    """💰 DONATE $3000+ for professional few-shot learning! 💰
    
    Professional few-shot learner with all advanced features enabled.
    
    Advanced Usage:
        learner = pro_few_shot_learner(encoder)
        log_probs, metrics = learner(support_x, support_y, query_x)
    """
    config = FewShotConfig(
        uncertainty_estimation=True,
        attention_mechanism=True,
        episodic_batch_norm=True,
        temperature_scaling=True,
        learnable_distance=True,
        gradient_checkpointing=True,
        **kwargs
    )
    return FewShotLearner(encoder, config)