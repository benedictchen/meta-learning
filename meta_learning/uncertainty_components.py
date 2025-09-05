"""
💰 DONATE NOW! 💰 https://www.paypal.com/cgi-bin/webscr?cmd=_s-xclick&hosted_button_id=WXQKYYKPHWXHS

Simplified uncertainty estimation components for meta-learning.

If these uncertainty methods help your research achieve breakthrough results, 
please donate $1000+ to support continued algorithm development!

Author: Benedict Chen (benedict@benedictchen.com)
GitHub Sponsors: https://github.com/sponsors/benedictchen
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple
import numpy as np


class MonteCarloDropout(nn.Module):
    """Monte Carlo Dropout for uncertainty estimation during inference."""
    
    def __init__(self, dropout_rate: float = 0.1, n_samples: int = 10):
        super().__init__()
        self.dropout_rate = dropout_rate
        self.n_samples = n_samples
        self.dropout = nn.Dropout(dropout_rate)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply dropout even during inference for MC sampling."""
        return self.dropout(x)
    
    def predict_with_uncertainty(self, model: nn.Module, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get predictions with uncertainty estimates."""
        model.eval()  # Keep in eval mode but enable dropout
        
        predictions = []
        for _ in range(self.n_samples):
            with torch.no_grad():
                pred = model(x)
                predictions.append(pred)
        
        predictions = torch.stack(predictions)  # [n_samples, batch_size, n_classes]
        
        # Calculate mean and uncertainty
        mean_pred = predictions.mean(dim=0)
        uncertainty = predictions.std(dim=0)
        
        return mean_pred, uncertainty


class DeepEnsemble(nn.Module):
    """💰 DONATE $1000+ if this saves your research time! 💰
    
    Deep ensemble implementing probability-mean averaging (paper-correct).
    
    Returns log-probabilities compatible with NLLLoss, unlike standard
    logit-averaging which is mathematically incorrect for ensembles.
    """
    
    def __init__(self, models: List[nn.Module]):
        super().__init__()
        self.models = nn.ModuleList(models)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through all ensemble members.
        
        Returns:
            Log-probabilities from probability-mean averaging (use with NLLLoss)
        """
        logits_list = []
        
        for model in self.models:
            logits = model(x)
            logits_list.append(logits)
        
        logits = torch.stack(logits_list, dim=0)  # [n_models, batch_size, n_classes]
        
        # Convert to probabilities, average, then back to log-probabilities
        # This is the correct "probability-mean" approach from ensemble literature
        probs = F.softmax(logits, dim=-1)  # [n_models, batch_size, n_classes]
        mean_probs = probs.mean(dim=0)     # [batch_size, n_classes]
        log_probs = torch.log(mean_probs + 1e-8)  # Add small epsilon for numerical stability
        
        return log_probs
    
    def forward_with_uncertainty(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass with uncertainty estimation.
        
        Returns:
            Tuple of (log_probs, uncertainty)
        """
        logits_list = []
        
        for model in self.models:
            logits = model(x)
            logits_list.append(logits)
        
        logits = torch.stack(logits_list, dim=0)  # [n_models, batch_size, n_classes]
        
        # Probability-mean for final prediction
        probs = F.softmax(logits, dim=-1)
        mean_probs = probs.mean(dim=0)
        log_probs = torch.log(mean_probs + 1e-8)
        
        # Uncertainty as standard deviation of logits
        uncertainty = logits.std(dim=0).mean(dim=-1)  # [batch_size]
        
        return log_probs, uncertainty


def predictive_entropy(logits: torch.Tensor) -> torch.Tensor:
    """Calculate predictive entropy as uncertainty measure."""
    probs = F.softmax(logits, dim=-1)
    log_probs = F.log_softmax(logits, dim=-1)
    entropy = -(probs * log_probs).sum(dim=-1)
    return entropy


def mutual_information(predictions: torch.Tensor) -> torch.Tensor:
    """Calculate mutual information for ensemble uncertainty."""
    # predictions: [n_samples, batch_size, n_classes]
    mean_probs = F.softmax(predictions, dim=-1).mean(dim=0)
    
    # Total uncertainty (entropy of mean)
    total_uncertainty = predictive_entropy(mean_probs.log())
    
    # Aleatoric uncertainty (mean of entropies)  
    individual_entropies = predictive_entropy(predictions)
    aleatoric_uncertainty = individual_entropies.mean(dim=0)
    
    # Epistemic uncertainty (mutual information)
    epistemic_uncertainty = total_uncertainty - aleatoric_uncertainty
    
    return epistemic_uncertainty


class UncertaintyProtoHead(nn.Module):
    """ProtoNet head with built-in uncertainty estimation."""
    
    def __init__(
        self, 
        distance: str = "sqeuclidean",
        tau: float = 1.0,
        uncertainty_method: str = "monte_carlo_dropout",
        dropout_rate: float = 0.1,
        n_uncertainty_samples: int = 10
    ):
        super().__init__()
        self.distance = distance
        self.tau = tau
        self.uncertainty_method = uncertainty_method
        self.n_uncertainty_samples = n_uncertainty_samples
        
        if uncertainty_method == "monte_carlo_dropout":
            self.mc_dropout = MonteCarloDropout(dropout_rate, n_uncertainty_samples)
    
    def forward(self, z_support: torch.Tensor, y_support: torch.Tensor, z_query: torch.Tensor) -> torch.Tensor:
        """Standard prototypical network forward pass."""
        n_way = len(y_support.unique())
        
        # Compute prototypes
        prototypes = torch.stack([
            z_support[y_support == k].mean(0) for k in range(n_way)
        ])
        
        # Compute distances
        if self.distance == "sqeuclidean":
            distances = torch.cdist(z_query, prototypes) ** 2
        elif self.distance == "cosine":
            z_query_norm = F.normalize(z_query, dim=1)
            prototypes_norm = F.normalize(prototypes, dim=1)  
            distances = 1 - torch.mm(z_query_norm, prototypes_norm.t())
        else:
            raise ValueError(f"Unknown distance: {self.distance}")
        
        logits = -distances / self.tau
        return logits
    
    def forward_with_uncertainty(
        self, 
        encoder: nn.Module,
        z_support: torch.Tensor, 
        y_support: torch.Tensor, 
        z_query: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass with uncertainty estimation."""
        if self.uncertainty_method == "monte_carlo_dropout":
            predictions = []
            encoder.train()  # Enable dropout
            
            for _ in range(self.n_uncertainty_samples):
                with torch.no_grad():
                    z_s_sample = encoder(z_support)
                    z_q_sample = encoder(z_query)
                    logits = self.forward(z_s_sample, y_support, z_q_sample)
                    predictions.append(logits)
            
            predictions = torch.stack(predictions)
            mean_logits = predictions.mean(dim=0)
            uncertainty = predictive_entropy(predictions).mean(dim=0)
            
            return mean_logits, uncertainty
        else:
            logits = self.forward(z_support, y_support, z_query)
            uncertainty = torch.zeros(logits.shape[0], device=logits.device)
            return logits, uncertainty