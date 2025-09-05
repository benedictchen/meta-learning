"""
💰 SUPPORT THIS RESEARCH - PLEASE DONATE! 💰

🙏 If this library helps your research or project, please consider donating:
💳 PayPal: https://www.paypal.com/cgi-bin/webscr?cmd=_s-xclick&hosted_button_id=WXQKYYKPHWXHS
⭐ GitHub Sponsors: https://github.com/sponsors/benedictchen

👨‍💻 Author: Benedict Chen
💰 Donations: Help support this work! Buy me a coffee ☕, beer 🍺, lamborghini 🏎️, or private island 🏝️
💖 Please consider recurring donations to fully support continued research

Your support enables cutting-edge AI research for everyone! 🚀

Uncertainty-Aware Distance Components for Few-Shot Learning 🎯🔬
==============================================================

🎯 **ELI5 Explanation**:
Imagine you're a detective trying to solve a case with limited clues!
Sometimes you're very confident in your conclusion (like having video evidence),
and sometimes you're uncertain (like having conflicting witness statements).
This module helps AI be honest about its confidence level:

- 🎯 **High Confidence**: "I'm 95% sure this is a cat" 
- 🤔 **Medium Confidence**: "I'm 70% sure this is a dog"
- 😟 **Low Confidence**: "I'm 30% sure... could be anything"

🔬 **Research Foundation**:
- **Monte Carlo Dropout**: Yarin Gal & Zoubin Ghahramani (2016) - "Dropout as a Bayesian Approximation"
- **Deep Ensembles**: Balaji Lakshminarayanan et al. (2017) - "Simple and Scalable Predictive Uncertainty"
- **Evidential Learning**: Murat Sensoy et al. (2018) - "Evidential Deep Learning to Quantify Classification Uncertainty"
- **Bayesian Neural Networks**: Charles Blundell et al. (2015) - "Weight Uncertainty in Neural Networks"

🧮 **Mathematical Framework**:
- **Epistemic Uncertainty**: σₑ² = Var[E[y|x,θ]]  [Model uncertainty - what we don't know]
- **Aleatoric Uncertainty**: σₐ² = E[Var[y|x,θ]]  [Data uncertainty - inherent noise]
- **Total Uncertainty**: σ² = σₑ² + σₐ²  [Combined uncertainty measure]
- **Evidence**: α = exp(logits)  [Evidential learning concentration parameters]

This module provides uncertainty estimation for few-shot learning
with numerically stable implementations using our enhanced math operations.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Union
import numpy as np
from dataclasses import dataclass
import math
from ..core.math_utils import pairwise_sqeuclidean, cosine_logits


@dataclass
class UncertaintyConfig:
    """Comprehensive configuration for uncertainty estimation methods."""
    # Core uncertainty method selection
    method: str = "monte_carlo_dropout"  # monte_carlo_dropout, deep_ensemble, evidential, bayesian
    
    # Monte Carlo Dropout parameters
    dropout_rate: float = 0.1
    n_samples: int = 10
    
    # Deep Ensemble parameters  
    ensemble_size: int = 5
    ensemble_hidden_dim: int = 32
    
    # Evidential learning parameters
    evidential_lambda: float = 1.0
    evidential_hidden_dim: int = 32
    num_classes: int = 10
    evidence_regularizer: float = 0.1
    uncertainty_method: str = "sensoy2018"  # "sensoy2018", "amini2020"
    
    # Bayesian parameters
    prior_mu: float = 0.0
    prior_sigma: float = 1.0
    bayesian_samples: int = 20
    
    # Distance computation
    distance_metric: str = "sqeuclidean"  # "sqeuclidean", "cosine"
    temperature: float = 1.0
    
    # Numerical stability
    eps: float = 1e-8
    max_uncertainty: float = 10.0


class MonteCarloDropout(nn.Module):
    """
    Monte Carlo Dropout for uncertainty estimation.
    
    Based on Gal & Ghahramani (2016): "Dropout as a Bayesian Approximation"
    
    Provides both epistemic and aleatoric uncertainty estimates by
    performing multiple forward passes with dropout enabled during inference.
    """
    
    def __init__(self, config: UncertaintyConfig):
        super().__init__()
        self.config = config
        self.dropout = nn.Dropout(config.dropout_rate)
    
    def forward(self, x: torch.Tensor, prototypes: torch.Tensor, 
                n_samples: Optional[int] = None) -> Dict[str, torch.Tensor]:
        """
        Compute uncertainty-aware distances using Monte Carlo Dropout.
        
        Args:
            x: Query features [N, D]
            prototypes: Class prototypes [K, D] 
            n_samples: Number of Monte Carlo samples
            
        Returns:
            Dictionary containing distances, uncertainties, and statistics
        """
        if n_samples is None:
            n_samples = self.config.n_samples
            
        # Store original training mode
        original_training = self.training
        
        # Enable training mode for dropout during inference
        self.train()
        
        distance_samples = []
        
        # Perform multiple stochastic forward passes
        for _ in range(n_samples):
            # Apply dropout to query features
            x_dropped = self.dropout(x)
            proto_dropped = self.dropout(prototypes)
            
            # Compute distances with enhanced numerical stability
            if self.config.distance_metric == "sqeuclidean":
                distances = pairwise_sqeuclidean(x_dropped, proto_dropped)
                # Convert to negative log-probabilities with temperature
                logits = -distances / self.config.temperature
            else:  # cosine
                logits = cosine_logits(x_dropped, proto_dropped, tau=self.config.temperature)
            
            distance_samples.append(logits)
        
        # Restore original training mode
        self.train(original_training)
        
        # Stack samples: [n_samples, N, K]
        distance_samples = torch.stack(distance_samples, dim=0)
        
        # Compute statistics
        mean_logits = distance_samples.mean(dim=0)  # [N, K]
        logit_variance = distance_samples.var(dim=0)  # [N, K] - Epistemic uncertainty
        
        # Convert to probabilities for cleaner uncertainty interpretation
        prob_samples = torch.softmax(distance_samples, dim=-1)  # [n_samples, N, K]
        mean_probs = prob_samples.mean(dim=0)  # [N, K]
        prob_variance = prob_samples.var(dim=0)  # [N, K]
        
        # Total uncertainty (entropy of mean predictions)
        eps = torch.finfo(mean_probs.dtype).eps
        mean_probs_safe = torch.clamp(mean_probs, min=eps, max=1.0 - eps)
        total_uncertainty = -(mean_probs_safe * torch.log(mean_probs_safe)).sum(dim=-1)  # [N]
        
        # Epistemic uncertainty (mean of entropies - entropy of mean)
        sample_entropies = -(prob_samples * torch.log(prob_samples + eps)).sum(dim=-1)  # [n_samples, N]
        epistemic_uncertainty = total_uncertainty - sample_entropies.mean(dim=0)  # [N]
        
        # Aleatoric uncertainty (mean of entropies)
        aleatoric_uncertainty = sample_entropies.mean(dim=0)  # [N]
        
        return {
            "logits": mean_logits,
            "probabilities": mean_probs,
            "total_uncertainty": total_uncertainty,
            "epistemic_uncertainty": epistemic_uncertainty,
            "aleatoric_uncertainty": aleatoric_uncertainty,
            "logit_variance": logit_variance,
            "probability_variance": prob_variance,
            "n_samples": n_samples
        }


class DeepEnsemble(nn.Module):
    """
    Deep Ensemble for uncertainty estimation.
    
    Based on Lakshminarayanan et al. (2017): "Simple and Scalable Predictive Uncertainty"
    
    Uses multiple independently trained networks to estimate uncertainty.
    """
    
    def __init__(self, feature_dim: int, config: UncertaintyConfig):
        super().__init__()
        self.config = config
        self.feature_dim = feature_dim
        
        # Create ensemble of networks
        self.ensemble = nn.ModuleList([
            nn.Sequential(
                nn.Linear(feature_dim, config.ensemble_hidden_dim),
                nn.ReLU(),
                nn.Linear(config.ensemble_hidden_dim, feature_dim)
            ) for _ in range(config.ensemble_size)
        ])
    
    def forward(self, x: torch.Tensor, prototypes: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute uncertainty using deep ensemble predictions.
        
        Args:
            x: Query features [N, D]
            prototypes: Class prototypes [K, D]
            
        Returns:
            Dictionary containing ensemble predictions and uncertainties
        """
        ensemble_logits = []
        
        # Get predictions from each ensemble member
        for network in self.ensemble:
            # Transform features through ensemble member
            x_transformed = network(x)
            proto_transformed = network(prototypes)
            
            # Compute distances with stable math
            if self.config.distance_metric == "sqeuclidean":
                distances = pairwise_sqeuclidean(x_transformed, proto_transformed)
                logits = -distances / self.config.temperature
            else:  # cosine
                logits = cosine_logits(x_transformed, proto_transformed, tau=self.config.temperature)
            
            ensemble_logits.append(logits)
        
        # Stack ensemble predictions: [ensemble_size, N, K]
        ensemble_logits = torch.stack(ensemble_logits, dim=0)
        
        # Compute ensemble statistics
        mean_logits = ensemble_logits.mean(dim=0)  # [N, K]
        logit_variance = ensemble_logits.var(dim=0)  # [N, K]
        
        # Convert to probabilities
        ensemble_probs = torch.softmax(ensemble_logits, dim=-1)  # [ensemble_size, N, K]
        mean_probs = ensemble_probs.mean(dim=0)  # [N, K]
        
        # Predictive entropy (total uncertainty)
        eps = torch.finfo(mean_probs.dtype).eps
        mean_probs_safe = torch.clamp(mean_probs, min=eps, max=1.0 - eps)
        predictive_entropy = -(mean_probs_safe * torch.log(mean_probs_safe)).sum(dim=-1)  # [N]
        
        # Expected entropy (aleatoric uncertainty)  
        individual_entropies = -(ensemble_probs * torch.log(ensemble_probs + eps)).sum(dim=-1)  # [ensemble_size, N]
        expected_entropy = individual_entropies.mean(dim=0)  # [N]
        
        # Mutual information (epistemic uncertainty)
        mutual_information = predictive_entropy - expected_entropy  # [N]
        
        return {
            "logits": mean_logits,
            "probabilities": mean_probs,
            "total_uncertainty": predictive_entropy,
            "epistemic_uncertainty": mutual_information,
            "aleatoric_uncertainty": expected_entropy,
            "logit_variance": logit_variance,
            "ensemble_size": self.config.ensemble_size
        }


class EvidentialLearning(nn.Module):
    """
    Evidential Learning for uncertainty quantification.
    
    Based on Sensoy et al. (2018): "Evidential Deep Learning to Quantify Classification Uncertainty"
    
    Models uncertainty using Dirichlet distributions and evidence theory.
    """
    
    def __init__(self, feature_dim: int, config: UncertaintyConfig):
        super().__init__()
        self.config = config
        
        # Evidence network to predict Dirichlet concentration parameters
        self.evidence_network = nn.Sequential(
            nn.Linear(feature_dim, config.evidential_hidden_dim),
            nn.ReLU(),
            nn.Linear(config.evidential_hidden_dim, config.num_classes),
            nn.Softplus()  # Ensure positive evidence
        )
    
    def forward(self, x: torch.Tensor, prototypes: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute evidential uncertainty estimates.
        
        Args:
            x: Query features [N, D]  
            prototypes: Class prototypes [K, D]
            
        Returns:
            Dictionary containing evidential predictions and uncertainties
        """
        # Compute base distances
        if self.config.distance_metric == "sqeuclidean":
            distances = pairwise_sqeuclidean(x, prototypes)
        else:  # cosine  
            # Convert cosine similarity to distance-like measure
            similarities = cosine_logits(x, prototypes, tau=1.0) 
            distances = -similarities  # Higher similarity = lower distance
        
        # Compute evidence for each class
        evidence = torch.exp(-distances / self.config.temperature) + self.config.eps  # [N, K]
        
        # Dirichlet concentration parameters: α = evidence + 1
        alpha = evidence + 1.0  # [N, K]
        
        # Dirichlet strength (total evidence)
        S = alpha.sum(dim=-1, keepdim=True)  # [N, 1]
        
        # Expected probabilities under Dirichlet distribution
        probabilities = alpha / S  # [N, K]
        
        # Uncertainty measures
        # Total uncertainty (entropy of expected probabilities)
        eps = self.config.eps
        prob_safe = torch.clamp(probabilities, min=eps, max=1.0 - eps)
        total_uncertainty = -(prob_safe * torch.log(prob_safe)).sum(dim=-1)  # [N]
        
        # Epistemic uncertainty (uncertainty of the Dirichlet itself)
        # Using expected entropy of Dirichlet distribution
        digamma_S = torch.digamma(S.squeeze(-1))  # [N]
        digamma_alpha = torch.digamma(alpha)  # [N, K]
        
        # Expected entropy under Dirichlet
        expected_entropy = -(probabilities * (digamma_alpha - digamma_S.unsqueeze(-1))).sum(dim=-1)  # [N]
        
        # Mutual information (epistemic uncertainty)
        epistemic_uncertainty = total_uncertainty - expected_entropy  # [N]
        
        # Data/aleatoric uncertainty (expected entropy)
        aleatoric_uncertainty = expected_entropy  # [N]
        
        # Confidence measures
        uncertainty_measure = self.config.num_classes / S.squeeze(-1)  # [N] - Higher = more uncertain
        
        # Convert distances to logits for compatibility
        logits = torch.log(probabilities + eps)  # [N, K]
        
        return {
            "logits": logits,
            "probabilities": probabilities,
            "evidence": evidence,
            "alpha": alpha,
            "dirichlet_strength": S.squeeze(-1),
            "total_uncertainty": total_uncertainty,
            "epistemic_uncertainty": epistemic_uncertainty,
            "aleatoric_uncertainty": aleatoric_uncertainty,
            "uncertainty_measure": uncertainty_measure
        }


class UncertaintyAwareDistance(nn.Module):
    """
    Unified uncertainty-aware distance computation for few-shot learning.
    
    Integrates multiple uncertainty estimation methods with numerically
    stable distance computations.
    """
    
    def __init__(self, feature_dim: int, config: UncertaintyConfig):
        super().__init__()
        self.config = config
        self.feature_dim = feature_dim
        
        # Initialize uncertainty method
        if config.method == "monte_carlo_dropout":
            self.uncertainty_module = MonteCarloDropout(config)
        elif config.method == "deep_ensemble":
            self.uncertainty_module = DeepEnsemble(feature_dim, config)
        elif config.method == "evidential":
            self.uncertainty_module = EvidentialLearning(feature_dim, config)
        else:
            raise ValueError(f"Unknown uncertainty method: {config.method}")
    
    def forward(self, query_features: torch.Tensor, 
                support_features: torch.Tensor, 
                support_labels: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute uncertainty-aware distances between queries and support prototypes.
        
        Args:
            query_features: Query embeddings [N_query, D]
            support_features: Support embeddings [N_support, D] 
            support_labels: Support labels [N_support]
            
        Returns:
            Dictionary with predictions, uncertainties, and diagnostics
        """
        # Compute class prototypes with numerical stability
        unique_labels = torch.unique(support_labels)
        n_classes = len(unique_labels)
        
        # Create label mapping for consistency
        label_map = {label.item(): i for i, label in enumerate(unique_labels)}
        mapped_labels = torch.tensor([label_map[label.item()] for label in support_labels], 
                                   device=support_labels.device)
        
        # Compute prototypes (centroids)
        prototypes = torch.zeros(n_classes, self.feature_dim, device=support_features.device)
        for i in range(n_classes):
            class_mask = mapped_labels == i
            if class_mask.sum() > 0:
                prototypes[i] = support_features[class_mask].mean(dim=0)
        
        # Apply uncertainty estimation method
        uncertainty_results = self.uncertainty_module(query_features, prototypes)
        
        # Add class information
        uncertainty_results.update({
            "prototypes": prototypes,
            "unique_labels": unique_labels,
            "n_classes": n_classes,
            "method": self.config.method
        })
        
        # Clip extreme uncertainties for numerical stability
        for key in ["total_uncertainty", "epistemic_uncertainty", "aleatoric_uncertainty"]:
            if key in uncertainty_results:
                uncertainty_results[key] = torch.clamp(
                    uncertainty_results[key], 
                    max=self.config.max_uncertainty
                )
        
        return uncertainty_results
    
    def get_predictions(self, uncertainty_results: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Extract class predictions from uncertainty results."""
        logits = uncertainty_results["logits"]
        return logits.argmax(dim=-1)
    
    def get_confidence_scores(self, uncertainty_results: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Extract confidence scores (1 - normalized uncertainty)."""
        total_uncertainty = uncertainty_results["total_uncertainty"]
        # Normalize to [0, 1] and invert for confidence
        max_uncertainty = self.config.max_uncertainty
        normalized_uncertainty = torch.clamp(total_uncertainty / max_uncertainty, 0, 1)
        confidence = 1.0 - normalized_uncertainty
        return confidence


def create_uncertainty_aware_distance(feature_dim: int, 
                                     method: str = "monte_carlo_dropout",
                                     **kwargs) -> UncertaintyAwareDistance:
    """
    Factory function for creating uncertainty-aware distance computations.
    
    Args:
        feature_dim: Dimensionality of feature vectors
        method: Uncertainty estimation method
        **kwargs: Additional configuration parameters
        
    Returns:
        Configured UncertaintyAwareDistance module
    """
    config = UncertaintyConfig(method=method, **kwargs)
    return UncertaintyAwareDistance(feature_dim, config)


if __name__ == "__main__":
    # Test uncertainty components
    print("Uncertainty Components Test")
    print("=" * 40)
    
    # Test configuration
    feature_dim = 64
    n_classes = 5
    
    # Test data
    query_features = torch.randn(10, feature_dim)
    support_features = torch.randn(25, feature_dim)  # 5-way 5-shot
    support_labels = torch.tensor([0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 
                                  3, 3, 3, 3, 3, 4, 4, 4, 4, 4])
    
    # Test different uncertainty methods
    methods = ["monte_carlo_dropout", "deep_ensemble", "evidential"]
    
    for method in methods:
        print(f"\nTesting {method}...")
        
        uncertainty_estimator = create_uncertainty_aware_distance(
            feature_dim=feature_dim,
            method=method,
            num_classes=n_classes
        )
        
        # Forward pass
        results = uncertainty_estimator(query_features, support_features, support_labels)
        
        # Extract metrics
        predictions = uncertainty_estimator.get_predictions(results)
        confidence = uncertainty_estimator.get_confidence_scores(results)
        
        print(f"  Predictions shape: {predictions.shape}")
        print(f"  Mean confidence: {confidence.mean():.4f}")
        print(f"  Mean total uncertainty: {results['total_uncertainty'].mean():.4f}")
        print(f"  Mean epistemic uncertainty: {results['epistemic_uncertainty'].mean():.4f}")
        print(f"  Mean aleatoric uncertainty: {results['aleatoric_uncertainty'].mean():.4f}")
    
    print("\n✓ All uncertainty estimation methods tested successfully!")