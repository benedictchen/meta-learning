from __future__ import annotations
import torch, torch.nn as nn, torch.nn.functional as F
from typing import Optional, Dict, Any
from ..core.math_utils import pairwise_sqeuclidean, cosine_logits

class ProtoHead(nn.Module):
    def __init__(self, distance: str = "sqeuclidean", tau: float = 1.0, 
                 prototype_shrinkage: float = 0.0, uncertainty_method: Optional[str] = None,
                 dropout_rate: float = 0.1, n_uncertainty_samples: int = 10):
        super().__init__()
        if distance not in {"sqeuclidean", "cosine"}:
            raise ValueError("distance must be 'sqeuclidean' or 'cosine'")
        self.distance = distance
        self.register_buffer("_tau", torch.tensor(float(tau)))
        self.prototype_shrinkage = prototype_shrinkage
        
        # Uncertainty estimation
        self.uncertainty_method = uncertainty_method
        self.dropout_rate = dropout_rate
        self.n_uncertainty_samples = n_uncertainty_samples
        
        if uncertainty_method == "monte_carlo_dropout":
            self.dropout = nn.Dropout(dropout_rate)

    def forward(self, z_support: torch.Tensor, y_support: torch.Tensor, z_query: torch.Tensor) -> torch.Tensor:
        if self.uncertainty_method:
            return self.forward_with_uncertainty(z_support, y_support, z_query)["logits"]
        else:
            return self.forward_deterministic(z_support, y_support, z_query)
    
    def forward_deterministic(self, z_support: torch.Tensor, y_support: torch.Tensor, z_query: torch.Tensor) -> torch.Tensor:
        """Standard deterministic forward pass."""
        classes = torch.unique(y_support)
        remap = {c.item(): i for i, c in enumerate(classes)}
        y = torch.tensor([remap[int(c.item())] for c in y_support], device=y_support.device)
        
        # Compute class prototypes with optional shrinkage regularization
        protos = torch.stack([z_support[y == i].mean(dim=0) for i in range(len(classes))], dim=0)
        
        # Apply prototype shrinkage: interpolate between class prototype and global mean
        # This reduces overfitting when support sets are small
        if self.prototype_shrinkage > 0.0:
            global_mean = z_support.mean(dim=0, keepdim=True)
            protos = (1 - self.prototype_shrinkage) * protos + self.prototype_shrinkage * global_mean
        
        # Compute distances using numerically stable operations
        if self.distance == "sqeuclidean":
            # Uses ||a-b||² = ||a||² + ||b||² - 2a^Tb with clamping for stability
            dist = pairwise_sqeuclidean(z_query, protos)
            # Standard temperature scaling: logits = -dist / tau
            # Higher tau -> higher entropy (less confident)
            logits = -dist / self._tau
        else:
            # Uses ε-guarded normalization to prevent division by zero
            logits = cosine_logits(z_query, protos, tau=float(self._tau.item()))
        return logits
    
    def forward_with_uncertainty(self, z_support: torch.Tensor, y_support: torch.Tensor, 
                               z_query: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass with uncertainty estimation."""
        if self.uncertainty_method == "monte_carlo_dropout":
            return self._monte_carlo_uncertainty(z_support, y_support, z_query)
        else:
            # Fallback to deterministic
            logits = self.forward_deterministic(z_support, y_support, z_query)
            return {"logits": logits, "uncertainty": torch.zeros(logits.size(0))}
    
    def _monte_carlo_uncertainty(self, z_support: torch.Tensor, y_support: torch.Tensor, 
                               z_query: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Monte Carlo Dropout for uncertainty estimation."""
        # Store original training mode
        original_training = self.training
        self.train()  # Enable dropout
        
        logit_samples = []
        
        # Perform multiple stochastic forward passes
        for _ in range(self.n_uncertainty_samples):
            # Apply dropout to features
            if hasattr(self, 'dropout'):
                z_support_dropped = self.dropout(z_support)
                z_query_dropped = self.dropout(z_query)
            else:
                z_support_dropped = z_support
                z_query_dropped = z_query
                
            # Compute prototypes and distances
            classes = torch.unique(y_support)
            remap = {c.item(): i for i, c in enumerate(classes)}
            y = torch.tensor([remap[int(c.item())] for c in y_support], device=y_support.device)
            
            protos = torch.stack([z_support_dropped[y == i].mean(dim=0) for i in range(len(classes))], dim=0)
            
            if self.prototype_shrinkage > 0.0:
                global_mean = z_support_dropped.mean(dim=0, keepdim=True)
                protos = (1 - self.prototype_shrinkage) * protos + self.prototype_shrinkage * global_mean
            
            if self.distance == "sqeuclidean":
                dist = pairwise_sqeuclidean(z_query_dropped, protos)
                logits = -dist / self._tau
            else:
                logits = cosine_logits(z_query_dropped, protos, tau=float(self._tau.item()))
                
            logit_samples.append(logits)
        
        # Restore original training mode
        self.train(original_training)
        
        # Compute statistics
        logit_samples = torch.stack(logit_samples, dim=0)  # [n_samples, n_query, n_classes]
        mean_logits = logit_samples.mean(dim=0)
        
        # Convert to probabilities for uncertainty calculation
        prob_samples = torch.softmax(logit_samples, dim=-1)
        mean_probs = prob_samples.mean(dim=0)
        
        # Total uncertainty (entropy of mean predictions)
        eps = 1e-8
        mean_probs_safe = torch.clamp(mean_probs, min=eps, max=1.0 - eps)
        total_uncertainty = -(mean_probs_safe * torch.log(mean_probs_safe)).sum(dim=-1)
        
        # Epistemic uncertainty (mutual information)
        sample_entropies = -(prob_samples * torch.log(prob_samples + eps)).sum(dim=-1)
        aleatoric_uncertainty = sample_entropies.mean(dim=0)
        epistemic_uncertainty = total_uncertainty - aleatoric_uncertainty
        
        return {
            "logits": mean_logits,
            "probabilities": mean_probs,
            "total_uncertainty": total_uncertainty,
            "epistemic_uncertainty": epistemic_uncertainty,
            "aleatoric_uncertainty": aleatoric_uncertainty,
            "n_samples": self.n_uncertainty_samples
        }
