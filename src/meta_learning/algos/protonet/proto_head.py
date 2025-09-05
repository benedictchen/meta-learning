import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple
from ...core.math_utils import pairwise_sqeuclidean, cosine_logits

class ProtoHead(nn.Module):
    """Professional Prototypical Networks head implementation.
    
    Based on Snell et al. 2017: "Prototypical Networks for Few-shot Learning"
    
    Computes class prototypes as:
        c_k = (1/|S_k|) × Σ(x_i ∈ S_k) f_φ(x_i)
    
    And classification via distance to prototypes:
        P(y=k|x) ∝ exp(-d(f_φ(x), c_k))
    """
    
    def __init__(self, distance: str = "sqeuclidean", tau: float = 1.0):
        """Initialize ProtoHead.
        
        Args:
            distance: Distance metric ("sqeuclidean" or "cosine")
            tau: Temperature parameter for logit scaling
        """
        super().__init__()
        if distance not in {"sqeuclidean", "cosine"}:
            raise ValueError("distance must be 'sqeuclidean' or 'cosine'")
        self.distance = distance
        self.register_buffer("_tau", torch.tensor(float(tau)))

    def forward(self, z_support: torch.Tensor, y_support: torch.Tensor, z_query: torch.Tensor) -> torch.Tensor:
        """Compute prototypical classification logits.
        
        Args:
            z_support: Support embeddings (N_s, D)
            y_support: Support labels (N_s,)  
            z_query: Query embeddings (N_q, D)
            
        Returns:
            Classification logits (N_q, K) where K is number of classes
        """
        classes = torch.unique(y_support)
        remap = {c.item(): i for i, c in enumerate(classes)}
        y = torch.tensor([remap[int(c.item())] for c in y_support], device=y_support.device)
        
        # Compute prototypes: c_k = (1/|S_k|) × Σ(x_i ∈ S_k) f_φ(x_i)
        protos = torch.stack([z_support[y == i].mean(dim=0) for i in range(len(classes))], dim=0)
        
        # Compute logits based on distance metric
        if self.distance == "sqeuclidean":
            dist = pairwise_sqeuclidean(z_query, protos)
            logits = -self._tau * dist  # P(y=k|x) ∝ exp(-d(f_φ(x), c_k))
        else:
            logits = cosine_logits(z_query, protos, tau=float(self._tau.item()))
            
        return logits