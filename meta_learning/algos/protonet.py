from __future__ import annotations
import torch, torch.nn as nn
from ..core.math_utils import pairwise_sqeuclidean, cosine_logits

class ProtoHead(nn.Module):
    def __init__(self, distance: str = "sqeuclidean", tau: float = 1.0, prototype_shrinkage: float = 0.0):
        super().__init__()
        if distance not in {"sqeuclidean", "cosine"}:
            raise ValueError("distance must be 'sqeuclidean' or 'cosine'")
        self.distance = distance
        self.register_buffer("_tau", torch.tensor(float(tau)))
        self.prototype_shrinkage = prototype_shrinkage

    def forward(self, z_support: torch.Tensor, y_support: torch.Tensor, z_query: torch.Tensor) -> torch.Tensor:
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
            logits = -self._tau * dist
        else:
            # Uses ε-guarded normalization to prevent division by zero
            logits = cosine_logits(z_query, protos, tau=float(self._tau.item()))
        return logits
