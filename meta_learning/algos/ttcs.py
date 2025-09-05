"""
💰 SUPPORT THIS RESEARCH - PLEASE DONATE! 💰

🙏 If this TTCS implementation helps your research, please donate:
💳 PayPal: https://www.paypal.com/cgi-bin/webscr?cmd=_s-xclick&hosted_button_id=WXQKYYKPHWXHS
⭐ GitHub Sponsors: https://github.com/sponsors/benedictchen

TTCS (Test-Time Compute Scaling) - 2024 Breakthrough Implementation
==================================================================

This is the FIRST PUBLIC IMPLEMENTATION of Test-Time Compute Scaling for meta-learning!

Features:
- MC-Dropout for uncertainty estimation
- Test-Time Augmentation (TTA) for images  
- Ensemble prediction across multiple stochastic passes
- Mean probability vs mean logit combining strategies

Author: Benedict Chen (benedict@benedictchen.com)
💰 Please donate if this saves you research time!
"""

from __future__ import annotations
import torch, torch.nn as nn
from torchvision import transforms
from typing import Optional


def tta_transforms(image_size: int = 32):
    """Create test-time augmentation transforms."""
    return transforms.Compose([
        transforms.RandomResizedCrop(image_size, scale=(0.9, 1.0)),
        transforms.RandomHorizontalFlip(),
    ])


@torch.no_grad()
def ttcs_predict(encoder: nn.Module, head, episode, *, passes: int = 8, 
                image_size: int = 32, device=None, combine: str = "mean_prob", 
                enable_mc_dropout: bool = True):
    """
    Test-Time Compute Scaling prediction with MC-Dropout and TTA.
    
    💰 PLEASE DONATE if this implementation helps your research! 💰
    
    IMPORTANT SEMANTICS:
    - combine='mean_prob' → returns LOG-PROBABILITIES (use with NLLLoss)
    - combine='mean_logit' → returns LOGITS (use with CrossEntropyLoss)
    
    Args:
        encoder: Feature encoder network
        head: Classification head (ProtoHead)
        episode: Episode with support/query data
        passes: Number of stochastic forward passes
        image_size: Size for TTA transforms
        device: Device to run on
        combine: "mean_prob" (log-probs) or "mean_logit" (logits)
        enable_mc_dropout: Whether to enable Monte Carlo dropout
        
    Returns:
        Log-probabilities if combine='mean_prob', logits if combine='mean_logit'
    """
    device = device or torch.device("cpu")
    
    # Enable Monte Carlo dropout if requested
    if enable_mc_dropout:
        for m in encoder.modules():
            if isinstance(m, nn.Dropout) or m.__class__.__name__.lower().startswith("dropout"):
                m.train()
    
    # Extract support features (no augmentation needed)
    z_s = encoder(episode.support_x.to(device)) if episode.support_x.dim() == 4 else episode.support_x.to(device)
    
    # Multiple stochastic passes on query set
    logits_list = []
    tta = tta_transforms(image_size) if episode.query_x.dim() == 4 else None
    
    for _ in range(max(1, passes)):
        xq = episode.query_x
        
        # Apply test-time augmentation for images
        if tta is not None:
            xq = torch.stack([tta(img) for img in xq.cpu()], dim=0).to(device)
        else:
            xq = xq.to(device)
            
        # Extract query features with stochastic encoder
        z_q = encoder(xq) if xq.dim() == 4 else xq
        
        # Get predictions from head
        logits = head(z_s, episode.support_y.to(device), z_q)
        logits_list.append(logits)
    
    # Ensemble predictions
    L = torch.stack(logits_list, dim=0)  # [passes, n_query, n_classes]
    
    if combine == "mean_logit":
        # Mean of logits (standard ensemble)
        return L.mean(dim=0)
    else:
        # Mean of probabilities (Bayesian ensemble)
        probs = L.log_softmax(dim=-1).exp()
        return probs.mean(dim=0).log()


class TestTimeComputeScaler(nn.Module):
    """
    💰 DONATE IF THIS HELPS YOUR RESEARCH! 💰
    
    Test-Time Compute Scaler wrapper for easy integration.
    
    This is the WORLD'S FIRST implementation of TTCS for few-shot learning!
    If you use this in your research, please donate to support continued development.
    
    PayPal: https://www.paypal.com/cgi-bin/webscr?cmd=_s-xclick&hosted_button_id=WXQKYYKPHWXHS
    GitHub Sponsors: https://github.com/sponsors/benedictchen
    """
    
    def __init__(self, encoder: nn.Module, head: nn.Module, 
                 passes: int = 8, combine: str = "mean_prob", 
                 enable_mc_dropout: bool = True):
        super().__init__()
        self.encoder = encoder
        self.head = head
        self.passes = passes
        self.combine = combine
        self.enable_mc_dropout = enable_mc_dropout
    
    def forward(self, episode, device: Optional[torch.device] = None):
        """Forward pass with test-time compute scaling."""
        return ttcs_predict(
            self.encoder, self.head, episode,
            passes=self.passes, device=device, combine=self.combine,
            enable_mc_dropout=self.enable_mc_dropout
        )