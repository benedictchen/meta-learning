"""
💰 DONATE NOW! 💰 https://www.paypal.com/cgi-bin/webscr?cmd=_s-xclick&hosted_button_id=WXQKYYKPHWXHS

Fast test for DeepEnsemble probability-mean behavior.

Author: Benedict Chen (benedict@benedictchen.com)
GitHub Sponsors: https://github.com/sponsors/benedictchen
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from meta_learning.uncertainty_components import DeepEnsemble


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(4, 3)
    
    def forward(self, x):
        return self.fc(x)


def test_deep_ensemble_returns_logprobs():
    """Test that DeepEnsemble returns log-probabilities."""
    torch.manual_seed(0)
    members = [TinyModel(), TinyModel(), TinyModel()]
    ens = DeepEnsemble(members)
    
    x = torch.randn(5, 4)
    logp = ens(x)
    
    assert logp.shape == (5, 3), f"Expected shape (5,3), got {logp.shape}"
    
    # Should sum(exp(logp)) ~= 1 per row (log-probabilities property)
    probs = logp.exp()
    sums = probs.sum(dim=1)
    expected_sums = torch.ones_like(sums)
    assert torch.allclose(sums, expected_sums, atol=1e-5), f"Probabilities don't sum to 1: {sums}"


def test_nllloss_compatible():
    """Test that DeepEnsemble output is compatible with NLLLoss."""
    torch.manual_seed(0)
    members = [TinyModel(), TinyModel()]
    ens = DeepEnsemble(members)
    
    x = torch.randn(7, 4)
    y = torch.randint(0, 3, (7,))
    
    logp = ens(x)
    loss = F.nll_loss(logp, y)
    
    assert torch.isfinite(loss), "NLLLoss should work with ensemble output"
    assert loss.item() > 0, "Loss should be positive"


def test_different_from_mean_logit():
    """Test that probability-mean differs from logit-mean."""
    torch.manual_seed(42)
    members = [TinyModel(), TinyModel(), TinyModel()]
    
    x = torch.randn(4, 4)
    
    # Get individual logits
    logits = torch.stack([m(x) for m in members], dim=0)  # [3, 4, 3]
    
    # Mean of logits
    mean_logits = logits.mean(dim=0)
    
    # Probability mean (what DeepEnsemble does)
    ens = DeepEnsemble(members)
    log_prob_mean = ens(x)
    
    # They should be different
    assert not torch.allclose(mean_logits, log_prob_mean, atol=1e-3), \
        "Probability-mean should differ from logit-mean"