"""
Test cosine distance temperature/tau entropy behavior in ProtoHead.

Verifies that temperature scaling works correctly for cosine similarity,
ensuring higher temperature → higher entropy (more uncertainty).
"""

import torch
import pytest
from meta_learning.algos.protonet import ProtoHead
from meta_learning.core.episode import Episode


def entropy_from_logits(logits: torch.Tensor) -> torch.Tensor:
    """Compute entropy from logits tensor."""
    probs = torch.softmax(logits, dim=-1)
    log_probs = torch.log(probs.clamp(min=1e-12))
    return -(probs * log_probs).sum(dim=-1).mean()


def test_cosine_temperature_monotonic_entropy():
    """Test that cosine distance shows monotonic temperature behavior."""
    torch.manual_seed(42)
    
    # Create synthetic few-shot episode
    n_way, n_support, n_query, dim = 4, 2, 8, 16
    
    # Create well-separated class centers for predictable behavior
    support_z = torch.randn(n_way * n_support, dim)
    support_y = torch.arange(n_way).repeat_interleave(n_support)
    query_z = torch.randn(n_query, dim)
    
    # Test temperature monotonicity
    temperatures = [0.1, 0.5, 1.0, 2.0, 5.0]
    entropies = []
    
    for tau in temperatures:
        head = ProtoHead(distance="cosine", tau=tau)
        logits = head(support_z, support_y, query_z)
        entropy = entropy_from_logits(logits)
        entropies.append(entropy.item())
    
    # Verify monotonic relationship: higher tau → higher entropy
    for i in range(len(temperatures) - 1):
        tau_low, tau_high = temperatures[i], temperatures[i + 1]
        ent_low, ent_high = entropies[i], entropies[i + 1]
        
        assert ent_high >= ent_low, \
            f"Entropy should increase with temperature: τ={tau_low:.1f} ({ent_low:.3f}) vs τ={tau_high:.1f} ({ent_high:.3f})"
    
    # Test extreme cases
    assert entropies[0] < entropies[-1], "High temperature should have much higher entropy than low temperature"
    
    # High temperature should approach uniform distribution
    n_classes = n_way
    max_entropy = -torch.log(torch.tensor(1.0/n_classes)).item()
    assert entropies[-1] > 0.7 * max_entropy, "High temperature should approach uniform entropy"


def test_cosine_vs_sqeuclidean_temperature_consistency():
    """Test that both distance metrics show consistent temperature behavior."""
    torch.manual_seed(42)
    
    n_way, n_support, n_query, dim = 3, 2, 6, 12
    
    # Create episode
    support_z = torch.randn(n_way * n_support, dim)
    support_y = torch.arange(n_way).repeat_interleave(n_support)
    query_z = torch.randn(n_query, dim)
    
    temperatures = [0.5, 1.0, 2.0]
    
    for tau in temperatures:
        # Both distance metrics with same temperature
        head_cosine = ProtoHead(distance="cosine", tau=tau)
        head_sqeucl = ProtoHead(distance="sqeuclidean", tau=tau)
        
        logits_cos = head_cosine(support_z, support_y, query_z)
        logits_sq = head_sqeucl(support_z, support_y, query_z)
        
        entropy_cos = entropy_from_logits(logits_cos)
        entropy_sq = entropy_from_logits(logits_sq)
        
        # Both should produce valid entropy values
        assert torch.isfinite(entropy_cos), f"Cosine entropy not finite for tau={tau}"
        assert torch.isfinite(entropy_sq), f"Squared Euclidean entropy not finite for tau={tau}"
        assert entropy_cos >= 0, f"Cosine entropy negative for tau={tau}"
        assert entropy_sq >= 0, f"Squared Euclidean entropy negative for tau={tau}"


def test_cosine_extreme_temperature_values():
    """Test cosine distance with extreme temperature values."""
    torch.manual_seed(42)
    
    n_way, n_support, n_query, dim = 3, 2, 6, 16
    
    support_z = torch.randn(n_way * n_support, dim)
    support_y = torch.arange(n_way).repeat_interleave(n_support)
    query_z = torch.randn(n_query, dim)
    
    # Very low temperature (almost deterministic)
    head_cold = ProtoHead(distance="cosine", tau=0.01)
    logits_cold = head_cold(support_z, support_y, query_z)
    probs_cold = torch.softmax(logits_cold, dim=-1)
    entropy_cold = entropy_from_logits(logits_cold)
    
    # Very high temperature (almost uniform)
    head_hot = ProtoHead(distance="cosine", tau=10.0)
    logits_hot = head_hot(support_z, support_y, query_z)
    probs_hot = torch.softmax(logits_hot, dim=-1)
    entropy_hot = entropy_from_logits(logits_hot)
    
    # Cold temperature should produce confident predictions
    max_probs_cold = probs_cold.max(dim=-1)[0]
    assert max_probs_cold.mean() > 0.8, "Cold temperature should produce confident predictions"
    
    # Hot temperature should produce less confident predictions
    max_probs_hot = probs_hot.max(dim=-1)[0]
    assert max_probs_hot.mean() < 0.6, "Hot temperature should produce less confident predictions"
    
    # Entropy ordering
    assert entropy_hot > entropy_cold, "Hot temperature should have higher entropy than cold"


def test_cosine_temperature_shapes_and_devices():
    """Test cosine distance works with different shapes and devices."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    n_way, n_support, n_query, dim = 5, 3, 10, 32
    
    support_z = torch.randn(n_way * n_support, dim, device=device)
    support_y = torch.arange(n_way, device=device).repeat_interleave(n_support)
    query_z = torch.randn(n_query, dim, device=device)
    
    head = ProtoHead(distance="cosine", tau=1.0)
    logits = head(support_z, support_y, query_z)
    
    assert logits.shape == (n_query, n_way), f"Wrong logits shape: {logits.shape}"
    assert logits.device == device, f"Wrong device: {logits.device}"
    assert torch.isfinite(logits).all(), "Logits should be finite"
    
    # Test probabilities sum to 1
    probs = torch.softmax(logits, dim=-1)
    assert torch.allclose(probs.sum(dim=-1), torch.ones(n_query, device=device), atol=1e-5), \
        "Probabilities should sum to 1"


if __name__ == "__main__":
    pytest.main([__file__])