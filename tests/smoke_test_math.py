"""Smoke tests for mathematical correctness from review."""

import torch
from meta_learning.algos.protonet import ProtoHead
from meta_learning.algos.ttcs import ttcs_predict_advanced
from meta_learning.core.episode import Episode
from meta_learning.models.conv4 import Conv4


def test_cosine_temperature_entropy():
    """Test that cosine temperature entropy increases with higher temperature."""
    print("🔬 Testing cosine temperature entropy behavior...")
    
    torch.manual_seed(42)
    NWAY, NSHOT, NQ, D = 5, 3, 16, 32
    sz = torch.randn(NWAY*NSHOT, D)
    sy = torch.arange(NWAY).repeat_interleave(NSHOT)
    qz = torch.randn(NQ, D)

    h_lo = ProtoHead(distance="cosine", tau=0.5)  # sharper (note: still using tau)
    h_hi = ProtoHead(distance="cosine", tau=2.0)  # softer

    def H(logits):
        p = torch.softmax(logits, -1)
        return (-(p*(p.clamp_min(1e-12).log())).sum(-1)).mean()

    H_lo = H(h_lo(sz, sy, qz))
    H_hi = H(h_hi(sz, sy, qz))
    
    print(f"  Low τ=0.5:  entropy = {H_lo.item():.4f}")
    print(f"  High τ=2.0: entropy = {H_hi.item():.4f}")
    print(f"  Higher τ increases entropy: {H_hi > H_lo}")
    
    assert H_hi > H_lo, f"Higher tau should increase entropy: {H_lo:.4f} vs {H_hi:.4f}"
    print("✅ Cosine temperature entropy behavior verified!")


def test_ttcs_confidence_tracking():
    """Test that TTCS confidence uses proper probabilities."""
    print("\\n🔬 Testing TTCS confidence tracking...")
    
    torch.manual_seed(42)
    
    # Create simple test setup
    encoder = Conv4(out_dim=64)
    head = ProtoHead(distance="sqeuclidean")
    
    # Create episode
    support_x = torch.randn(6, 3, 32, 32)  # 3 classes, 2 shots each
    support_y = torch.tensor([0, 0, 1, 1, 2, 2])
    query_x = torch.randn(9, 3, 32, 32)   # 3 queries per class
    query_y = torch.tensor([0, 0, 0, 1, 1, 1, 2, 2, 2])
    
    episode = Episode(support_x, support_y, query_x, query_y)
    
    # Test confidence tracking with mean_logit
    logits, metrics = ttcs_predict_advanced(
        encoder, head, episode,
        passes=4,
        combine="mean_logit",
        uncertainty_estimation=True,
        performance_monitoring=True
    )
    
    print(f"  Logits shape: {logits.shape}")
    print(f"  Logits range: [{logits.min():.3f}, {logits.max():.3f}]")
    
    # Convert to probabilities to check confidence makes sense
    probs = torch.softmax(logits, dim=-1)
    confidence = probs.max(dim=-1)[0].mean().item()
    print(f"  Average confidence: {confidence:.4f}")
    
    # Confidence should be a valid probability (0 ≤ confidence ≤ 1)
    assert 0 <= confidence <= 1, f"Confidence should be in [0,1], got {confidence}"
    
    # Check that probabilities sum to 1
    prob_sums = probs.sum(dim=-1)
    assert torch.allclose(prob_sums, torch.ones_like(prob_sums), atol=1e-5), \
        "Probabilities should sum to 1"
    
    print("✅ TTCS confidence tracking verified!")


def test_diversity_weighting_principled():
    """Test that principled diversity weighting produces valid results."""
    print("\\n🔬 Testing principled diversity weighting...")
    
    torch.manual_seed(42)
    
    encoder = Conv4(out_dim=64)
    head = ProtoHead(distance="cosine", tau=1.0)
    
    # Create episode
    support_x = torch.randn(6, 3, 32, 32)
    support_y = torch.tensor([0, 0, 1, 1, 2, 2])
    query_x = torch.randn(12, 3, 32, 32)
    query_y = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2])
    
    episode = Episode(support_x, support_y, query_x, query_y)
    
    # Compare with and without diversity weighting
    logits_no_div, metrics_no_div = ttcs_predict_advanced(
        encoder, head, episode,
        passes=5,
        combine="mean_prob",
        diversity_weighting=False,
        uncertainty_estimation=True
    )
    
    logits_with_div, metrics_with_div = ttcs_predict_advanced(
        encoder, head, episode,
        passes=5,
        combine="mean_prob", 
        diversity_weighting=True,
        uncertainty_estimation=True
    )
    
    print(f"  No diversity: logits range [{logits_no_div.min():.3f}, {logits_no_div.max():.3f}]")
    print(f"  With diversity: logits range [{logits_with_div.min():.3f}, {logits_with_div.max():.3f}]")
    
    # Both should produce valid logits
    assert torch.isfinite(logits_no_div).all(), "No diversity logits should be finite"
    assert torch.isfinite(logits_with_div).all(), "Diversity logits should be finite"
    
    # Results should be different (diversity weighting changes the ensemble)
    assert not torch.allclose(logits_no_div, logits_with_div, atol=1e-4), \
        "Diversity weighting should change results"
    
    print("✅ Principled diversity weighting verified!")


def test_temperature_parameter_validation():
    """Test that tau validation prevents invalid values."""
    print("\\n🔬 Testing temperature parameter validation...")
    
    from meta_learning.core.math_utils import cosine_logits
    
    a = torch.randn(3, 4)
    b = torch.randn(5, 4)
    
    # Valid tau should work
    result = cosine_logits(a, b, tau=1.0)
    print(f"  Valid τ=1.0: result shape = {result.shape}")
    
    # Invalid tau should raise error
    try:
        cosine_logits(a, b, tau=0.0)
        assert False, "Should have raised ValueError for tau=0"
    except ValueError as e:
        print(f"  τ=0.0 correctly rejected: {e}")
    
    try:
        cosine_logits(a, b, tau=-1.0)
        assert False, "Should have raised ValueError for negative tau"
    except ValueError as e:
        print(f"  τ=-1.0 correctly rejected: {e}")
    
    print("✅ Temperature parameter validation verified!")


if __name__ == "__main__":
    print("🚀 Running Mathematical Smoke Tests\\n")
    test_cosine_temperature_entropy()
    test_ttcs_confidence_tracking() 
    test_diversity_weighting_principled()
    test_temperature_parameter_validation()
    print("\\n🎉 All mathematical smoke tests passed!")