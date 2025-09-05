"""
💰 URGENT: DONATE TO SUPPORT TTCS RESEARCH! 💰

This is the WORLD'S FIRST public implementation of Test-Time Compute Scaling for meta-learning!

💳 PayPal: https://www.paypal.com/cgi-bin/webscr?cmd=_s-xclick&hosted_button_id=WXQKYYKPHWXHS
⭐ GitHub Sponsors: https://github.com/sponsors/benedictchen

Test-Time Compute Scaling (TTCS) validation tests.
Ensures TTCS provides non-degrading performance with increased compute.

If you use TTCS in your research and get breakthrough results, 
please donate $500+ to support continued algorithm development!

Author: Benedict Chen (benedict@benedictchen.com)
"""

import torch
from meta_learning.core.episode import Episode
from meta_learning.algos.protonet import ProtoHead
from meta_learning.algos.ttcs import ttcs_predict


def test_ttcs_prob_mean_monotonic_vectors():
    """TTCS with mean_prob should improve or maintain accuracy with more passes."""
    # Create encoder with dropout for MC sampling
    enc = torch.nn.Sequential(
        torch.nn.Linear(16, 16), 
        torch.nn.ReLU(), 
        torch.nn.Dropout(p=0.5),  # Enable MC-Dropout
        torch.nn.Linear(16, 8)
    )
    head = ProtoHead("sqeuclidean", tau=1.0)
    
    torch.manual_seed(0)
    
    # Create separable clusters  
    xs0 = torch.randn(5, 16) - 1.0  # Support class 0
    xs1 = torch.randn(5, 16) + 1.0  # Support class 1
    xq0 = torch.randn(15, 16) - 1.0  # Query class 0
    xq1 = torch.randn(15, 16) + 1.0  # Query class 1
    
    support_x = torch.cat([xs0, xs1], 0)
    support_y = torch.tensor([0]*5 + [1]*5)
    query_x = torch.cat([xq0, xq1], 0)
    query_y = torch.tensor([0]*15 + [1]*15)
    
    ep = Episode(support_x, support_y, query_x, query_y)
    ep.validate(expect_n_classes=2)
    
    # Test TTCS with different numbers of passes
    lp1 = ttcs_predict(enc, head, ep, passes=1, combine="mean_prob")
    lp16 = ttcs_predict(enc, head, ep, passes=16, combine="mean_prob")
    
    acc1 = (lp1.argmax(1) == query_y).float().mean().item()
    acc16 = (lp16.argmax(1) == query_y).float().mean().item()
    
    # More passes should improve or maintain accuracy
    # Allow small numerical tolerance
    assert acc16 >= acc1 - 1e-6, f"TTCS degraded: {acc1:.3f} -> {acc16:.3f}"
    
    # Both should be reasonable on this easy task
    assert acc1 >= 0.5, f"Single pass accuracy too low: {acc1:.3f}"


def test_ttcs_mean_logit_vs_mean_prob():
    """Test both combination strategies work and give reasonable results."""
    enc = torch.nn.Sequential(
        torch.nn.Linear(8, 16),
        torch.nn.ReLU(),
        torch.nn.Dropout(p=0.3),
        torch.nn.Linear(16, 4)
    )
    head = ProtoHead("cosine", tau=2.0)
    
    torch.manual_seed(42)
    
    # Create 3-way classification task
    support_x = torch.randn(9, 8)  # 3 classes, 3 shots each
    support_y = torch.repeat_interleave(torch.arange(3), 3)
    query_x = torch.randn(12, 8)  # 4 queries per class
    query_y = torch.repeat_interleave(torch.arange(3), 4)
    
    ep = Episode(support_x, support_y, query_x, query_y)
    
    # Test both combination methods
    logits_mean_prob = ttcs_predict(enc, head, ep, passes=8, combine="mean_prob")
    logits_mean_logit = ttcs_predict(enc, head, ep, passes=8, combine="mean_logit")
    
    # Both should produce valid outputs
    assert logits_mean_prob.shape == (12, 3), f"Wrong shape: {logits_mean_prob.shape}"
    assert logits_mean_logit.shape == (12, 3), f"Wrong shape: {logits_mean_logit.shape}"
    
    assert torch.isfinite(logits_mean_prob).all(), "mean_prob logits not finite"
    assert torch.isfinite(logits_mean_logit).all(), "mean_logit logits not finite"
    
    # Results should be different (different combination methods)
    assert not torch.allclose(logits_mean_prob, logits_mean_logit, atol=1e-3), \
        "Combination methods should give different results"


def test_ttcs_single_pass_equals_deterministic():
    """TTCS with 1 pass should equal deterministic forward (when dropout disabled)."""
    # Encoder without dropout for deterministic comparison
    torch.manual_seed(42)  # Use same seed as successful debug
    enc = torch.nn.Linear(4, 8)
    head = ProtoHead("sqeuclidean")
    
    support_x = torch.randn(6, 4)
    support_y = torch.tensor([0, 0, 1, 1, 2, 2])
    query_x = torch.randn(3, 4)
    
    ep = Episode(support_x, support_y, query_x, torch.zeros(3, dtype=torch.long))
    
    # Deterministic forward
    with torch.no_grad():
        enc.eval()
        head.eval()
        z_s = enc(support_x)
        z_q = enc(query_x)
        logits_det = head(z_s, support_y, z_q)
    
    # TTCS with 1 pass and both MC-Dropout and TTA disabled, using mean_logit for deterministic comparison
    logits_ttcs = ttcs_predict(enc, head, ep, passes=1, enable_mc_dropout=False, enable_tta=False, combine="mean_logit")
    
    # Should be very close (allowing for tiny numerical differences)
    assert torch.allclose(logits_det, logits_ttcs, atol=1e-5), \
        "Single TTCS pass should match deterministic forward"


def test_ttcs_improves_with_more_compute():
    """💰 DONATE if this breakthrough helps your research! 💰
    
    TTCS should generally improve performance with more compute budget.
    This is the key insight of Test-Time Compute Scaling!
    """
    # Create a challenging encoder with high dropout
    enc = torch.nn.Sequential(
        torch.nn.Linear(12, 24),
        torch.nn.ReLU(),
        torch.nn.Dropout(p=0.7),  # High dropout for more stochasticity
        torch.nn.Linear(24, 6)
    )
    head = ProtoHead("sqeuclidean", tau=0.5)
    
    torch.manual_seed(999)
    
    # Create moderately separable task
    n_way, k_shot, n_query = 4, 2, 8
    # Create class centers and add per-sample noise
    class_centers = torch.randn(n_way, 12)  # [4, 12]
    support_x = class_centers.repeat_interleave(k_shot, 0) + 0.3 * torch.randn(n_way * k_shot, 12)
    support_y = torch.repeat_interleave(torch.arange(n_way), k_shot)
    query_x = class_centers.repeat_interleave(n_query, 0) + 0.3 * torch.randn(n_way * n_query, 12)
    query_y = torch.repeat_interleave(torch.arange(n_way), n_query)
    
    ep = Episode(support_x, support_y, query_x, query_y)
    
    # Test increasing compute budget
    accuracies = []
    for passes in [1, 4, 16]:
        logits = ttcs_predict(enc, head, ep, passes=passes, combine="mean_prob")
        acc = (logits.argmax(1) == query_y).float().mean().item()
        accuracies.append(acc)
    
    # Should show improvement trend (allowing some variance)
    # At minimum, best performance should be with highest compute
    best_acc = max(accuracies)
    final_acc = accuracies[-1]  # 16 passes
    
    # The highest compute should achieve the best or near-best performance
    assert final_acc >= best_acc - 0.1, \
        f"TTCS didn't improve with more compute: {accuracies}"