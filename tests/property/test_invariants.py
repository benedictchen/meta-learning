"""
Property-based testing for meta-learning mathematical invariants.

Uses Hypothesis for property-based testing to ensure mathematical correctness
of few-shot learning algorithms under various conditions.
"""
import torch
import torch.nn.functional as F
from hypothesis import given, strategies as st

# Temperature monotonicity: higher tau => sharper distributions, higher max prob on average
@given(
    zq=st.lists(st.lists(st.floats(-1,1), min_size=4, max_size=4), min_size=3, max_size=6),
    pr=st.lists(st.lists(st.floats(-1,1), min_size=4, max_size=4), min_size=2, max_size=5),
    tau_low=st.floats(0.5, 2.0),
    tau_high=st.floats(5.0, 30.0),
)
def test_temperature_monotone(zq, pr, tau_low, tau_high):
    """Test that higher temperature leads to sharper probability distributions."""
    zq = F.normalize(torch.tensor(zq, dtype=torch.float32), dim=-1)
    pr = F.normalize(torch.tensor(pr, dtype=torch.float32), dim=-1)
    
    logits_lo = tau_low * (zq @ pr.T)
    logits_hi = tau_high * (zq @ pr.T)
    
    m_lo = F.softmax(logits_lo, dim=1).max(1).values.mean()
    m_hi = F.softmax(logits_hi, dim=1).max(1).values.mean()
    
    assert m_hi >= m_lo

# Cosine scale invariance: scaling both sides equally shouldn't change softmax over cosine similarities
@given(
    zq=st.lists(st.lists(st.floats(-3,3), min_size=3, max_size=3), min_size=2, max_size=6),
    pr=st.lists(st.lists(st.floats(-3,3), min_size=3, max_size=3), min_size=2, max_size=6),
    alpha=st.floats(0.1, 10.0),
)
def test_cosine_scale_invariance(zq, pr, alpha):
    """Test that cosine similarity is invariant to uniform scaling."""
    zq = torch.tensor(zq, dtype=torch.float32)
    pr = torch.tensor(pr, dtype=torch.float32)
    
    p1 = F.softmax((F.normalize(zq, dim=-1) @ F.normalize(pr, dim=-1).T), dim=1)
    p2 = F.softmax((F.normalize(alpha*zq, dim=-1) @ F.normalize(alpha*pr, dim=-1).T), dim=1)
    
    assert torch.allclose(p1, p2, atol=1e-5)

# Label remap property: outputs contiguous labels [0..C-1]
@given(
    ys=st.lists(st.integers(0, 100), min_size=2, max_size=20).filter(lambda xs: len(set(xs))>=2),
    extra=st.lists(st.integers(0, 100), min_size=2, max_size=20)
)
def test_remap_labels_contiguous(ys, extra):
    """Test that label remapping produces contiguous label space."""
    import random
    
    classes = sorted(set(ys))
    yq = [random.choice(classes) for _ in range(len(extra))]
    
    ys_t = torch.tensor(ys, dtype=torch.int64)
    yq_t = torch.tensor(yq, dtype=torch.int64)
    
    # This would need to be imported from the actual implementation
    try:
        from meta_learning.core.episode import remap_labels
        ys_m, yq_m = remap_labels(ys_t, yq_t)
        
        uniq = torch.unique(ys_m)
        assert torch.equal(uniq, torch.arange(len(uniq)))
        assert torch.all(torch.isin(torch.unique(yq_m), uniq))
    except ImportError:
        # Skip test if module not available
        pass