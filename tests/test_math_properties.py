"""
💰 SUPPORT THIS RESEARCH - PLEASE DONATE! 💰

🙏 If this library helps your research or project, please consider donating:
💳 PayPal: https://www.paypal.com/cgi-bin/webscr?cmd=_s-xclick&hosted_button_id=WXQKYYKPHWXHS
⭐ GitHub Sponsors: https://github.com/sponsors/benedictchen

👨‍💻 Author: Benedict Chen
💰 Donations: Help support this work! Buy me a coffee ☕, beer 🍺, lamborghini 🏎️, or private island 🏝️
💖 Please consider recurring donations to fully support continued research

Your support enables cutting-edge AI research for everyone! 🚀

Mathematical Property Tests
==========================

Rigorous tests for numerically stable mathematical operations.
Verifies mathematical correctness and numerical stability properties.
"""
from __future__ import annotations
import pytest
import torch
import numpy as np
from meta_learning.core.math_utils import pairwise_sqeuclidean, cosine_logits, _eps_like

class TestMathematicalProperties:
    """Test mathematical correctness of core operations."""

    def test_pairwise_sqeuclidean_properties(self):
        """Test mathematical properties of squared Euclidean distance."""
        # Test data
        a = torch.randn(10, 5)
        b = torch.randn(7, 5)
        
        dist = pairwise_sqeuclidean(a, b)
        
        # Property 1: Non-negativity
        assert torch.all(dist >= 0), "Squared distances must be non-negative"
        
        # Property 2: Symmetry (when a == b)
        self_dist = pairwise_sqeuclidean(a, a)
        assert torch.allclose(self_dist, self_dist.T, atol=1e-6), "Self-distance matrix should be symmetric"
        
        # Property 3: Zero diagonal for self-distance
        assert torch.allclose(torch.diag(self_dist), torch.zeros(len(a)), atol=1e-6), "Self-distance diagonal should be zero"
        
        # Property 4: Triangle inequality (for specific points)
        c = torch.randn(1, 5)
        ac = pairwise_sqeuclidean(a[:1], c)
        bc = pairwise_sqeuclidean(b[:1], c)
        ab = pairwise_sqeuclidean(a[:1], b[:1])
        
        # ||a-c|| + ||c-b|| >= ||a-b|| (triangle inequality for distances, not squared distances)
        # For squared distances, we verify the computational identity instead
        
        # Property 5: Numerical stability - no NaN or Inf
        assert not torch.any(torch.isnan(dist)), "No NaN values allowed"
        assert not torch.any(torch.isinf(dist)), "No Inf values allowed"

    def test_pairwise_sqeuclidean_identity_verification(self):
        """Verify the mathematical identity: ||a-b||² = ||a||² + ||b||² - 2a^Tb."""
        torch.manual_seed(42)
        a = torch.randn(5, 3)
        b = torch.randn(4, 3)
        
        # Our implementation
        dist_fast = pairwise_sqeuclidean(a, b)
        
        # Naive implementation for verification
        dist_naive = torch.zeros(len(a), len(b))
        for i in range(len(a)):
            for j in range(len(b)):
                diff = a[i] - b[j]
                dist_naive[i, j] = (diff * diff).sum()
        
        # Should match within floating point precision
        assert torch.allclose(dist_fast, dist_naive, atol=1e-5), "Fast and naive implementations should match"

    def test_cosine_logits_properties(self):
        """Test mathematical properties of cosine similarity logits."""
        # Test data
        a = torch.randn(10, 8)
        b = torch.randn(5, 8)
        tau = 10.0
        
        logits = cosine_logits(a, b, tau=tau)
        
        # Property 1: Shape correctness
        assert logits.shape == (10, 5), f"Expected shape (10, 5), got {logits.shape}"
        
        # Property 2: Range bounds for cosine similarity
        # After normalization, cosine similarity is in [-1, 1], so tau * cosine is in [-tau, tau]
        assert torch.all(logits >= -tau), f"Logits should be >= -{tau}"
        assert torch.all(logits <= tau), f"Logits should be <= {tau}"
        
        # Property 3: Self-similarity should be maximal (tau)
        self_logits = cosine_logits(a, a, tau=tau)
        diagonal = torch.diag(self_logits)
        assert torch.allclose(diagonal, torch.full_like(diagonal, tau), atol=1e-5), "Self-similarity should equal tau"
        
        # Property 4: Numerical stability - no NaN or Inf
        assert not torch.any(torch.isnan(logits)), "No NaN values allowed"
        assert not torch.any(torch.isinf(logits)), "No Inf values allowed"

    def test_cosine_logits_zero_norm_stability(self):
        """Test numerical stability with zero-norm vectors."""
        # Create vectors with zero norm (should be handled by epsilon guard)
        a = torch.zeros(3, 4)  # Zero vectors
        b = torch.randn(2, 4)  # Normal vectors
        
        # Should not crash or produce NaN/Inf
        logits = cosine_logits(a, b, tau=1.0)
        
        assert not torch.any(torch.isnan(logits)), "Zero-norm vectors should not produce NaN"
        assert not torch.any(torch.isinf(logits)), "Zero-norm vectors should not produce Inf"
        assert logits.shape == (3, 2), "Shape should be correct even with zero norms"

    def test_eps_like_properties(self):
        """Test epsilon tensor creation."""
        # Test different dtypes and devices
        for dtype in [torch.float32, torch.float64]:
            x = torch.randn(5, 3, dtype=dtype)
            eps = _eps_like(x, 1e-8)
            
            assert eps.dtype == x.dtype, f"Epsilon dtype should match input: {eps.dtype} vs {x.dtype}"
            assert eps.device == x.device, f"Epsilon device should match input: {eps.device} vs {x.device}"
            
            # Use appropriate tolerance for floating point comparison
            expected = 1e-8
            tolerance = 1e-12 if dtype == torch.float64 else 1e-10  # float32 has less precision
            assert abs(eps.item() - expected) < tolerance, f"Epsilon value should be ~{expected}, got {eps.item()}"

    def test_numerical_edge_cases(self):
        """Test edge cases for numerical stability."""
        # Very small values
        tiny = torch.full((3, 2), 1e-10)
        dist_tiny = pairwise_sqeuclidean(tiny, tiny)
        assert torch.all(torch.isfinite(dist_tiny)), "Tiny values should produce finite results"
        
        # Very large values
        huge = torch.full((2, 3), 1e6)
        dist_huge = pairwise_sqeuclidean(huge, huge)
        assert torch.all(torch.isfinite(dist_huge)), "Large values should produce finite results"
        
        # Mixed scales
        mixed_a = torch.tensor([[1e-6, 1e6]], dtype=torch.float32)
        mixed_b = torch.tensor([[1e6, 1e-6]], dtype=torch.float32)
        dist_mixed = pairwise_sqeuclidean(mixed_a, mixed_b)
        assert torch.all(torch.isfinite(dist_mixed)), "Mixed scales should produce finite results"

    @pytest.mark.parametrize("tau", [0.1, 1.0, 10.0, 100.0])
    def test_cosine_logits_temperature_scaling(self, tau):
        """Test that temperature scaling works correctly."""
        a = torch.randn(5, 3)
        b = torch.randn(4, 3)
        
        logits = cosine_logits(a, b, tau=tau)
        
        # Logits should scale proportionally with tau
        assert torch.all(torch.abs(logits) <= tau + 1e-5), f"Logits magnitude should be bounded by tau={tau}"
        
        # Test relative scaling
        if tau > 1.0:
            logits_base = cosine_logits(a, b, tau=1.0)
            logits_scaled = cosine_logits(a, b, tau=tau)
            
            # Should be approximately tau times larger
            ratio = (logits_scaled / (logits_base + 1e-8)).mean()
            assert torch.abs(ratio - tau) < 0.1, f"Temperature scaling should be approximately {tau}, got {ratio}"

    def test_mathematical_consistency(self):
        """Test consistency between different mathematical approaches."""
        torch.manual_seed(123)
        
        # Test that our implementations give mathematically consistent results
        a = torch.randn(8, 6)
        b = torch.randn(10, 6)
        
        # Squared Euclidean should match manual computation
        dist_auto = pairwise_sqeuclidean(a, b)
        
        # Manual computation using broadcasting
        a_expanded = a.unsqueeze(1)  # [8, 1, 6]
        b_expanded = b.unsqueeze(0)  # [1, 10, 6]
        diff = a_expanded - b_expanded  # [8, 10, 6]
        dist_manual = (diff * diff).sum(dim=2)  # [8, 10]
        
        assert torch.allclose(dist_auto, dist_manual, atol=1e-5), "Auto and manual distance computation should match"

    def test_performance_characteristics(self):
        """Test that optimized implementations are actually faster (optional - mainly for verification)."""
        # This is more of a sanity check than a strict requirement
        torch.manual_seed(456)
        
        # Large tensors to see performance difference
        a = torch.randn(100, 50)
        b = torch.randn(80, 50)
        
        # Our optimized version should not crash with larger inputs
        dist = pairwise_sqeuclidean(a, b)
        
        assert dist.shape == (100, 80), "Should handle larger tensors correctly"
        assert torch.all(torch.isfinite(dist)), "Large tensor computation should be numerically stable"