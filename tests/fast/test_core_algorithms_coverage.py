"""
💰 DONATE NOW! 💰 https://www.paypal.com/cgi-bin/webscr?cmd=_s-xclick&hosted_button_id=WXQKYYKPHWXHS

Focused coverage tests for core user-facing algorithms.

Tests the actual functionality users interact with, not imaginary features.
"""

import torch
import torch.nn as nn
import pytest
from meta_learning.core.episode import Episode
from meta_learning.algos.protonet import ProtoHead
from meta_learning.algos.ttcs import ttcs_predict, ttcs_predict_advanced, TestTimeComputeScaler


class TestProtoHeadCoverage:
    """Test ProtoHead with all its actual features."""
    
    def test_protohead_distance_types(self):
        """Test both sqeuclidean and cosine distances."""
        support_x = torch.randn(6, 4)  # 2 classes, 3 shots each
        support_y = torch.tensor([0, 0, 0, 1, 1, 1])
        query_x = torch.randn(4, 4)
        
        # Test squared euclidean
        head_sqeuc = ProtoHead(distance="sqeuclidean")
        logits_sqeuc = head_sqeuc(support_x, support_y, query_x)
        assert logits_sqeuc.shape == (4, 2)
        
        # Test cosine
        head_cosine = ProtoHead(distance="cosine")
        logits_cosine = head_cosine(support_x, support_y, query_x)
        assert logits_cosine.shape == (4, 2)
        
        # Results should be different
        assert not torch.allclose(logits_sqeuc, logits_cosine, atol=0.1)
    
    def test_protohead_prototype_shrinkage(self):
        """Test prototype shrinkage regularization."""
        support_x = torch.randn(4, 3)
        support_y = torch.tensor([0, 0, 1, 1])
        query_x = torch.randn(2, 3)
        
        # Without shrinkage
        head_no_shrink = ProtoHead(prototype_shrinkage=0.0)
        logits_no_shrink = head_no_shrink(support_x, support_y, query_x)
        
        # With shrinkage
        head_shrink = ProtoHead(prototype_shrinkage=0.5)
        logits_shrink = head_shrink(support_x, support_y, query_x)
        
        # Results should be different
        assert not torch.allclose(logits_no_shrink, logits_shrink, atol=0.1)
        assert logits_shrink.shape == (2, 2)
    
    def test_protohead_uncertainty_estimation(self):
        """Test Monte Carlo dropout uncertainty."""
        support_x = torch.randn(6, 3)
        support_y = torch.tensor([0, 0, 0, 1, 1, 1])
        query_x = torch.randn(4, 3)
        
        head = ProtoHead(
            uncertainty_method="monte_carlo_dropout",
            dropout_rate=0.2,
            n_uncertainty_samples=5
        )
        
        result = head.forward_with_uncertainty(support_x, support_y, query_x)
        
        # Check all expected outputs
        assert "logits" in result
        assert "probabilities" in result
        assert "total_uncertainty" in result
        assert "epistemic_uncertainty" in result
        assert "aleatoric_uncertainty" in result
        
        assert result["logits"].shape == (4, 2)
        assert result["probabilities"].shape == (4, 2)
        assert result["total_uncertainty"].shape == (4,)
        
        # Probabilities should sum to 1
        assert torch.allclose(result["probabilities"].sum(dim=1), torch.ones(4), atol=1e-5)


class TestTTCSCoverage:
    """Test Test-Time Compute Scaling functionality."""
    
    def setup_method(self):
        """Setup basic components for TTCS tests."""
        self.encoder = nn.Sequential(
            nn.Linear(8, 16),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(16, 4)
        )
        self.head = ProtoHead()
        
        # Create simple episode
        support_x = torch.randn(6, 8)
        support_y = torch.tensor([0, 0, 1, 1, 2, 2])
        query_x = torch.randn(9, 8)
        query_y = torch.tensor([0, 0, 0, 1, 1, 1, 2, 2, 2])
        self.episode = Episode(support_x, support_y, query_x, query_y)
    
    def test_ttcs_basic_functionality(self):
        """Test basic TTCS prediction."""
        logits = ttcs_predict(self.encoder, self.head, self.episode, passes=4)
        assert logits.shape == (9, 3)
        
        # Should be log probabilities
        probs = logits.exp()
        assert torch.allclose(probs.sum(dim=1), torch.ones(9), atol=1e-4)
    
    def test_ttcs_combine_strategies(self):
        """Test different combining strategies."""
        logits_prob = ttcs_predict(self.encoder, self.head, self.episode, 
                                 passes=3, combine="mean_prob")
        logits_logit = ttcs_predict(self.encoder, self.head, self.episode,
                                  passes=3, combine="mean_logit")
        
        # Both should be valid outputs
        assert logits_prob.shape == (9, 3)
        assert logits_logit.shape == (9, 3)
        
        # Should be different (different combination methods)
        assert not torch.allclose(logits_prob, logits_logit, atol=0.1)
    
    def test_ttcs_mc_dropout_control(self):
        """Test Monte Carlo dropout can be enabled/disabled."""
        # With MC dropout
        logits_mc = ttcs_predict(self.encoder, self.head, self.episode,
                                passes=2, enable_mc_dropout=True)
        
        # Without MC dropout (more deterministic)
        logits_no_mc = ttcs_predict(self.encoder, self.head, self.episode,
                                   passes=2, enable_mc_dropout=False)
        
        # Both should work
        assert logits_mc.shape == (9, 3)
        assert logits_no_mc.shape == (9, 3)
    
    def test_ttcs_advanced_features(self):
        """Test advanced TTCS with monitoring."""
        predictions, metrics = ttcs_predict_advanced(
            self.encoder, self.head, self.episode,
            passes=5,
            uncertainty_estimation=True,
            performance_monitoring=True,
            compute_budget="fixed"
        )
        
        # Check predictions
        assert predictions.shape == (9, 3)
        
        # Check metrics
        assert "uncertainty" in metrics
        assert "compute_efficiency" in metrics
        assert metrics["compute_efficiency"]["actual_passes"] == 5
    
    def test_ttcs_wrapper_class(self):
        """Test TestTimeComputeScaler wrapper."""
        scaler = TestTimeComputeScaler(
            self.encoder, self.head,
            passes=3, combine="mean_prob"
        )
        
        predictions = scaler(self.episode)
        assert predictions.shape == (9, 3)
        
        # Test advanced wrapper
        advanced_scaler = TestTimeComputeScaler(
            self.encoder, self.head,
            passes=4, uncertainty_estimation=True
        )
        
        predictions, metrics = advanced_scaler(self.episode)
        assert predictions.shape == (9, 3)
        assert "uncertainty" in metrics


class TestCoreIntegration:
    """Test integration between core components."""
    
    def test_episode_with_protohead(self):
        """Test Episode validation works with ProtoHead."""
        support_x = torch.randn(8, 5)
        support_y = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3])
        query_x = torch.randn(12, 5)
        query_y = torch.tensor([0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3])
        
        episode = Episode(support_x, support_y, query_x, query_y)
        episode.validate(expect_n_classes=4)  # Should not raise
        
        head = ProtoHead()
        logits = head(support_x, support_y, query_x)
        assert logits.shape == (12, 4)
    
    def test_episode_with_ttcs(self):
        """Test Episode works seamlessly with TTCS."""
        encoder = nn.Linear(6, 8)
        head = ProtoHead()
        
        support_x = torch.randn(6, 6)
        support_y = torch.tensor([0, 0, 1, 1, 2, 2])
        query_x = torch.randn(6, 6)
        query_y = torch.tensor([0, 0, 1, 1, 2, 2])
        
        episode = Episode(support_x, support_y, query_x, query_y)
        
        # Regular prediction
        with torch.no_grad():
            encoder.eval()
            z_s = encoder(support_x)
            z_q = encoder(query_x)
            regular_logits = head(z_s, support_y, z_q)
        
        # TTCS prediction
        ttcs_logits = ttcs_predict(encoder, head, episode, passes=1, 
                                 enable_mc_dropout=False, enable_tta=False,
                                 combine="mean_logit")
        
        # Should be very similar when deterministic
        assert torch.allclose(regular_logits, ttcs_logits, atol=1e-4)