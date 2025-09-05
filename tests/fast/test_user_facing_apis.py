"""
💰 DONATE NOW! 💰 https://www.paypal.com/cgi-bin/webscr?cmd=_s-xclick&hosted_button_id=WXQKYYKPHWXHS

Tests for the APIs users interact with most: evaluate() and Conv4.

These are the first things users call, so they need comprehensive coverage.
"""

import torch
import torch.nn as nn
import pytest
import tempfile
import os
import json
from unittest.mock import MagicMock

from meta_learning.eval import evaluate, _get_t_critical
from meta_learning.models.conv4 import Conv4
from meta_learning.core.episode import Episode


class TestEvaluateFunction:
    """Test the main evaluate() function users call."""
    
    def setup_method(self):
        """Create mock episodes for testing."""
        self.episodes = []
        
        # Create 5 mock episodes with known outcomes
        for i in range(5):
            support_x = torch.randn(6, 4)
            support_y = torch.tensor([0, 0, 1, 1, 2, 2])
            query_x = torch.randn(9, 4)
            # Make some queries correct for predictable accuracy
            query_y = torch.tensor([0, 0, 0, 1, 1, 1, 2, 2, 2])
            
            episode = Episode(support_x, support_y, query_x, query_y)
            self.episodes.append(episode)
    
    def test_evaluate_basic_functionality(self):
        """Test basic evaluation with known accuracy."""
        episode_count = 0
        def mock_run_logits(episode):
            nonlocal episode_count
            # Create different accuracies for different episodes to get variance
            logits = torch.zeros(len(episode.query_y), 3)
            correct_ratio = 0.5 + 0.1 * episode_count  # 0.5, 0.6, 0.7, 0.8, 0.9
            n_correct = int(correct_ratio * len(episode.query_y))
            
            for i, true_label in enumerate(episode.query_y):
                if i < n_correct:
                    logits[i, true_label] = 1.0  # Correct prediction
                else:
                    logits[i, (true_label + 1) % 3] = 1.0  # Wrong prediction
            episode_count += 1
            return logits
        
        results = evaluate(mock_run_logits, self.episodes)
        
        # Check all expected fields
        assert "episodes" in results
        assert "mean_acc" in results  
        assert "ci95" in results
        assert "std_err" in results
        assert "elapsed_s" in results
        
        # Check values
        assert results["episodes"] == 5
        assert 0.6 < results["mean_acc"] < 0.8  # Should be around 0.7 with variance
        assert results["ci95"] > 0  # Should have non-zero variance now
        assert results["std_err"] > 0
        assert results["elapsed_s"] > 0
    
    def test_evaluate_perfect_accuracy(self):
        """Test evaluation with 100% accuracy."""
        def perfect_run_logits(episode):
            # Return logits that always predict correctly
            logits = torch.zeros(len(episode.query_y), 3)
            for i, true_label in enumerate(episode.query_y):
                logits[i, true_label] = 1.0
            return logits
        
        results = evaluate(perfect_run_logits, self.episodes)
        
        assert results["mean_acc"] == 1.0
        assert results["ci95"] == 0.0  # No variance = no CI
        assert results["std_err"] == 0.0
    
    def test_evaluate_single_episode(self):
        """Test evaluation with single episode (edge case)."""
        def run_logits(episode):
            logits = torch.zeros(len(episode.query_y), 3)
            for i, true_label in enumerate(episode.query_y):
                logits[i, true_label] = 1.0
            return logits
        
        results = evaluate(run_logits, [self.episodes[0]])
        
        assert results["episodes"] == 1
        assert results["ci95"] == 0.0  # Single sample = no CI
        assert results["std_err"] == 0.0
    
    def test_evaluate_file_output(self):
        """Test evaluation with file output."""
        def run_logits(episode):
            logits = torch.ones(len(episode.query_y), 3)
            return logits
        
        with tempfile.TemporaryDirectory() as tmpdir:
            results = evaluate(run_logits, self.episodes[:2], 
                             outdir=tmpdir, dump_preds=True)
            
            # Check files were created
            assert os.path.exists(os.path.join(tmpdir, "metrics.json"))
            assert os.path.exists(os.path.join(tmpdir, "preds.jsonl"))
            
            # Check metrics file content
            with open(os.path.join(tmpdir, "metrics.json")) as f:
                saved_metrics = json.load(f)
                assert saved_metrics["episodes"] == results["episodes"]
                assert saved_metrics["mean_acc"] == results["mean_acc"]
            
            # Check predictions file
            with open(os.path.join(tmpdir, "preds.jsonl")) as f:
                lines = f.readlines()
                assert len(lines) == 2  # Two episodes
                
                pred_data = json.loads(lines[0])
                assert "pred" in pred_data
                assert "y" in pred_data


class TestTCriticalFunction:
    """Test the statistical t-critical value lookup."""
    
    def test_t_critical_table_values(self):
        """Test known t-critical values from statistical tables."""
        # Test some key values
        assert _get_t_critical(1) == 12.71  # t(1) for 95% CI
        assert _get_t_critical(9) == 2.26   # t(9) for 95% CI
        assert _get_t_critical(29) == 2.05  # t(29) for 95% CI
    
    def test_t_critical_large_samples(self):
        """Test normal approximation for large samples."""
        # For df >= 30, should use normal approximation
        assert _get_t_critical(30) == 1.96
        assert _get_t_critical(100) == 1.96
        assert _get_t_critical(1000) == 1.96
    
    def test_t_critical_interpolation(self):
        """Test interpolation for missing values."""
        # df=3 IS in table (3.18), test actual interpolation case
        result = _get_t_critical(0)  # df=0 not in table, should use interpolation
        assert 2.0 < result < 5.0  # Should be reasonable
        
        # Test another interpolation case
        result2 = _get_t_critical(31)  # > 30, should be 1.96
        assert result2 == 1.96


class TestConv4Model:
    """Test the Conv4 backbone model."""
    
    def test_conv4_creation_default(self):
        """Test Conv4 creation with default parameters."""
        model = Conv4()
        
        # Check it's a nn.Module
        assert isinstance(model, nn.Module)
        
        # Test forward pass with standard input
        x = torch.randn(2, 3, 84, 84)  # Batch=2, RGB, 84x84
        output = model(x)
        
        assert output.shape == (2, 64)  # Default out_dim=64
        assert torch.isfinite(output).all()
    
    def test_conv4_custom_output_dim(self):
        """Test Conv4 with custom output dimension."""
        model = Conv4(out_dim=128)
        
        x = torch.randn(3, 3, 32, 32)
        output = model(x)
        
        assert output.shape == (3, 128)
        
        # Check projection layer was created
        assert not isinstance(model.projection, nn.Identity)
        assert isinstance(model.projection, nn.Linear)
    
    def test_conv4_with_dropout(self):
        """Test Conv4 with dropout enabled."""
        model = Conv4(p_drop=0.5)
        
        # Check dropout layer was created
        assert isinstance(model.dropout, nn.Dropout)
        assert model.dropout.p == 0.5
        
        x = torch.randn(2, 3, 28, 28)
        
        # In training mode, dropout should affect output
        model.train()
        out1 = model(x)
        out2 = model(x)
        
        # Very unlikely to be identical with dropout
        assert not torch.equal(out1, out2)
        
        # In eval mode, should be deterministic
        model.eval()
        out1 = model(x)
        out2 = model(x)
        assert torch.equal(out1, out2)
    
    def test_conv4_different_input_channels(self):
        """Test Conv4 with different number of input channels."""
        # Grayscale
        model_gray = Conv4(input_channels=1)
        x_gray = torch.randn(2, 1, 32, 32)
        output_gray = model_gray(x_gray)
        assert output_gray.shape == (2, 64)
        
        # Multi-spectral
        model_multi = Conv4(input_channels=10)
        x_multi = torch.randn(2, 10, 32, 32)
        output_multi = model_multi(x_multi)
        assert output_multi.shape == (2, 64)
    
    def test_conv4_variable_input_sizes(self):
        """Test Conv4 handles different input image sizes."""
        model = Conv4()
        
        # Different image sizes (adaptive pooling should handle this)
        sizes = [(32, 32), (64, 64), (84, 84), (128, 128)]
        
        for h, w in sizes:
            x = torch.randn(1, 3, h, w)
            output = model(x)
            assert output.shape == (1, 64), f"Failed for size {h}x{w}"
    
    def test_conv4_architecture_structure(self):
        """Test Conv4 has expected architectural components."""
        model = Conv4()
        
        # Should have features, dropout, flatten, projection
        assert hasattr(model, 'features')
        assert hasattr(model, 'dropout')
        assert hasattr(model, 'flatten')
        assert hasattr(model, 'projection')
        
        # Features should be Sequential with conv blocks
        assert isinstance(model.features, nn.Sequential)
        
        # Count conv layers (should be 4)
        conv_count = sum(1 for m in model.features.modules() 
                        if isinstance(m, nn.Conv2d))
        assert conv_count == 4
        
        # Count BatchNorm layers (should be 4)
        bn_count = sum(1 for m in model.features.modules()
                      if isinstance(m, nn.BatchNorm2d))
        assert bn_count == 4


class TestUserWorkflowIntegration:
    """Test integration of user-facing APIs in realistic workflows."""
    
    def test_conv4_with_evaluate(self):
        """Test typical user workflow: Conv4 + ProtoHead + evaluate."""
        from meta_learning.algos.protonet import ProtoHead
        
        # User's typical setup
        encoder = Conv4(out_dim=32)  # Custom output dim
        head = ProtoHead()
        
        # Create episodes
        episodes = []
        for _ in range(3):
            support_x = torch.randn(4, 3, 32, 32)  # 2-way 2-shot
            support_y = torch.tensor([0, 0, 1, 1])
            query_x = torch.randn(6, 3, 32, 32)
            query_y = torch.tensor([0, 0, 0, 1, 1, 1])
            episodes.append(Episode(support_x, support_y, query_x, query_y))
        
        # User's evaluation function
        def run_logits(episode):
            with torch.no_grad():
                encoder.eval()
                head.eval()
                z_s = encoder(episode.support_x)
                z_q = encoder(episode.query_x)
                return head(z_s, episode.support_y, z_q)
        
        # Should work without errors
        results = evaluate(run_logits, episodes)
        
        assert results["episodes"] == 3
        assert 0.0 <= results["mean_acc"] <= 1.0
        assert results["ci95"] >= 0
    
    def test_evaluation_statistical_correctness(self):
        """Test that evaluation provides statistically sound results."""
        # Create scenario with some variance
        episode_idx = 0
        def deterministic_70_percent(episode):
            nonlocal episode_idx
            logits = torch.zeros(len(episode.query_y), 3)
            # Make between 65-75% correct for variance
            accuracy = 0.65 + 0.1 * (episode_idx % 2)  # Alternates between 0.65 and 0.75
            n_correct = int(accuracy * len(episode.query_y))
            for i, true_label in enumerate(episode.query_y):
                if i < n_correct:
                    logits[i, true_label] = 1.0
                else:
                    logits[i, (true_label + 1) % 3] = 1.0
            episode_idx += 1
            return logits
        
        # Create enough episodes for statistics
        episodes = []
        for _ in range(20):
            query_y = torch.tensor([0, 1, 2] * 10)  # 30 queries per episode
            episodes.append(Episode(
                support_x=torch.randn(6, 4),
                support_y=torch.tensor([0, 0, 1, 1, 2, 2]),
                query_x=torch.randn(30, 4),
                query_y=query_y
            ))
        
        results = evaluate(deterministic_70_percent, episodes)
        
        # Should be very close to 70%
        assert abs(results["mean_acc"] - 0.7) < 0.05
        
        # Should have reasonable confidence interval for this sample size
        assert 0 < results["ci95"] < 0.2
        assert results["std_err"] > 0