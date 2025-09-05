"""Comprehensive Evaluation Tests for maximum eval.py coverage."""

import pytest
import torch
import tempfile
import os
import json
import meta_learning as ml
from meta_learning.data import SyntheticFewShotDataset, make_episodes
from meta_learning.eval import evaluate, _get_t_critical


class TestEvaluateFunction:
    """Comprehensive tests for the evaluate function."""
    
    def test_basic_evaluation(self):
        """Test basic evaluation functionality."""
        dataset = SyntheticFewShotDataset(n_classes=5, dim=16, noise=0.1)
        episodes = list(make_episodes(dataset, n_way=3, k_shot=2, m_query=4, episodes=5))
        
        def simple_logits_fn(episode):
            return torch.randn(episode.query_x.shape[0], 3)
        
        result = evaluate(simple_logits_fn, episodes)
        
        assert "episodes" in result
        assert "mean_acc" in result
        assert "ci95" in result
        assert "std_err" in result
        assert "elapsed_s" in result
        assert result["episodes"] == 5
        assert 0 <= result["mean_acc"] <= 1
    
    def test_evaluation_with_files(self):
        """Test evaluation with output files."""
        dataset = SyntheticFewShotDataset(n_classes=4, dim=12, noise=0.05)
        episodes = list(make_episodes(dataset, n_way=2, k_shot=3, m_query=3, episodes=8))
        
        def deterministic_logits_fn(episode):
            logits = torch.zeros(episode.query_x.shape[0], 2)
            logits[:, 0] = 2.0
            logits[:, 1] = -1.0
            return logits
        
        with tempfile.TemporaryDirectory() as temp_dir:
            result = evaluate(deterministic_logits_fn, episodes, outdir=temp_dir, dump_preds=True)
            
            metrics_file = os.path.join(temp_dir, "metrics.json")
            preds_file = os.path.join(temp_dir, "preds.jsonl")
            
            assert os.path.exists(metrics_file)
            assert os.path.exists(preds_file)
            
            with open(metrics_file, 'r') as f:
                saved_metrics = json.load(f)
            assert saved_metrics["episodes"] == result["episodes"]
    
    def test_statistical_calculations_edge_cases(self):
        """Test statistical calculations with various sample sizes.""" 
        dataset = SyntheticFewShotDataset(n_classes=4, dim=16, noise=0.1)
        
        # Single episode case
        single_episode = list(make_episodes(dataset, n_way=2, k_shot=2, m_query=3, episodes=1))
        
        def random_logits_fn(episode):
            return torch.randn(episode.query_x.shape[0], 2)
        
        result_single = evaluate(random_logits_fn, single_episode)
        assert result_single["episodes"] == 1
        assert result_single["ci95"] == 0.0
        assert result_single["std_err"] == 0.0
        
        # Small sample size (n < 30)
        small_episodes = list(make_episodes(dataset, n_way=2, k_shot=1, m_query=2, episodes=15))
        result_small = evaluate(random_logits_fn, small_episodes)
        assert result_small["episodes"] == 15
        assert result_small["ci95"] > 0
        
        # Large sample size (n >= 30)
        large_episodes = list(make_episodes(dataset, n_way=2, k_shot=1, m_query=1, episodes=35))
        result_large = evaluate(random_logits_fn, large_episodes)
        assert result_large["episodes"] == 35
        assert result_large["ci95"] > 0


class TestTCriticalFunction:
    """Tests for the _get_t_critical helper function."""
    
    def test_known_t_values(self):
        """Test known t-critical values from the table."""
        assert _get_t_critical(1) == 12.71
        assert _get_t_critical(5) == 2.57
        assert _get_t_critical(10) == 2.23
        assert _get_t_critical(29) == 2.05
    
    def test_large_df_normal_approximation(self):
        """Test that large degrees of freedom use normal approximation."""
        assert _get_t_critical(30) == 1.96
        assert _get_t_critical(100) == 1.96
    
    def test_interpolation_for_missing_values(self):
        """Test interpolation for degrees of freedom not in table."""
        result = _get_t_critical(3)
        assert isinstance(result, float)
        assert 2.0 < result < 3.5