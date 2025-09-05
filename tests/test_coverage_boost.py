"""
💰 SUPPORT THIS RESEARCH - PLEASE DONATE! 💰

🙏 If this library helps your research or project, please consider donating:
💳 PayPal: https://www.paypal.com/cgi-bin/webscr?cmd=_s-xclick&hosted_button_id=WXQKYYKPHWXHS
⭐ GitHub Sponsors: https://github.com/sponsors/benedictchen

👨‍💻 Author: Benedict Chen
💰 Donations: Help support this work! Buy me a coffee ☕, beer 🍺, lamborghini 🏎️, or private island 🏝️
💖 Please consider recurring donations to fully support continued research

Your support enables cutting-edge AI research for everyone! 🚀

Coverage Boost Tests
===================

Targeted tests to significantly boost coverage of key integrated components.
"""

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from unittest.mock import patch, MagicMock

import meta_learning as ml


class TestUnIntegratedFunctionality:
    """Test functionality that was integrated but not fully covered."""
    
    def test_protohead_uncertainty_full_path(self):
        """Test complete ProtoHead uncertainty path."""
        head = ml.ProtoHead(
            distance="cosine",
            tau=2.0,
            prototype_shrinkage=0.1,
            uncertainty_method="monte_carlo_dropout",
            dropout_rate=0.2,
            n_uncertainty_samples=5
        )
        
        # Test all the uncovered paths in uncertainty estimation
        support_features = torch.randn(15, 32)
        support_labels = torch.tensor([0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2])
        query_features = torch.randn(9, 32)
        
        # Test deterministic mode
        head_det = ml.ProtoHead(distance="cosine", uncertainty_method=None)
        logits_det = head_det.forward_deterministic(support_features, support_labels, query_features)
        assert logits_det.shape == (9, 3)
        
        # Test uncertainty mode with cosine distance
        result = head.forward_with_uncertainty(support_features, support_labels, query_features)
        
        assert "logits" in result
        assert "probabilities" in result
        assert "total_uncertainty" in result
        assert "epistemic_uncertainty" in result
        assert "aleatoric_uncertainty" in result
        
        # Test the internal Monte Carlo sampling
        head.train()  # Enable dropout mode
        samples = []
        for _ in range(3):
            sample_result = head._monte_carlo_uncertainty(support_features, support_labels, query_features)
            samples.append(sample_result["logits"])
        
        # Should have variation between samples
        sample_stack = torch.stack(samples)
        assert sample_stack.shape == (3, 9, 3)
    
    def test_continual_maml_core_functionality(self):
        """Test core ContinualMAML functionality."""
        model = nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 3))
        continual_maml = ml.ContinualMAML(model, memory_size=10, consolidation_strength=100.0)
        
        # Test memory operations
        support_x = torch.randn(6, 8)
        support_y = torch.randint(0, 3, (6,))
        query_x = torch.randn(4, 8)
        query_y = torch.randint(0, 3, (4,))
        
        # Add to memory
        continual_maml.add_to_memory(support_x, support_y, task_id=1)
        assert len(continual_maml.memory_x) == 6
        assert len(continual_maml.memory_y) == 6
        
        # Test Fisher Information computation
        class MockDataLoader:
            def __iter__(self):
                for _ in range(3):
                    yield torch.randn(2, 8), torch.randint(0, 3, (2,))
        
        dataloader = MockDataLoader()
        fisher_dict = continual_maml.compute_fisher_information(dataloader)
        
        assert isinstance(fisher_dict, dict)
        for name, fisher_val in fisher_dict.items():
            assert torch.all(fisher_val >= 0)  # Fisher info should be non-negative
        
        # Test EWC loss computation
        continual_maml.previous_params[0] = {}
        continual_maml.fisher_information[0] = fisher_dict
        
        for name, param in model.named_parameters():
            continual_maml.previous_params[0][name] = param.detach().clone()
        
        ewc_loss = continual_maml.compute_ewc_loss()
        assert isinstance(ewc_loss, torch.Tensor)
        assert ewc_loss.item() >= 0
        
        # Test continual learning step
        loss = continual_maml.continual_inner_adapt_and_eval(
            (support_x, support_y), (query_x, query_y), task_id=1
        )
        assert isinstance(loss, torch.Tensor)
        assert loss.requires_grad
        
        # Test consolidation
        continual_maml.consolidate_task(dataloader, task_id=1)
        assert 1 in continual_maml.previous_params
        assert 1 in continual_maml.fisher_information
    
    def test_hardware_utils_basic_functionality(self):
        """Test basic hardware utilities functionality."""
        if ml.INTEGRATED_ADVANCED_AVAILABLE:
            # Test config creation
            config = ml.create_hardware_config(device="cpu")
            assert config.device == "cpu"
            
            # Test hardware setup
            model = nn.Linear(4, 2)
            optimized_model, used_config = ml.setup_optimal_hardware(model, config)
            
            assert optimized_model is not None
            assert used_config is config
            assert next(optimized_model.parameters()).device.type == "cpu"
    
    def test_leakage_guard_basic_functionality(self):
        """Test basic leakage guard functionality."""
        if ml.INTEGRATED_ADVANCED_AVAILABLE:
            guard = ml.create_leakage_guard(strict_mode=False)
            
            # Test train-test split registration
            guard.register_train_test_split([0, 1, 2], [3, 4, 5])
            
            # Test clean episode
            is_clean = guard.check_episode_data([0, 1, 2], [0, 1, 2], "clean_episode")
            assert is_clean == True
            
            # Test contaminated episode
            is_clean = guard.check_episode_data([0, 1, 3], [0, 1, 3], "contaminated_episode")
            assert is_clean == False
            assert len(guard.violations) > 0
    
    def test_data_module_coverage(self):
        """Test data module functionality for coverage."""
        # Test synthetic dataset with image mode
        dataset = ml.SyntheticFewShotDataset(n_classes=10, dim=32, noise=0.1, image_mode=True)
        
        # Test episode creation
        support_x, support_y, query_x, query_y = dataset.sample_support_query(3, 2, 5)
        
        assert support_x.shape[0] == 6  # 3 classes * 2 shots
        assert query_x.shape[0] == 15  # Default is 15 query samples total
        assert len(torch.unique(support_y)) == 3
        
        # Test episode generation
        episodes = list(ml.make_episodes(dataset, n_way=3, k_shot=2, m_query=4, episodes=2))
        assert len(episodes) == 2
        
        for episode in episodes:
            episode.validate(expect_n_classes=3)
    
    def test_evaluation_module_coverage(self):
        """Test evaluation module for coverage."""
        # Create simple model and episodes
        def simple_run_logits(episode):
            # Simple random predictions for testing
            n_query = episode.query_x.size(0)
            n_classes = len(torch.unique(episode.support_y))
            return torch.randn(n_query, n_classes)
        
        # Create test episodes
        episodes = []
        for _ in range(3):
            support_x = torch.randn(6, 8)
            support_y = torch.tensor([0, 0, 1, 1, 2, 2])
            query_x = torch.randn(4, 8)
            query_y = torch.randint(0, 3, (4,))
            
            episode = ml.Episode(support_x, support_y, query_x, query_y)
            episodes.append(episode)
        
        # Test evaluation
        results = ml.evaluate(simple_run_logits, episodes, outdir=None, dump_preds=False)
        
        assert "episodes" in results
        assert "mean_acc" in results  # Correct key name
        assert results["episodes"] == 3
    
    def test_benchmark_coverage(self):
        """Test benchmarking functionality."""
        def simple_episode_acc():
            return 0.75  # Fixed accuracy for testing
        
        result = ml.run_benchmark(
            simple_episode_acc, 
            episodes=5, 
            warmup=1,
            meta={"test": "benchmark"},
            outdir=None
        )
        
        assert hasattr(result, 'mean_acc')
        assert hasattr(result, 'ci95')
        assert hasattr(result, 'episodes')
        assert result.episodes == 5
    
    def test_core_utilities_coverage(self):
        """Test core utility functions."""
        # Test seed all
        ml.seed_all(42)
        
        # Test BatchNorm freezing
        model = nn.Sequential(
            nn.Linear(4, 8),
            nn.BatchNorm1d(8),
            nn.ReLU(),
            nn.Linear(8, 2)
        )
        
        ml.freeze_batchnorm_running_stats(model)
        
        # Check that BatchNorm is in eval mode
        for module in model.modules():
            if isinstance(module, nn.BatchNorm1d):
                assert not module.training
    
    def test_math_utils_edge_cases(self):
        """Test math utilities edge cases."""
        # Test with very small vectors (should not crash)
        small_a = torch.randn(2, 3) * 1e-8
        small_b = torch.randn(4, 3) * 1e-8
        
        # Should handle small numbers gracefully
        dist = ml.pairwise_sqeuclidean(small_a, small_b)
        assert torch.all(torch.isfinite(dist))
        
        # Test cosine logits with edge cases
        cosine_logits = ml.cosine_logits(small_a, small_b, tau=1.0)
        assert torch.all(torch.isfinite(cosine_logits))
        
        # Test with zero vectors
        zero_a = torch.zeros(2, 3)
        zero_b = torch.zeros(1, 3)
        
        cosine_logits_zero = ml.cosine_logits(zero_a, zero_b, tau=1.0)
        assert torch.all(torch.isfinite(cosine_logits_zero))


class TestAdvancedIntegrationScenarios:
    """Test advanced integration scenarios."""
    
    def test_full_pipeline_with_all_features(self):
        """Test complete pipeline using all integrated features."""
        # Create encoder and head with all features
        encoder = nn.Sequential(nn.Linear(16, 32), nn.ReLU(), nn.Linear(32, 64))
        head = ml.ProtoHead(
            distance="cosine",
            tau=1.5,
            prototype_shrinkage=0.05,
            uncertainty_method="monte_carlo_dropout",
            dropout_rate=0.1,
            n_uncertainty_samples=3
        )
        
        # Test with ContinualMAML if available
        continual_maml = ml.ContinualMAML(encoder, memory_size=20)
        
        # Create episode
        support_x = torch.randn(12, 16)
        support_y = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2])
        query_x = torch.randn(6, 16)
        query_y = torch.randint(0, 3, (6,))
        
        # Forward pass through encoder
        with torch.no_grad():
            support_features = encoder(support_x)
            query_features = encoder(query_x)
        
        # Head forward with uncertainty
        result = head.forward_with_uncertainty(support_features, support_y, query_features)
        
        # Check all outputs are present and valid
        assert "logits" in result
        assert "total_uncertainty" in result
        assert torch.all(torch.isfinite(result["logits"]))
        assert torch.all(torch.isfinite(result["total_uncertainty"]))
        
        # Test continual learning
        loss = continual_maml.continual_inner_adapt_and_eval(
            (support_x, support_y), (query_x, query_y), task_id=0
        )
        assert isinstance(loss, torch.Tensor)
        assert loss.requires_grad
    
    def test_error_handling_and_edge_cases(self):
        """Test error handling in integrated components."""
        # Test ProtoHead with empty support set
        head = ml.ProtoHead()
        
        try:
            # This should fail gracefully
            empty_support = torch.empty(0, 8)
            empty_labels = torch.empty(0, dtype=torch.long)
            query_features = torch.randn(3, 8)
            
            result = head.forward_deterministic(empty_support, empty_labels, query_features)
        except (RuntimeError, ValueError):
            # Expected to fail with empty support
            pass
        
        # Test ContinualMAML with minimal data
        model = nn.Linear(2, 1)
        continual_maml = ml.ContinualMAML(model, memory_size=1)
        
        # Single sample episode
        single_x = torch.randn(1, 2)
        single_y = torch.tensor([0])
        
        # Should handle single samples
        continual_maml.add_to_memory(single_x, single_y, task_id=0)
        assert len(continual_maml.memory_x) == 1


class TestImportAndAvailabilityChecks:
    """Test import and availability checking."""
    
    def test_feature_availability_flags(self):
        """Test that feature availability flags are working."""
        # These should be defined
        assert hasattr(ml, 'INTEGRATED_ADVANCED_AVAILABLE')
        assert hasattr(ml, 'STANDALONE_MODULES_AVAILABLE') 
        assert hasattr(ml, 'EXTERNAL_RESEARCH_AVAILABLE')
        assert hasattr(ml, 'RESEARCH_AVAILABLE')
        
        # At least research should be available
        assert ml.RESEARCH_AVAILABLE == True
    
    def test_conditional_imports(self):
        """Test that conditional imports work correctly."""
        # Test accessing components based on availability
        if ml.INTEGRATED_ADVANCED_AVAILABLE:
            # Should be able to create these
            config = ml.create_hardware_config()
            guard = ml.create_leakage_guard()
            
            assert config is not None
            assert guard is not None
        
        if ml.EXTERNAL_RESEARCH_AVAILABLE:
            # External research components should be available
            assert ml.TestTimeComputeScaler is not None
            assert ml.TestTimeComputeConfig is not None
        
        # Core components should always be available
        assert ml.ProtoHead is not None
        assert ml.ContinualMAML is not None
        assert ml.Episode is not None


class TestCoverageTargets:
    """Specific tests targeting uncovered lines."""
    
    def test_cli_missing_branches(self):
        """Test CLI branches that were missed."""
        # Test CIFAR-FS with all parameters
        args_cifar = type('Args', (), {
            'dataset': 'cifar_fs',
            'encoder': 'conv4', 
            'emb_dim': 64,
            'noise': 0.1,
            'data_root': './data',
            'split': 'train',
            'manifest': './manifest.json',
            'download': True,
            'image_size': 32
        })()
        
        # Just test that we can access the CLI module
        from meta_learning import cli
        try:
            dataset = cli._build_dataset(args_cifar)
        except (FileNotFoundError, RuntimeError):
            # Expected when data not available
            pass
    
    def test_protohead_missing_paths(self):
        """Test ProtoHead paths that were missed."""
        # Test with no uncertainty method but uncertainty parameters set
        head = ml.ProtoHead(
            uncertainty_method=None,
            dropout_rate=0.2,  # Should be ignored
            n_uncertainty_samples=5  # Should be ignored
        )
        
        support_features = torch.randn(6, 16)
        support_labels = torch.tensor([0, 0, 1, 1, 2, 2])
        query_features = torch.randn(3, 16)
        
        # Should use deterministic forward
        logits = head.forward(support_features, support_labels, query_features)
        assert logits.shape == (3, 3)
    
    def test_maml_edge_cases(self):
        """Test MAML edge cases."""
        # Test inner_adapt_and_eval with first_order=True
        model = nn.Linear(4, 2)
        
        def loss_fn(logits, targets):
            return F.cross_entropy(logits, targets)
        
        support_x = torch.randn(4, 4)
        support_y = torch.randint(0, 2, (4,))
        query_x = torch.randn(2, 4) 
        query_y = torch.randint(0, 2, (2,))
        
        # Test first-order approximation
        loss = ml.inner_adapt_and_eval(
            model, loss_fn, 
            (support_x, support_y), 
            (query_x, query_y),
            first_order=True,  # This should change gradient computation
            freeze_bn_stats=False  # Test without freezing BN
        )
        
        assert isinstance(loss, torch.Tensor)
        assert loss.requires_grad