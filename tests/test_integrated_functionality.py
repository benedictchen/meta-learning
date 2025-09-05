"""
💰 SUPPORT THIS RESEARCH - PLEASE DONATE! 💰

🙏 If this library helps your research or project, please consider donating:
💳 PayPal: https://www.paypal.com/cgi-bin/webscr?cmd=_s-xclick&hosted_button_id=WXQKYYKPHWXHS
⭐ GitHub Sponsors: https://github.com/sponsors/benedictchen

👨‍💻 Author: Benedict Chen
💰 Donations: Help support this work! Buy me a coffee ☕, beer 🍺, lamborghini 🏎️, or private island 🏝️
💖 Please consider recurring donations to fully support continued research

Your support enables cutting-edge AI research for everyone! 🚀

Comprehensive Tests for Integrated Advanced Functionality
=========================================================

Tests all the newly integrated advanced features to ensure they work correctly
and are properly integrated with the existing codebase.
"""

import pytest
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, Any
import tempfile
import json
import subprocess
import sys

import meta_learning as ml


class TestContinualMAMLIntegration:
    """Test the integrated ContinualMAML functionality."""
    
    def test_continual_maml_creation(self):
        """Test ContinualMAML can be created and initialized."""
        model = nn.Sequential(
            nn.Linear(10, 20),
            nn.ReLU(),
            nn.Linear(20, 5)
        )
        
        continual_maml = ml.ContinualMAML(
            model=model,
            memory_size=100,
            consolidation_strength=1000.0,
            fisher_samples=50
        )
        
        assert continual_maml.memory_size == 100
        assert continual_maml.consolidation_strength == 1000.0
        assert continual_maml.fisher_samples == 50
        assert len(continual_maml.memory_x) == 0
        assert continual_maml.task_count == 0
    
    def test_continual_maml_memory_operations(self):
        """Test episodic memory functionality."""
        model = nn.Sequential(nn.Linear(10, 5))
        continual_maml = ml.ContinualMAML(model, memory_size=10)
        
        # Add examples to memory
        x = torch.randn(5, 10)
        y = torch.randint(0, 5, (5,))
        task_id = 1
        
        continual_maml.add_to_memory(x, y, task_id)
        
        assert len(continual_maml.memory_x) == 5
        assert len(continual_maml.memory_y) == 5
        assert len(continual_maml.memory_task_ids) == 5
        assert all(tid == task_id for tid in continual_maml.memory_task_ids)
    
    def test_continual_maml_ewc_computation(self):
        """Test EWC loss computation."""
        model = nn.Sequential(nn.Linear(10, 5))
        continual_maml = ml.ContinualMAML(model)
        
        # Initially no EWC loss
        ewc_loss = continual_maml.compute_ewc_loss()
        assert ewc_loss == 0.0
        
        # Simulate having previous task parameters
        continual_maml.previous_params[0] = {
            '0.weight': torch.randn(5, 10),
            '0.bias': torch.randn(5)
        }
        continual_maml.fisher_information[0] = {
            '0.weight': torch.ones(5, 10) * 0.1,
            '0.bias': torch.ones(5) * 0.1
        }
        
        ewc_loss = continual_maml.compute_ewc_loss()
        assert isinstance(ewc_loss, torch.Tensor)
        assert ewc_loss.item() > 0
    
    def test_continual_maml_forward_pass(self):
        """Test continual MAML forward pass."""
        model = nn.Sequential(
            nn.Linear(10, 20),
            nn.ReLU(),
            nn.Linear(20, 5)
        )
        continual_maml = ml.ContinualMAML(model, memory_size=50)
        
        # Create support and query data
        support_x = torch.randn(15, 10)  # 3-way 5-shot
        support_y = torch.tensor([0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2])
        query_x = torch.randn(9, 10)    # 3-way 3-query
        query_y = torch.randint(0, 3, (9,))
        
        # Test continual inner adaptation
        loss = continual_maml.continual_inner_adapt_and_eval(
            support=(support_x, support_y),
            query=(query_x, query_y),
            task_id=1,
            inner_lr=0.01
        )
        
        assert isinstance(loss, torch.Tensor)
        assert loss.requires_grad
        assert len(continual_maml.memory_x) == 15  # Support examples added to memory


class TestProtoHeadUncertaintyIntegration:
    """Test the integrated uncertainty estimation in ProtoHead."""
    
    def test_protohead_deterministic_mode(self):
        """Test ProtoHead works in deterministic mode (backward compatibility)."""
        head = ml.ProtoHead(distance="sqeuclidean", tau=1.0, prototype_shrinkage=0.1)
        
        # Create test data
        support_features = torch.randn(15, 64)  # 3-way 5-shot
        support_labels = torch.tensor([0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2])
        query_features = torch.randn(9, 64)     # 3 queries
        
        # Forward pass
        logits = head(support_features, support_labels, query_features)
        
        assert logits.shape == (9, 3)  # 9 queries, 3 classes
        assert isinstance(logits, torch.Tensor)
        assert head.uncertainty_method is None
    
    def test_protohead_uncertainty_mode(self):
        """Test ProtoHead with uncertainty estimation enabled."""
        head = ml.ProtoHead(
            distance="sqeuclidean", 
            tau=1.0,
            uncertainty_method="monte_carlo_dropout",
            dropout_rate=0.1,
            n_uncertainty_samples=5
        )
        
        # Create test data
        support_features = torch.randn(15, 64)
        support_labels = torch.tensor([0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2])
        query_features = torch.randn(9, 64)
        
        # Test uncertainty-aware forward pass
        results = head.forward_with_uncertainty(support_features, support_labels, query_features)
        
        assert "logits" in results
        assert "probabilities" in results
        assert "total_uncertainty" in results
        assert "epistemic_uncertainty" in results
        assert "aleatoric_uncertainty" in results
        
        assert results["logits"].shape == (9, 3)
        assert results["probabilities"].shape == (9, 3)
        assert results["total_uncertainty"].shape == (9,)
        assert results["n_samples"] == 5
        
        # Test probabilities sum to 1
        prob_sums = results["probabilities"].sum(dim=-1)
        assert torch.allclose(prob_sums, torch.ones(9), atol=1e-5)
    
    def test_protohead_prototype_shrinkage(self):
        """Test prototype shrinkage functionality."""
        head_no_shrinkage = ml.ProtoHead(prototype_shrinkage=0.0)
        head_with_shrinkage = ml.ProtoHead(prototype_shrinkage=0.3)
        
        # Same data for both
        support_features = torch.randn(15, 64)
        support_labels = torch.tensor([0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2])
        query_features = torch.randn(9, 64)
        
        logits_no_shrinkage = head_no_shrinkage(support_features, support_labels, query_features)
        logits_with_shrinkage = head_with_shrinkage(support_features, support_labels, query_features)
        
        # Results should be different due to prototype shrinkage
        assert not torch.allclose(logits_no_shrinkage, logits_with_shrinkage, atol=1e-5)
        assert logits_no_shrinkage.shape == logits_with_shrinkage.shape
    
    def test_protohead_cosine_distance_with_uncertainty(self):
        """Test ProtoHead with cosine distance and uncertainty."""
        head = ml.ProtoHead(
            distance="cosine",
            tau=10.0,
            uncertainty_method="monte_carlo_dropout",
            n_uncertainty_samples=3
        )
        
        support_features = torch.randn(10, 32)
        support_labels = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3, 4, 4])  # 5-way 2-shot
        query_features = torch.randn(5, 32)
        
        results = head.forward_with_uncertainty(support_features, support_labels, query_features)
        
        assert results["logits"].shape == (5, 5)  # 5 queries, 5 classes
        assert torch.all(torch.isfinite(results["logits"]))
        assert torch.all(results["total_uncertainty"] >= 0)


class TestHardwareIntegration:
    """Test integrated hardware acceleration functionality."""
    
    def test_hardware_config_creation(self):
        """Test hardware configuration can be created."""
        if ml.INTEGRATED_ADVANCED_AVAILABLE:
            config = ml.create_hardware_config()
            assert config.device in ["cuda", "mps", "cpu"]
            assert isinstance(config.use_mixed_precision, bool)
            assert isinstance(config.precision_dtype, str)
        else:
            pytest.skip("Hardware utils not available")
    
    def test_hardware_detector(self):
        """Test hardware detection functionality."""
        if ml.INTEGRATED_ADVANCED_AVAILABLE:
            detector = ml.HardwareDetector()
            device = detector.detect_best_device()
            assert isinstance(device, torch.device)
            
            hardware_info = detector.get_hardware_info()
            assert "platform" in hardware_info
            assert "cpu_cores" in hardware_info
            assert "memory_gb" in hardware_info
        else:
            pytest.skip("Hardware utils not available")
    
    def test_hardware_optimization(self):
        """Test model hardware optimization."""
        if ml.INTEGRATED_ADVANCED_AVAILABLE:
            model = nn.Sequential(
                nn.Linear(10, 20),
                nn.BatchNorm1d(20),
                nn.ReLU(),
                nn.Linear(20, 5)
            )
            
            config = ml.create_hardware_config()
            optimized_model, used_config = ml.setup_optimal_hardware(model, config)
            
            assert optimized_model is not None
            assert used_config.device == config.device
        else:
            pytest.skip("Hardware utils not available")


class TestLeakageDetectionIntegration:
    """Test integrated leakage detection functionality."""
    
    def test_leakage_guard_creation(self):
        """Test leakage guard can be created."""
        if ml.INTEGRATED_ADVANCED_AVAILABLE:
            guard = ml.create_leakage_guard(strict_mode=False)
            assert not guard.strict_mode
            assert len(guard.violations) == 0
        else:
            pytest.skip("Leakage guard not available")
    
    def test_leakage_guard_train_test_split(self):
        """Test train/test split registration."""
        if ml.INTEGRATED_ADVANCED_AVAILABLE:
            guard = ml.create_leakage_guard(strict_mode=False)
            
            train_classes = [0, 1, 2, 3, 4]
            test_classes = [5, 6, 7, 8, 9]
            
            guard.register_train_test_split(train_classes, test_classes)
            
            assert guard.train_classes == set(train_classes)
            assert guard.test_classes == set(test_classes)
        else:
            pytest.skip("Leakage guard not available")
    
    def test_leakage_detection(self):
        """Test leakage detection functionality."""
        if ml.INTEGRATED_ADVANCED_AVAILABLE:
            guard = ml.create_leakage_guard(strict_mode=False)
            guard.register_train_test_split([0, 1, 2], [3, 4, 5])
            
            # Clean episode (no leakage)
            clean = guard.check_episode_data([0, 1], [0, 1], "clean_episode")
            assert clean
            
            # Contaminated episode (leakage)
            contaminated = guard.check_episode_data([0, 3], [0, 3], "contaminated_episode")
            assert not contaminated
            
            # Check violations were recorded
            report = guard.get_violations_report()
            assert report["total_violations"] > 0
        else:
            pytest.skip("Leakage guard not available")


class TestCLIIntegration:
    """Test the enhanced CLI with integrated features."""
    
    def test_cli_basic_functionality_preserved(self):
        """Test that basic CLI functionality still works."""
        result = subprocess.run([
            sys.executable, "-m", "meta_learning.cli", "eval",
            "--episodes", "5", "--n-way", "3", "--k-shot", "2"
        ], env={"PYTHONPATH": "."}, capture_output=True, text=True)
        
        assert result.returncode == 0
        output = json.loads(result.stdout)
        assert "episodes" in output
        assert output["episodes"] == 5
    
    def test_cli_uncertainty_integration(self):
        """Test CLI with uncertainty estimation."""
        if ml.INTEGRATED_ADVANCED_AVAILABLE:
            result = subprocess.run([
                sys.executable, "-m", "meta_learning.cli", "eval",
                "--episodes", "3", 
                "--uncertainty", "monte_carlo_dropout",
                "--uncertainty-samples", "5"
            ], env={"PYTHONPATH": "."}, capture_output=True, text=True)
            
            assert result.returncode == 0
            output = json.loads(result.stdout)
            assert "episodes" in output
    
    def test_cli_prototype_shrinkage(self):
        """Test CLI with prototype shrinkage."""
        result = subprocess.run([
            sys.executable, "-m", "meta_learning.cli", "eval",
            "--episodes", "3",
            "--prototype-shrinkage", "0.1"
        ], env={"PYTHONPATH": "."}, capture_output=True, text=True)
        
        assert result.returncode == 0
        output = json.loads(result.stdout)
        assert "episodes" in output
    
    def test_cli_hardware_optimization(self):
        """Test CLI with hardware optimization."""
        if ml.INTEGRATED_ADVANCED_AVAILABLE:
            result = subprocess.run([
                sys.executable, "-m", "meta_learning.cli", "eval",
                "--episodes", "3",
                "--optimize-hardware"
            ], env={"PYTHONPATH": "."}, capture_output=True, text=True)
            
            assert result.returncode == 0
            assert "Hardware optimized" in result.stderr or "✓ Hardware optimized" in result.stdout


class TestIntegrationCombinations:
    """Test combinations of integrated features working together."""
    
    def test_uncertainty_with_prototype_shrinkage(self):
        """Test uncertainty estimation combined with prototype shrinkage."""
        head = ml.ProtoHead(
            distance="sqeuclidean",
            prototype_shrinkage=0.2,
            uncertainty_method="monte_carlo_dropout",
            n_uncertainty_samples=5
        )
        
        support_features = torch.randn(15, 32)
        support_labels = torch.tensor([0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2])
        query_features = torch.randn(6, 32)
        
        # Test combined functionality
        results = head.forward_with_uncertainty(support_features, support_labels, query_features)
        
        assert results["logits"].shape == (6, 3)
        assert torch.all(torch.isfinite(results["total_uncertainty"]))
        assert head.prototype_shrinkage == 0.2
    
    def test_continual_maml_with_hardware_optimization(self):
        """Test ContinualMAML with hardware optimization."""
        if ml.INTEGRATED_ADVANCED_AVAILABLE:
            model = nn.Sequential(nn.Linear(10, 20), nn.ReLU(), nn.Linear(20, 5))
            
            # Hardware optimization
            config = ml.create_hardware_config()
            optimized_model, _ = ml.setup_optimal_hardware(model, config)
            
            # Get the device the model was moved to
            device = next(optimized_model.parameters()).device
            
            # Continual MAML
            continual_maml = ml.ContinualMAML(optimized_model, memory_size=50)
            
            # Test they work together - ensure tensors are on correct device
            support_x = torch.randn(10, 10).to(device)
            support_y = torch.randint(0, 5, (10,)).to(device)
            query_x = torch.randn(5, 10).to(device)
            query_y = torch.randint(0, 5, (5,)).to(device)
            
            loss = continual_maml.continual_inner_adapt_and_eval(
                (support_x, support_y), (query_x, query_y), task_id=1
            )
            
            assert isinstance(loss, torch.Tensor)
            assert loss.requires_grad
    
    def test_full_pipeline_integration(self):
        """Test complete pipeline with all integrated features."""
        # Create model
        model = nn.Sequential(
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 32)
        )
        
        # Hardware optimization (if available)
        device = torch.device("cpu")  # Default to CPU for testing
        if ml.INTEGRATED_ADVANCED_AVAILABLE:
            config = ml.create_hardware_config()
            model, config_used = ml.setup_optimal_hardware(model, config)
            # Get the device the model was moved to
            device = next(model.parameters()).device
        
        # ProtoHead with uncertainty and shrinkage - move to same device as model
        head = ml.ProtoHead(
            distance="sqeuclidean",
            tau=1.0,
            prototype_shrinkage=0.1,
            uncertainty_method="monte_carlo_dropout",
            n_uncertainty_samples=3
        ).to(device)
        
        # Leakage detection (if available)
        leakage_guard = None
        if ml.INTEGRATED_ADVANCED_AVAILABLE:
            leakage_guard = ml.create_leakage_guard(strict_mode=False)
            leakage_guard.register_train_test_split([0, 1, 2], [3, 4])
        
        # Generate episode data on the correct device
        support_x = torch.randn(15, 32).to(device)
        support_y = torch.tensor([0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2]).to(device)
        query_x = torch.randn(9, 32).to(device)
        query_y = torch.randint(0, 3, (9,)).to(device)
        
        # Check for leakage
        if leakage_guard:
            is_clean = leakage_guard.check_episode_data(
                support_y.tolist(), query_y.tolist(), "test_episode"
            )
            # This should be clean since we're using classes 0,1,2 which are in train set
        
        # Extract features
        with torch.no_grad():
            support_features = model(support_x)
            query_features = model(query_x)
        
        # Forward pass with all enhancements
        if head.uncertainty_method:
            results = head.forward_with_uncertainty(support_features, support_y, query_features)
            logits = results["logits"]
            uncertainty = results["total_uncertainty"]
            
            assert logits.shape == (9, 3)
            assert uncertainty.shape == (9,)
        else:
            logits = head(support_features, support_y, query_features)
            assert logits.shape == (9, 3)
        
        # Compute accuracy
        predictions = logits.argmax(dim=-1)
        accuracy = (predictions == query_y).float().mean()
        
        assert 0 <= accuracy <= 1


class TestBackwardCompatibility:
    """Test that all integrations maintain backward compatibility."""
    
    def test_protohead_backward_compatibility(self):
        """Test ProtoHead without any new arguments works as before."""
        head = ml.ProtoHead()  # Default arguments only
        
        support_features = torch.randn(15, 64)
        support_labels = torch.tensor([0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2])
        query_features = torch.randn(9, 64)
        
        logits = head(support_features, support_labels, query_features)
        
        assert logits.shape == (9, 3)
        assert head.uncertainty_method is None
        assert head.prototype_shrinkage == 0.0
    
    def test_maml_functions_backward_compatibility(self):
        """Test original MAML functions still work."""
        model = nn.Sequential(nn.Linear(10, 20), nn.ReLU(), nn.Linear(20, 5))
        
        support_x = torch.randn(15, 10)
        support_y = torch.randint(0, 5, (15,))
        query_x = torch.randn(10, 10)
        query_y = torch.randint(0, 5, (10,))
        
        # Test original inner_adapt_and_eval
        loss = ml.inner_adapt_and_eval(
            model, nn.CrossEntropyLoss(),
            support=(support_x, support_y),
            query=(query_x, query_y),
            inner_lr=0.01
        )
        
        assert isinstance(loss, torch.Tensor)
        assert loss.requires_grad
    
    def test_cli_backward_compatibility(self):
        """Test CLI without new arguments works as before."""
        result = subprocess.run([
            sys.executable, "-m", "meta_learning.cli", "eval",
            "--episodes", "3",
            "--distance", "cosine",
            "--tau", "5.0"
        ], env={"PYTHONPATH": "."}, capture_output=True, text=True)
        
        assert result.returncode == 0
        output = json.loads(result.stdout)
        assert "episodes" in output
        assert output["episodes"] == 3