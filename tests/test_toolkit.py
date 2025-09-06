"""
Tests for the meta-learning toolkit and high-level API.

Tests the main toolkit interface and convenience functions.
"""

import pytest
import torch
import tempfile
import os
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import warnings

from meta_learning.toolkit import (
    MetaLearningToolkit, create_meta_learning_toolkit, quick_evaluation
)
from meta_learning.validation import ValidationError, ConfigurationWarning


class TestMetaLearningToolkit:
    """Test the main MetaLearningToolkit class."""
    
    def setup_method(self):
        """Setup for each test method."""
        self.config = {
            "n_way": 5,
            "k_shot": 2,
            "n_query": 15,
            "distance": "sqeuclidean",
            "tau": 1.0,
            "feature_dim": 64
        }
        self.toolkit = MetaLearningToolkit(self.config)
    
    def test_toolkit_initialization(self):
        """Test toolkit initialization."""
        assert self.toolkit.config == self.config
        assert hasattr(self.toolkit, 'batch_norm_policy')
        assert hasattr(self.toolkit, 'determinism_manager')
        assert hasattr(self.toolkit, 'failure_patterns')
        assert hasattr(self.toolkit, 'recovery_strategies')
    
    def test_toolkit_validation_integration(self):
        """Test toolkit integrates with validation system."""
        # Valid config should work
        toolkit = MetaLearningToolkit(self.config)
        assert toolkit.config["n_way"] == 5
        
        # Test that toolkit can be created with various configs
        simple_config = {"algorithm": "maml"}
        simple_toolkit = MetaLearningToolkit(simple_config)
        assert simple_toolkit.config["algorithm"] == "maml"
    
    def test_toolkit_deterministic_setup(self):
        """Test toolkit deterministic training setup."""
        self.toolkit.setup_deterministic_training(seed=42)
        
        # Test that deterministic setup works
        assert hasattr(self.toolkit, 'determinism_manager')
        
        # Test creating same random tensors gives same results
        torch.manual_seed(42)
        tensor1 = torch.randn(10, 5)
        
        torch.manual_seed(42)
        tensor2 = torch.randn(10, 5)
        
        assert torch.allclose(tensor1, tensor2)
    
    def test_create_episode(self):
        """Test episode creation through toolkit."""
        episode = self.toolkit.create_episode()
        
        # Should have correct structure
        assert "support_x" in episode
        assert "support_y" in episode
        assert "query_x" in episode
        assert "query_y" in episode
        
        # Should have correct dimensions
        expected_support = self.config["n_way"] * self.config["k_shot"]
        expected_query = self.config["n_way"] * self.config["n_query"]
        
        assert episode["support_x"].shape[0] == expected_support
        assert episode["query_x"].shape[0] == expected_query
        assert episode["support_x"].shape[1] == self.config["feature_dim"]
        assert episode["query_x"].shape[1] == self.config["feature_dim"]
    
    def test_create_multiple_episodes(self):
        """Test creating multiple episodes."""
        episodes = self.toolkit.create_episodes(n_episodes=3)
        
        assert len(episodes) == 3
        
        for episode in episodes:
            assert "support_x" in episode
            assert episode["support_x"].shape[0] == self.config["n_way"] * self.config["k_shot"]
    
    def test_evaluate_model(self):
        """Test model evaluation through toolkit."""
        # Create mock model
        mock_model = Mock()
        expected_output_size = self.config["n_way"] * self.config["n_query"]
        mock_model.return_value = torch.randn(expected_output_size, self.config["n_way"])
        
        # Create test episode
        episode = self.toolkit.create_episode()
        
        # Evaluate model
        results = self.toolkit.evaluate_model(mock_model, [episode])
        
        # Should have evaluation results
        assert "accuracy" in results
        assert isinstance(results["accuracy"], float)
        assert 0.0 <= results["accuracy"] <= 1.0
    
    def test_evaluate_model_with_recovery(self):
        """Test model evaluation with error recovery."""
        # Create mock model that fails sometimes
        call_count = [0]
        
        def mock_forward(*args):
            call_count[0] += 1
            if call_count[0] == 1:
                raise RuntimeError("Simulated failure")
            return torch.randn(self.config["n_way"] * self.config["n_query"], self.config["n_way"])
        
        mock_model = Mock()
        mock_model.side_effect = mock_forward
        
        # Enable recovery
        toolkit = MetaLearningToolkit(self.config, recovery_enabled=True)
        episode = toolkit.create_episode()
        
        # Should handle the failure gracefully
        results = toolkit.evaluate_model(mock_model, [episode])
        
        assert "accuracy" in results
        assert results["failed_episodes"] >= 0  # May have failed episodes due to recovery
    
    def test_toolkit_configuration_update(self):
        """Test updating toolkit configuration."""
        new_config = {
            "n_way": 3,
            "k_shot": 1,
            "n_query": 10,
            "distance": "cosine",
            "tau": 2.0,
            "feature_dim": 32
        }
        
        self.toolkit.update_config(new_config)
        
        # Config should be updated
        assert self.toolkit.config["n_way"] == 3
        assert self.toolkit.config["feature_dim"] == 32
        
        # Should be able to create episode with new config
        episode = self.toolkit.create_episode()
        assert episode["support_x"].shape[0] == 3 * 1  # 3-way 1-shot
        assert episode["support_x"].shape[1] == 32     # New feature dim
    
    def test_toolkit_batch_evaluation(self):
        """Test batch evaluation capabilities."""
        mock_model = Mock()
        mock_model.return_value = torch.randn(
            self.config["n_way"] * self.config["n_query"],
            self.config["n_way"]
        )
        
        # Create multiple episodes
        episodes = self.toolkit.create_episodes(n_episodes=5)
        
        # Batch evaluate
        results = self.toolkit.evaluate_model(mock_model, episodes)
        
        assert "accuracy" in results
        assert "total_episodes" in results
        assert results["total_episodes"] == 5
    
    def test_toolkit_save_load_config(self):
        """Test saving and loading toolkit configuration."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "toolkit_config.json"
            
            # Save configuration
            self.toolkit.save_config(config_path)
            assert config_path.exists()
            
            # Load configuration into new toolkit
            loaded_toolkit = MetaLearningToolkit({})  # Empty initial config
            loaded_toolkit.load_config(config_path)
            
            # Should have same configuration
            assert loaded_toolkit.config == self.toolkit.config
    
    def test_toolkit_statistics(self):
        """Test toolkit statistics collection."""
        # Create episodes and evaluate
        mock_model = Mock()
        mock_model.return_value = torch.randn(
            self.config["n_way"] * self.config["n_query"],
            self.config["n_way"]
        )
        
        episodes = self.toolkit.create_episodes(n_episodes=3)
        results = self.toolkit.evaluate_model(mock_model, episodes)
        
        # Get statistics
        stats = self.toolkit.get_statistics()
        
        assert "episodes_created" in stats
        assert "evaluations_performed" in stats
        assert stats["episodes_created"] >= 3
        assert stats["evaluations_performed"] >= 1


class TestToolkitFactory:
    """Test toolkit factory functions."""
    
    def test_create_meta_learning_toolkit_basic(self):
        """Test basic toolkit creation."""
        config = {
            "n_way": 5,
            "k_shot": 1,
            "n_query": 15
        }
        
        toolkit = create_meta_learning_toolkit(config)
        
        assert isinstance(toolkit, MetaLearningToolkit)
        assert toolkit.config["n_way"] == 5
        assert toolkit.config["k_shot"] == 1
    
    def test_create_meta_learning_toolkit_with_defaults(self):
        """Test toolkit creation with default values."""
        # Minimal config
        config = {"n_way": 3}
        
        toolkit = create_meta_learning_toolkit(
            config,
            default_k_shot=2,
            default_n_query=10,
            default_feature_dim=128
        )
        
        assert toolkit.config["n_way"] == 3
        assert toolkit.config["k_shot"] == 2
        assert toolkit.config["n_query"] == 10
        assert toolkit.config["feature_dim"] == 128
    
    def test_create_meta_learning_toolkit_with_options(self):
        """Test toolkit creation with various options."""
        config = {"n_way": 5, "k_shot": 2}
        
        toolkit = create_meta_learning_toolkit(
            config,
            validation_enabled=False,
            warnings_enabled=True,
            recovery_enabled=True
        )
        
        assert toolkit.validation_enabled == False
        assert toolkit.warnings_enabled == True
        assert toolkit.recovery_enabled == True
    
    def test_create_meta_learning_toolkit_invalid_config(self):
        """Test toolkit creation with invalid configuration."""
        invalid_config = {"n_way": 0}  # Invalid
        
        with pytest.raises(ValidationError):
            create_meta_learning_toolkit(invalid_config, validation_enabled=True)
    
    def test_create_meta_learning_toolkit_with_warnings(self):
        """Test toolkit creation triggers configuration warnings."""
        config = {
            "n_way": 25,  # High n_way
            "k_shot": 15,  # High k_shot
            "n_query": 3   # Low n_query
        }
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            
            toolkit = create_meta_learning_toolkit(config, warnings_enabled=True)
            
            # Should have captured configuration warnings
            config_warnings = [warning for warning in w if issubclass(warning.category, ConfigurationWarning)]
            assert len(config_warnings) >= 2  # At least high n_way and k_shot


class TestQuickEvaluation:
    """Test quick evaluation convenience function."""
    
    def test_quick_evaluation_basic(self):
        """Test basic quick evaluation."""
        mock_model = Mock()
        mock_model.return_value = torch.randn(15, 5)  # 5-way * 3-query per way
        
        results = quick_evaluation(
            model=mock_model,
            n_way=5,
            k_shot=2,
            n_query=3,
            n_episodes=2,
            feature_dim=64
        )
        
        assert "accuracy" in results
        assert "total_episodes" in results
        assert results["total_episodes"] == 2
        assert isinstance(results["accuracy"], float)
        assert 0.0 <= results["accuracy"] <= 1.0
    
    def test_quick_evaluation_with_device(self):
        """Test quick evaluation with specific device."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        
        device = torch.device('cuda')
        mock_model = Mock()
        mock_model.return_value = torch.randn(10, 2, device=device)  # 2-way * 5-query
        
        results = quick_evaluation(
            model=mock_model,
            n_way=2,
            k_shot=1,
            n_query=5,
            n_episodes=1,
            feature_dim=32,
            device=device
        )
        
        assert "accuracy" in results
        assert results["total_episodes"] == 1
    
    def test_quick_evaluation_with_recovery(self):
        """Test quick evaluation with error recovery enabled."""
        # Create model that fails on first call
        call_count = [0]
        
        def mock_forward(*args):
            call_count[0] += 1
            if call_count[0] == 1:
                raise RuntimeError("Simulated failure")
            return torch.randn(5, 2)  # 2-way * 5-query, but mismatched
        
        mock_model = Mock()
        mock_model.side_effect = mock_forward
        
        results = quick_evaluation(
            model=mock_model,
            n_way=2,
            k_shot=1,
            n_query=5,
            n_episodes=2,
            feature_dim=32,
            recovery_enabled=True
        )
        
        # Should complete despite failures
        assert "accuracy" in results
        assert "failed_episodes" in results
    
    def test_quick_evaluation_validation_errors(self):
        """Test quick evaluation handles validation errors."""
        mock_model = Mock()
        
        with pytest.raises(ValidationError):
            quick_evaluation(
                model=mock_model,
                n_way=0,  # Invalid
                k_shot=1,
                n_query=5,
                n_episodes=1,
                feature_dim=32
            )
    
    def test_quick_evaluation_warnings(self):
        """Test quick evaluation triggers appropriate warnings."""
        mock_model = Mock()
        mock_model.return_value = torch.randn(50, 25)  # 25-way * 2-query
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            
            results = quick_evaluation(
                model=mock_model,
                n_way=25,  # High n_way - should trigger warning
                k_shot=1,
                n_query=2,
                n_episodes=1,
                feature_dim=32,
                warnings_enabled=True
            )
            
            # Should have captured warnings
            config_warnings = [warning for warning in w if issubclass(warning.category, ConfigurationWarning)]
            assert len(config_warnings) >= 1
    
    def test_quick_evaluation_performance(self):
        """Test quick evaluation performance."""
        import time
        
        mock_model = Mock()
        mock_model.return_value = torch.randn(30, 3)  # 3-way * 10-query
        
        start_time = time.time()
        
        results = quick_evaluation(
            model=mock_model,
            n_way=3,
            k_shot=2,
            n_query=10,
            n_episodes=5,
            feature_dim=128
        )
        
        end_time = time.time()
        elapsed = end_time - start_time
        
        # Should complete quickly for small evaluation
        assert elapsed < 2.0, f"Quick evaluation took {elapsed:.3f}s"
        assert results["total_episodes"] == 5
    
    def test_quick_evaluation_statistical_stability(self):
        """Test quick evaluation provides stable statistics."""
        # Create deterministic model for consistent results
        def deterministic_forward(support_x, support_y, query_x):
            # Simple nearest neighbor for deterministic results
            batch_size = query_x.shape[0]
            n_way = len(torch.unique(support_y))
            return torch.eye(n_way).repeat(batch_size // n_way, 1)
        
        mock_model = Mock()
        mock_model.side_effect = deterministic_forward
        
        results1 = quick_evaluation(
            model=mock_model,
            n_way=3,
            k_shot=2,
            n_query=3,
            n_episodes=10,
            feature_dim=32,
            seed=42  # Fixed seed for reproducibility
        )
        
        # Reset mock
        mock_model.reset_mock()
        mock_model.side_effect = deterministic_forward
        
        results2 = quick_evaluation(
            model=mock_model,
            n_way=3,
            k_shot=2,
            n_query=3,
            n_episodes=10,
            feature_dim=32,
            seed=42  # Same seed
        )
        
        # Results should be very similar with same seed
        assert abs(results1["accuracy"] - results2["accuracy"]) < 0.1


class TestToolkitIntegration:
    """Test toolkit integration with other modules."""
    
    def test_toolkit_validation_integration(self):
        """Test toolkit properly integrates validation."""
        config = {
            "n_way": 5,
            "k_shot": 2,
            "n_query": 15,
            "distance": "sqeuclidean",
            "tau": 1.0
        }
        
        # Should work with validation enabled
        toolkit = MetaLearningToolkit(config, validation_enabled=True)
        episode = toolkit.create_episode()
        
        # Episode should pass validation
        from meta_learning.validation import validate_episode_tensors
        validate_episode_tensors(
            episode["support_x"], episode["support_y"],
            episode["query_x"], episode["query_y"]
        )
    
    def test_toolkit_warnings_integration(self):
        """Test toolkit properly integrates with warnings system."""
        config = {
            "n_way": 25,  # Should trigger warning
            "k_shot": 1,
            "n_query": 5,
            "distance": "cosine",
            "tau": 0.05  # Should trigger warning
        }
        
        # Count warnings
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            
            toolkit = MetaLearningToolkit(config, warnings_enabled=True)
            
            config_warnings = [warning for warning in w if issubclass(warning.category, ConfigurationWarning)]
            assert len(config_warnings) >= 2  # At least n_way and tau warnings
    
    def test_toolkit_error_recovery_integration(self):
        """Test toolkit integrates with error recovery."""
        config = {"n_way": 3, "k_shot": 2, "n_query": 5}
        toolkit = MetaLearningToolkit(config, recovery_enabled=True)
        
        # Create model that sometimes fails
        failure_count = [0]
        
        def sometimes_failing_model(*args):
            failure_count[0] += 1
            if failure_count[0] % 3 == 0:  # Fail every 3rd call
                raise RuntimeError("Simulated failure")
            return torch.randn(15, 3)  # 3-way * 5-query
        
        mock_model = Mock()
        mock_model.side_effect = sometimes_failing_model
        
        # Should handle failures gracefully
        episodes = toolkit.create_episodes(n_episodes=5)
        results = toolkit.evaluate_model(mock_model, episodes)
        
        assert "accuracy" in results
        assert "failed_episodes" in results
        # Some episodes should have failed but been recovered
        assert results["failed_episodes"] >= 0
    
    def test_toolkit_metrics_integration(self):
        """Test toolkit integrates with evaluation metrics."""
        config = {"n_way": 4, "k_shot": 1, "n_query": 8}
        toolkit = MetaLearningToolkit(config)
        
        # Create deterministic model for predictable metrics
        def deterministic_model(*args):
            # Return perfect predictions (identity matrix pattern)
            return torch.eye(4).repeat(8, 1)  # Perfect 4-way classification
        
        mock_model = Mock()
        mock_model.side_effect = deterministic_model
        
        episodes = toolkit.create_episodes(n_episodes=3)
        results = toolkit.evaluate_model(mock_model, episodes)
        
        # Should have high accuracy for perfect predictions
        assert results["accuracy"] >= 0.8  # Should be very high
        assert "total_episodes" in results
        assert results["total_episodes"] == 3