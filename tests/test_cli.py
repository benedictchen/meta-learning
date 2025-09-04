"""
Tests for CLI Module - Boost Coverage to 30%+
============================================

Tests the command-line interface functionality to achieve maximum coverage
while validating demo workflows and CLI argument parsing.
"""

import pytest
import torch
# DEMO/TEST REPRODUCIBILITY: Set seed for consistent synthetic data
torch.manual_seed(42)  # Reproducible test/demo data
import sys
from io import StringIO
from unittest.mock import patch, MagicMock

# Import CLI components
from meta_learning.cli import (
    SimpleClassifier,
    generate_demo_data,
    demo_test_time_compute,
    demo_maml_variants,
    demo_online_meta_learning,
    demo_advanced_dataset,
    main
)

class TestCLIComponents:
    """Test CLI component functionality."""
    
    def test_simple_classifier_creation(self):
        """Test SimpleClassifier initialization and forward pass."""
        # Test default parameters
        classifier = SimpleClassifier()
        assert classifier is not None
        
        # Test custom parameters
        custom_classifier = SimpleClassifier(input_dim=512, hidden_dim=128, output_dim=10)
        assert custom_classifier is not None
        
        # Test forward pass
        batch_size = 32
        input_data = torch.randn(batch_size, 784)
        output = classifier(input_data)
        
        assert output.shape == (batch_size, 5)  # Default output_dim
        assert not torch.isnan(output).any()
        
        # Test custom forward pass
        custom_input = torch.randn(16, 512)
        custom_output = custom_classifier(custom_input)
        assert custom_output.shape == (16, 10)
    
    def test_generate_demo_data(self):
        """Test demo data generation with various parameters."""
        # Test default parameters
        data, labels = generate_demo_data()
        
        assert data.shape == (200, 784)  # 10 classes * 20 samples
        assert labels.shape == (200,)
        assert len(torch.unique(labels)) == 10
        assert labels.min() >= 0
        assert labels.max() < 10
        
        # Test custom parameters
        custom_data, custom_labels = generate_demo_data(n_classes=5, samples_per_class=15)
        
        assert custom_data.shape == (75, 784)  # 5 classes * 15 samples
        assert custom_labels.shape == (75,)
        assert len(torch.unique(custom_labels)) == 5
        assert custom_labels.max() < 5
        
        # Test data structure (class patterns should be different)
        class_means = []
        for class_id in range(5):
            class_mask = custom_labels == class_id
            class_data = custom_data[class_mask]
            class_mean = class_data.mean(dim=0)
            class_means.append(class_mean)
        
        # Classes should have different means (due to class_mean generation)
        class_means_tensor = torch.stack(class_means)
        pairwise_distances = torch.cdist(class_means_tensor, class_means_tensor)
        # Not all distances should be zero (classes should be distinguishable)
        assert (pairwise_distances > 0.1).any()

class TestCLIDemoFunctions:
    """Test CLI demo functions with output capture."""
    
    def test_demo_test_time_compute(self):
        """Test test-time compute demo execution."""
        # Capture stdout to verify demo runs
        captured_output = StringIO()
        
        with patch('sys.stdout', captured_output):
            with patch('meta_learning.cli.TestTimeComputeScaler') as mock_scaler:
                with patch('meta_learning.cli.PrototypicalNetworks') as mock_proto:
                    # Mock the scaler behavior
                    mock_instance = MagicMock()
                    mock_instance.scale_compute.return_value = (
                        torch.randn(15, 5),  # predictions
                        {'compute_steps': 8, 'confidence': 0.85}  # metrics
                    )
                    mock_scaler.return_value = mock_instance
                    
                    # Mock prototypical networks
                    mock_proto.return_value = MagicMock()
                    
                    # Run the demo
                    demo_test_time_compute()
        
        output = captured_output.getvalue()
        
        # Verify demo output contains expected content
        assert "🚀 Demo: Test-Time Compute Scaling" in output
        assert "test-time compute scaling research" in output
        assert "Steps used" in output
        assert "Confidence" in output
        
        # Verify mocks were called
        mock_scaler.assert_called_once()
        mock_instance.scale_compute.assert_called_once()
    
    def test_demo_maml_variants(self):
        """Test MAML variants demo execution."""
        captured_output = StringIO()
        
        with patch('sys.stdout', captured_output):
            with patch('meta_learning.cli.MAMLLearner') as mock_maml:
                # Mock the MAML learner behavior
                mock_instance = MagicMock()
                mock_instance.meta_test.return_value = {
                    'accuracy': 0.78,
                    'loss': 0.45,
                    'adaptation_steps': 5
                }
                mock_maml.return_value = mock_instance
                
                # Run the demo
                demo_maml_variants()
        
        output = captured_output.getvalue()
        
        # Verify demo output contains some expected content
        assert "Demo" in output
        
        # Verify mock was called
        mock_maml.assert_called_once()
        mock_instance.meta_test.assert_called_once()
    
    def test_demo_online_meta_learning(self):
        """Test online meta learning demo execution."""
        captured_output = StringIO()
        
        with patch('sys.stdout', captured_output):
            with patch('meta_learning.cli.OnlineMetaLearner') as mock_online:
                # Mock online learner behavior
                mock_instance = MagicMock()
                mock_instance.learn_task.return_value = {
                    'task_accuracy': 0.81,
                    'forgetting_metric': 0.15,
                    'memory_usage': 0.42
                }
                mock_online.return_value = mock_instance
                
                # Run the demo
                demo_online_meta_learning()
        
        output = captured_output.getvalue()
        
        # Verify demo output contains some expected content
        assert "Demo" in output
        
        # Verify mock was called
        mock_online.assert_called_once()
    
    def test_demo_advanced_dataset(self):
        """Test advanced dataset demo execution."""
        captured_output = StringIO()
        
        with patch('sys.stdout', captured_output):
            with patch('meta_learning.cli.MetaLearningDataset') as mock_dataset:
                # Mock dataset behavior
                mock_instance = MagicMock()
                mock_dataset.return_value = mock_instance
                
                # Run the demo
                demo_advanced_dataset()
        
        output = captured_output.getvalue()
        
        # Verify demo output contains some expected content
        assert "Demo" in output
        
        # Verify mock was called
        mock_dataset.assert_called_once()

class TestCLIMain:
    """Test main CLI function with argument parsing."""
    
    def test_main_help(self):
        """Test CLI help functionality."""
        with patch('sys.argv', ['meta_learning', '--help']):
            with patch('sys.stdout', StringIO()) as captured:
                with pytest.raises(SystemExit):
                    main()
                
                output = captured.getvalue()
                assert "Meta-Learning CLI Tool" in output
                assert "--demo" in output
    
    def test_main_demo_test_time_compute(self):
        """Test main with test-time compute demo."""
        with patch('sys.argv', ['meta_learning', '--demo', 'test-time-compute']):
            with patch('meta_learning.cli.demo_test_time_compute') as mock_demo:
                main()
                mock_demo.assert_called_once()
    
    def test_main_demo_maml(self):
        """Test main with MAML demo."""
        with patch('sys.argv', ['meta_learning', '--demo', 'maml']):
            with patch('meta_learning.cli.demo_maml_variants') as mock_demo:
                main()
                mock_demo.assert_called_once()
    
    def test_main_demo_online(self):
        """Test main with online meta learning demo."""
        with patch('sys.argv', ['meta_learning', '--demo', 'online']):
            with patch('meta_learning.cli.demo_online_meta_learning') as mock_demo:
                main()
                mock_demo.assert_called_once()
    
    def test_main_demo_dataset(self):
        """Test main with dataset demo."""
        with patch('sys.argv', ['meta_learning', '--demo', 'dataset']):
            with patch('meta_learning.cli.demo_advanced_dataset') as mock_demo:
                main()
                mock_demo.assert_called_once()
    
    def test_main_all_demos(self):
        """Test main running all demos."""
        with patch('sys.argv', ['meta_learning', '--demo', 'all']):
            with patch('meta_learning.cli.demo_test_time_compute') as mock_ttc:
                with patch('meta_learning.cli.demo_maml_variants') as mock_maml:
                    with patch('meta_learning.cli.demo_online_meta_learning') as mock_online:
                        with patch('meta_learning.cli.demo_advanced_dataset') as mock_dataset:
                            main()
                            
                            # Verify all demos were called
                            mock_ttc.assert_called_once()
                            mock_maml.assert_called_once()
                            mock_online.assert_called_once()
                            mock_dataset.assert_called_once()
    
    def test_main_invalid_demo(self):
        """Test main with invalid demo name."""
        with patch('sys.argv', ['meta_learning', '--demo', 'invalid']):
            with patch('sys.stdout', StringIO()) as captured:
                main()
                
                output = captured.getvalue()
                assert "Unknown demo" in output
                assert "Available demos" in output

class TestCLIIntegration:
    """Integration tests for CLI functionality."""
    
    def test_data_generation_integration(self):
        """Test integration between data generation and model usage."""
        # Generate data
        data, labels = generate_demo_data(n_classes=3, samples_per_class=10)
        
        # Create model
        model = SimpleClassifier(output_dim=3)
        
        # Test forward pass with generated data
        outputs = model(data)
        
        assert outputs.shape == (30, 3)  # 3 classes * 10 samples, 3 outputs
        assert not torch.isnan(outputs).any()
        
        # Test that different classes produce different patterns
        class_outputs = []
        for class_id in range(3):
            class_mask = labels == class_id
            class_output = outputs[class_mask].mean(dim=0)
            class_outputs.append(class_output)
        
        class_outputs_tensor = torch.stack(class_outputs)
        distances = torch.cdist(class_outputs_tensor, class_outputs_tensor)
        
        # Classes should produce distinguishable outputs
        off_diagonal = distances[~torch.eye(3, dtype=bool)]
        assert (off_diagonal > 0.01).any()  # Some separation between classes
    
    def test_cli_robustness(self):
        """Test CLI robustness with edge cases."""
        # Test with minimal data
        data, labels = generate_demo_data(n_classes=2, samples_per_class=1)
        assert data.shape == (2, 784)
        assert len(torch.unique(labels)) == 2
        
        # Test with single class
        single_data, single_labels = generate_demo_data(n_classes=1, samples_per_class=5)
        assert single_data.shape == (5, 784)
        assert len(torch.unique(single_labels)) == 1
        
        # Test model with minimal output
        minimal_model = SimpleClassifier(output_dim=1)
        minimal_output = minimal_model(single_data)
        assert minimal_output.shape == (5, 1)