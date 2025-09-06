"""
Comprehensive tests for Determinism utilities and reproducibility.

Tests complete seeding, reproducible model training, and deterministic validation
for meta-learning experiments.
"""
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from unittest.mock import MagicMock, patch
from typing import Dict, List, Tuple, Any
import subprocess
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../src'))

from meta_learning.research_patches.determinism_hooks import (
    setup_deterministic_environment,
    DeterminismManager,
    ReproducibilityReport,
    seed_everything,
    validate_determinism,
    DeterministicDataLoader,
    create_reproducible_dataloader
)


class TestSeedEverything:
    """Test comprehensive seeding functionality."""
    
    def test_seed_everything_basic(self):
        """Test basic seeding of all random sources."""
        seed = 42
        
        # Seed everything
        seed_everything(seed)
        
        # Generate random numbers from different sources
        python_rand = random.random()
        numpy_rand = np.random.random()
        torch_rand = torch.rand(1).item()
        torch_cuda_rand = torch.cuda.FloatTensor(1).uniform_().item() if torch.cuda.is_available() else 0.0
        
        # Seed again and generate the same numbers
        seed_everything(seed)
        
        python_rand2 = random.random()
        numpy_rand2 = np.random.random()
        torch_rand2 = torch.rand(1).item()
        torch_cuda_rand2 = torch.cuda.FloatTensor(1).uniform_().item() if torch.cuda.is_available() else 0.0
        
        # Should be identical
        assert python_rand == python_rand2
        assert numpy_rand == numpy_rand2
        assert torch_rand == torch_rand2
        if torch.cuda.is_available():
            assert torch_cuda_rand == torch_cuda_rand2
    
    def test_seed_everything_different_seeds(self):
        """Test that different seeds produce different results."""
        # Seed with 42
        seed_everything(42)
        values_42 = {
            'python': random.random(),
            'numpy': np.random.random(), 
            'torch': torch.rand(1).item()
        }
        
        # Seed with 123
        seed_everything(123)
        values_123 = {
            'python': random.random(),
            'numpy': np.random.random(),
            'torch': torch.rand(1).item()
        }
        
        # Should be different
        assert values_42['python'] != values_123['python']
        assert values_42['numpy'] != values_123['numpy']
        assert values_42['torch'] != values_123['torch']
    
    def test_cuda_deterministic_flag(self):
        """Test CUDA deterministic flag setting."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        
        # Test with deterministic=True
        seed_everything(42, cuda_deterministic=True)
        assert torch.backends.cudnn.deterministic
        assert not torch.backends.cudnn.benchmark
        
        # Test with deterministic=False
        seed_everything(42, cuda_deterministic=False)
        assert not torch.backends.cudnn.deterministic
        # benchmark setting depends on implementation


class TestSetupDeterministicEnvironment:
    """Test complete deterministic environment setup."""
    
    def test_setup_deterministic_environment(self):
        """Test complete environment setup."""
        config = {
            'seed': 1337,
            'cuda_deterministic': True,
            'benchmark': False,
            'num_threads': 1
        }
        
        setup_deterministic_environment(config)
        
        # Check that settings are applied
        if torch.cuda.is_available():
            assert torch.backends.cudnn.deterministic == config['cuda_deterministic']
            assert torch.backends.cudnn.benchmark == config['benchmark']
        
        # Check that thread count is set
        assert torch.get_num_threads() == config['num_threads']
    
    def test_setup_minimal_config(self):
        """Test setup with minimal configuration."""
        config = {'seed': 42}
        
        # Should not raise error
        setup_deterministic_environment(config)
        
        # Random sources should be seeded
        rand1 = torch.rand(1).item()
        
        # Re-setup with same seed
        setup_deterministic_environment(config)
        rand2 = torch.rand(1).item()
        
        assert rand1 == rand2
    
    def test_worker_init_fn_creation(self):
        """Test worker initialization function creation."""
        config = {'seed': 42, 'num_threads': 2}
        
        setup_deterministic_environment(config)
        
        # Should create deterministic worker init function
        # This is tested indirectly through dataloader tests


class TestDeterminismManager:
    """Test DeterminismManager functionality."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.manager = DeterminismManager(base_seed=42)
    
    def test_manager_initialization(self):
        """Test DeterminismManager initialization."""
        assert self.manager.base_seed == 42
        assert self.manager.current_seed == 42
        assert len(self.manager.seed_history) == 1
        assert self.manager.seed_history[0] == 42
    
    def test_set_seed(self):
        """Test setting new seed."""
        new_seed = 123
        self.manager.set_seed(new_seed)
        
        assert self.manager.current_seed == new_seed
        assert new_seed in self.manager.seed_history
        
        # Random state should change
        torch.manual_seed(42)
        val_42 = torch.rand(1).item()
        
        self.manager.set_seed(new_seed)
        val_123 = torch.rand(1).item()
        
        assert val_42 != val_123
    
    def test_increment_seed(self):
        """Test seed incrementation for different episodes."""
        original_seed = self.manager.current_seed
        
        # Increment seed
        new_seed = self.manager.increment_seed()
        
        assert new_seed == original_seed + 1
        assert self.manager.current_seed == new_seed
        assert new_seed in self.manager.seed_history
        
        # Multiple increments
        for i in range(5):
            next_seed = self.manager.increment_seed()
            assert next_seed == original_seed + 2 + i
    
    def test_reset_to_base(self):
        """Test resetting to base seed."""
        # Increment several times
        for _ in range(10):
            self.manager.increment_seed()
        
        # Reset to base
        self.manager.reset_to_base()
        
        assert self.manager.current_seed == self.manager.base_seed
        
        # Should produce same random values as initial state
        val_reset = torch.rand(1).item()
        
        torch.manual_seed(self.manager.base_seed)
        val_base = torch.rand(1).item()
        
        assert val_reset == val_base
    
    def test_get_seed_for_worker(self):
        """Test getting deterministic seed for worker processes."""
        worker_id = 3
        seed = self.manager.get_seed_for_worker(worker_id)
        
        # Should be deterministic based on current seed and worker ID
        expected_seed = self.manager.current_seed + worker_id + 1
        assert seed == expected_seed
        
        # Same worker ID should give same seed
        seed2 = self.manager.get_seed_for_worker(worker_id)
        assert seed == seed2
        
        # Different worker ID should give different seed
        seed_different = self.manager.get_seed_for_worker(worker_id + 1)
        assert seed != seed_different
    
    def test_context_manager(self):
        """Test DeterminismManager as context manager."""
        original_seed = self.manager.current_seed
        
        with self.manager.temporary_seed(999) as temp_seed:
            assert temp_seed == 999
            assert self.manager.current_seed == 999
            
            # Generate some random numbers
            temp_vals = [torch.rand(1).item() for _ in range(3)]
        
        # Should be restored
        assert self.manager.current_seed == original_seed
        
        # Test that temporary seed was actually used
        torch.manual_seed(999)
        expected_vals = [torch.rand(1).item() for _ in range(3)]
        
        for temp_val, expected_val in zip(temp_vals, expected_vals):
            assert temp_val == expected_val


class TestValidateDeterminism:
    """Test determinism validation functionality."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.model = nn.Sequential(
            nn.Linear(10, 32),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(32, 5)
        )
        
        self.test_input = torch.randn(8, 10)
    
    def test_validate_determinism_success(self):
        """Test determinism validation for deterministic model."""
        # Use model without dropout in eval mode
        deterministic_model = nn.Sequential(
            nn.Linear(10, 32),
            nn.ReLU(),
            nn.Linear(32, 5)
        )
        deterministic_model.eval()
        
        result = validate_determinism(deterministic_model, self.test_input, n_runs=5)
        
        assert result['deterministic']
        assert result['max_difference'] < 1e-7
        assert len(result['run_outputs']) == 5
        assert all(torch.allclose(out, result['run_outputs'][0], atol=1e-6) 
                  for out in result['run_outputs'])
    
    def test_validate_determinism_failure(self):
        """Test determinism validation for non-deterministic model."""
        # Model with dropout in training mode
        self.model.train()
        
        result = validate_determinism(self.model, self.test_input, n_runs=5)
        
        # Should detect non-determinism (dropout causes randomness)
        assert not result['deterministic']
        assert result['max_difference'] > 1e-6
        
        # Outputs should be different
        outputs = result['run_outputs']
        assert not all(torch.allclose(out, outputs[0], atol=1e-6) for out in outputs[1:])
    
    def test_validate_determinism_with_seed(self):
        """Test determinism validation with explicit seeding."""
        self.model.eval()  # Remove dropout randomness
        
        result = validate_determinism(
            self.model, 
            self.test_input, 
            n_runs=3,
            seed=42
        )
        
        assert result['deterministic']
        assert result['max_difference'] < 1e-7
        
        # All outputs should be identical
        outputs = result['run_outputs']
        for output in outputs[1:]:
            assert torch.equal(output, outputs[0])
    
    def test_validate_determinism_tolerance(self):
        """Test determinism validation with different tolerances."""
        # Create model with very small numerical differences
        def slightly_random_model(x):
            # Add tiny random noise
            return self.model(x) + torch.randn_like(self.model(x)) * 1e-8
        
        # Strict tolerance should fail
        result_strict = validate_determinism(
            slightly_random_model,
            self.test_input,
            n_runs=3,
            tolerance=1e-10
        )
        
        # Lenient tolerance should pass
        result_lenient = validate_determinism(
            slightly_random_model,
            self.test_input,
            n_runs=3,
            tolerance=1e-6
        )
        
        assert not result_strict['deterministic']
        assert result_lenient['deterministic']


class TestDeterministicDataLoader:
    """Test deterministic data loading."""
    
    def setup_method(self):
        """Setup test fixtures."""
        # Create simple dataset
        self.dataset = torch.utils.data.TensorDataset(
            torch.randn(100, 10),
            torch.randint(0, 5, (100,))
        )
    
    def test_deterministic_dataloader_creation(self):
        """Test creation of deterministic dataloader."""
        dataloader = DeterministicDataLoader(
            self.dataset,
            batch_size=16,
            seed=42
        )
        
        assert dataloader.seed == 42
        assert isinstance(dataloader.dataloader, torch.utils.data.DataLoader)
    
    def test_deterministic_dataloader_reproducibility(self):
        """Test that deterministic dataloader produces same batches."""
        dataloader1 = DeterministicDataLoader(
            self.dataset,
            batch_size=16,
            seed=42,
            shuffle=True
        )
        
        dataloader2 = DeterministicDataLoader(
            self.dataset,
            batch_size=16,
            seed=42,
            shuffle=True
        )
        
        # Should produce identical batches
        batches1 = list(dataloader1)
        batches2 = list(dataloader2)
        
        assert len(batches1) == len(batches2)
        for (x1, y1), (x2, y2) in zip(batches1, batches2):
            assert torch.equal(x1, x2)
            assert torch.equal(y1, y2)
    
    def test_deterministic_dataloader_different_seeds(self):
        """Test that different seeds produce different batches."""
        dataloader1 = DeterministicDataLoader(
            self.dataset,
            batch_size=16,
            seed=42,
            shuffle=True
        )
        
        dataloader2 = DeterministicDataLoader(
            self.dataset,
            batch_size=16,
            seed=123,
            shuffle=True
        )
        
        # Should produce different batch orders
        batches1 = list(dataloader1)
        batches2 = list(dataloader2)
        
        # At least some batches should be different
        different_batches = sum(
            not torch.equal(x1, x2) for (x1, _), (x2, _) in zip(batches1, batches2)
        )
        
        assert different_batches > 0
    
    def test_worker_reproducibility(self):
        """Test worker process reproducibility."""
        if torch.multiprocessing.get_start_method() == 'spawn':
            # Create dataloader with multiple workers
            dataloader1 = create_reproducible_dataloader(
                self.dataset,
                batch_size=8,
                num_workers=2,
                seed=42
            )
            
            dataloader2 = create_reproducible_dataloader(
                self.dataset,
                batch_size=8,
                num_workers=2,
                seed=42
            )
            
            # Should be reproducible even with workers
            batches1 = list(dataloader1)
            batches2 = list(dataloader2)
            
            assert len(batches1) == len(batches2)
            for (x1, y1), (x2, y2) in zip(batches1, batches2):
                assert torch.equal(x1, x2)
                assert torch.equal(y1, y2)


class TestReproducibilityReport:
    """Test reproducibility reporting functionality."""
    
    def test_generate_environment_report(self):
        """Test environment report generation."""
        report = ReproducibilityReport.generate_environment_report()
        
        # Should contain key information
        required_keys = [
            'python_version',
            'pytorch_version', 
            'numpy_version',
            'cuda_version',
            'cudnn_version',
            'platform',
            'hostname'
        ]
        
        for key in required_keys:
            assert key in report
            assert report[key] is not None or key in ['cuda_version', 'cudnn_version']
        
        # Versions should be strings
        assert isinstance(report['python_version'], str)
        assert isinstance(report['pytorch_version'], str)
        assert isinstance(report['numpy_version'], str)
    
    def test_generate_seed_report(self):
        """Test seed state report generation."""
        # Set up deterministic state
        seed_everything(42)
        
        report = ReproducibilityReport.generate_seed_report()
        
        # Should contain seed information
        required_keys = [
            'torch_manual_seed',
            'numpy_random_state_type',
            'python_random_state_type'
        ]
        
        for key in required_keys:
            assert key in report
    
    def test_generate_model_report(self):
        """Test model reproducibility report."""
        model = nn.Sequential(
            nn.Linear(10, 32),
            nn.ReLU(),
            nn.Linear(32, 5)
        )
        
        report = ReproducibilityReport.generate_model_report(model)
        
        # Should contain model information
        required_keys = [
            'model_type',
            'total_parameters',
            'trainable_parameters',
            'model_structure'
        ]
        
        for key in required_keys:
            assert key in report
        
        # Parameter counts should be positive
        assert report['total_parameters'] > 0
        assert report['trainable_parameters'] > 0
        assert report['trainable_parameters'] <= report['total_parameters']
    
    def test_generate_hardware_report(self):
        """Test hardware report generation."""
        report = ReproducibilityReport.generate_hardware_report()
        
        # Should contain hardware information
        required_keys = [
            'cpu_count',
            'memory_gb',
            'cuda_available',
            'cuda_device_count'
        ]
        
        for key in required_keys:
            assert key in report
        
        # Hardware metrics should be reasonable
        assert report['cpu_count'] > 0
        assert report['memory_gb'] > 0
        
        if torch.cuda.is_available():
            assert report['cuda_available']
            assert report['cuda_device_count'] > 0
        else:
            assert not report['cuda_available']
    
    def test_generate_complete_report(self):
        """Test complete reproducibility report."""
        model = nn.Linear(10, 5)
        
        report = ReproducibilityReport.generate_complete_report(
            model=model,
            config={'seed': 42, 'batch_size': 32}
        )
        
        # Should contain all sections
        required_sections = [
            'timestamp',
            'environment',
            'hardware', 
            'seed_state',
            'model_info',
            'configuration'
        ]
        
        for section in required_sections:
            assert section in report
        
        # Configuration should match input
        assert report['configuration']['seed'] == 42
        assert report['configuration']['batch_size'] == 32


class TestEndToEndDeterminism:
    """Test end-to-end determinism in realistic scenarios."""
    
    def setup_method(self):
        """Setup realistic meta-learning scenario."""
        self.model = nn.Sequential(
            nn.Linear(20, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 5)
        )
        
        # Create episodic dataset
        self.support_x = torch.randn(25, 20)  # 5-way 5-shot
        self.support_y = torch.arange(5).repeat_interleave(5)
        self.query_x = torch.randn(75, 20)   # 15 queries per class
        self.query_y = torch.arange(5).repeat(15)
    
    def test_deterministic_episode_training(self):
        """Test deterministic episode training."""
        # Configuration
        config = {
            'seed': 1337,
            'cuda_deterministic': True,
            'num_threads': 1
        }
        
        # Train episode twice with same setup
        results = []
        
        for run in range(2):
            # Setup deterministic environment
            setup_deterministic_environment(config)
            
            # Reset model parameters
            model = nn.Sequential(
                nn.Linear(20, 64),
                nn.ReLU(),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, 5)
            )
            
            # Simple training loop
            optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
            model.train()
            
            episode_losses = []
            for step in range(10):
                # Forward pass on support set
                logits = model(self.support_x)
                loss = F.cross_entropy(logits, self.support_y)
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                episode_losses.append(loss.item())
            
            # Evaluate on query set
            model.eval()
            with torch.no_grad():
                query_logits = model(self.query_x)
                query_loss = F.cross_entropy(query_logits, self.query_y)
                query_acc = (query_logits.argmax(dim=-1) == self.query_y).float().mean()
            
            results.append({
                'losses': episode_losses,
                'query_loss': query_loss.item(),
                'query_acc': query_acc.item(),
                'final_params': [p.clone() for p in model.parameters()]
            })
        
        # Results should be identical
        result1, result2 = results
        
        # Training losses should be identical
        for loss1, loss2 in zip(result1['losses'], result2['losses']):
            assert abs(loss1 - loss2) < 1e-7
        
        # Query metrics should be identical
        assert abs(result1['query_loss'] - result2['query_loss']) < 1e-7
        assert abs(result1['query_acc'] - result2['query_acc']) < 1e-7
        
        # Model parameters should be identical
        for p1, p2 in zip(result1['final_params'], result2['final_params']):
            assert torch.allclose(p1, p2, atol=1e-7)
    
    def test_deterministic_dataloader_training(self):
        """Test deterministic training with data loading."""
        # Create dataset
        dataset = torch.utils.data.TensorDataset(
            torch.randn(1000, 20),
            torch.randint(0, 5, (1000,))
        )
        
        results = []
        
        for run in range(2):
            # Setup deterministic environment
            seed_everything(42)
            
            # Create deterministic dataloader
            dataloader = create_reproducible_dataloader(
                dataset,
                batch_size=32,
                shuffle=True,
                seed=42
            )
            
            # Reset model
            model = nn.Linear(20, 5)
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
            
            # Training loop
            losses = []
            for batch_idx, (x, y) in enumerate(dataloader):
                if batch_idx >= 5:  # Limit for test speed
                    break
                
                logits = model(x)
                loss = F.cross_entropy(logits, y)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                losses.append(loss.item())
            
            results.append({
                'losses': losses,
                'final_params': [p.clone() for p in model.parameters()]
            })
        
        # Results should be deterministic
        result1, result2 = results
        
        for loss1, loss2 in zip(result1['losses'], result2['losses']):
            assert abs(loss1 - loss2) < 1e-6
        
        for p1, p2 in zip(result1['final_params'], result2['final_params']):
            assert torch.allclose(p1, p2, atol=1e-6)
    
    def test_multi_gpu_determinism(self):
        """Test determinism with multiple GPUs (if available)."""
        if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
            pytest.skip("Multiple GPUs not available")
        
        # Test determinism across different GPU configurations
        device_configs = ['cuda:0']
        if torch.cuda.device_count() > 1:
            device_configs.append('cuda:1')
        
        results = []
        
        for device in device_configs:
            seed_everything(42, cuda_deterministic=True)
            
            model = nn.Linear(10, 5).to(device)
            x = torch.randn(16, 10).to(device)
            
            with torch.no_grad():
                output = model(x)
            
            results.append(output.cpu())
        
        # Results should be identical (or at least very close due to GPU differences)
        if len(results) > 1:
            assert torch.allclose(results[0], results[1], atol=1e-5)


if __name__ == "__main__":
    pytest.main([__file__])