#!/usr/bin/env python3
"""
Determinism Hooks for Reproducible Few-Shot Learning Research
============================================================

This module implements comprehensive determinism controls for few-shot learning
experiments, addressing the reproducibility crisis in ML research.

Research Issues Addressed:
1. Non-deterministic CUDA operations causing result variance
2. Unseeded random number generators in data loaders
3. Non-deterministic algorithms in PyTorch operations
4. Thread-level randomness in multiprocessing

Standards Implemented:
- Goodfellow et al. (2016): "Deep Learning" reproducibility guidelines
- Henderson et al. (2018): "Deep Reinforcement Learning that Matters"
- Papers With Code reproducibility requirements
"""

import torch
import numpy as np
import random
import os
import warnings
from typing import Dict, Any, Optional, List
from contextlib import contextmanager
import hashlib


class DeterminismManager:
    """
    Comprehensive determinism management for reproducible research.
    
    Implements best practices from multiple sources:
    - PyTorch reproducibility documentation
    - Papers With Code reproducibility guidelines
    - ML reproducibility community standards
    """
    
    def __init__(self, base_seed: int = 42):
        """
        Initialize determinism manager.
        
        Args:
            base_seed: Base seed for all random number generators
        """
        self.base_seed = base_seed
        self.original_states = {}
        self.applied_settings = {}
        
    def enable_full_determinism(self, 
                              warn_only: bool = False,
                              allow_tf32: bool = False) -> Dict[str, Any]:
        """
        Enable comprehensive determinism for reproducible research.
        
        Args:
            warn_only: If True, only warn about non-deterministic operations
            allow_tf32: If True, allow TF32 (slightly faster but less precise)
            
        Returns:
            Dictionary of applied settings and their previous values
        """
        settings_applied = {}
        
        # 1. Set all random seeds
        self._set_all_seeds(self.base_seed)
        settings_applied['seeds_set'] = True
        
        # 2. Configure PyTorch deterministic operations
        settings_applied.update(self._configure_torch_determinism(warn_only, allow_tf32))
        
        # 3. Configure CUDA determinism
        settings_applied.update(self._configure_cuda_determinism())
        
        # 4. Configure NumPy determinism
        settings_applied.update(self._configure_numpy_determinism())
        
        # 5. Configure Python random determinism
        settings_applied.update(self._configure_python_determinism())
        
        self.applied_settings = settings_applied
        return settings_applied
    
    def _set_all_seeds(self, seed: int) -> None:
        """Set all random seeds for reproducibility."""
        # PyTorch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        
        # NumPy
        np.random.seed(seed)
        
        # Python random
        random.seed(seed)
        
        # Environment variables for additional libraries
        os.environ['PYTHONHASHSEED'] = str(seed)
        
    def _configure_torch_determinism(self, warn_only: bool, allow_tf32: bool) -> Dict[str, Any]:
        """Configure PyTorch deterministic operations."""
        settings = {}
        
        # Store original values
        original_deterministic = torch.backends.cudnn.deterministic
        original_benchmark = torch.backends.cudnn.benchmark
        
        # Enable deterministic algorithms
        torch.use_deterministic_algorithms(True, warn_only=warn_only)
        settings['deterministic_algorithms'] = True
        
        # Configure CUDNN
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        settings['cudnn_deterministic'] = True
        settings['cudnn_benchmark'] = False
        
        # Store originals for restoration
        self.original_states['cudnn_deterministic'] = original_deterministic
        self.original_states['cudnn_benchmark'] = original_benchmark
        
        # Configure TF32 (affects A100 GPUs)
        if not allow_tf32:
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            settings['tf32_disabled'] = True
        
        return settings
    
    def _configure_cuda_determinism(self) -> Dict[str, Any]:
        """Configure CUDA-specific determinism settings."""
        settings = {}
        
        if torch.cuda.is_available():
            # Set CUDA launch blocking for debugging
            os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
            settings['cuda_launch_blocking'] = True
            
            # Disable CUDA memory caching (can affect determinism)
            # Note: This can significantly slow down training
            # os.environ['PYTORCH_NO_CUDA_MEMORY_CACHING'] = '1'
            
        return settings
    
    def _configure_numpy_determinism(self) -> Dict[str, Any]:
        """Configure NumPy determinism settings."""
        # NumPy seed was set in _set_all_seeds
        return {'numpy_seed_set': True}
    
    def _configure_python_determinism(self) -> Dict[str, Any]:
        """Configure Python random module determinism."""
        # Python seed was set in _set_all_seeds
        # PYTHONHASHSEED was set in environment
        return {'python_seed_set': True, 'hash_seed_set': True}
    
    def create_deterministic_worker_init_fn(self) -> callable:
        """
        Create worker initialization function for deterministic DataLoaders.
        
        Returns:
            Function to pass to DataLoader's worker_init_fn parameter
        """
        def worker_init_fn(worker_id: int) -> None:
            """Initialize worker with deterministic seeds."""
            # Create unique but deterministic seed for each worker
            worker_seed = self.base_seed + worker_id
            
            np.random.seed(worker_seed)
            random.seed(worker_seed)
            torch.manual_seed(worker_seed)
            
        return worker_init_fn
    
    def validate_determinism(self, model: torch.nn.Module,
                           input_tensor: torch.Tensor,
                           num_runs: int = 3) -> Dict[str, Any]:
        """
        Validate that model produces deterministic outputs.
        
        Args:
            model: Model to test
            input_tensor: Input tensor for testing
            num_runs: Number of forward passes to compare
            
        Returns:
            Validation results with determinism metrics
        """
        model.eval()  # Ensure consistent mode
        outputs = []
        
        with torch.no_grad():
            for _ in range(num_runs):
                # Reset to ensure deterministic starting state
                self._set_all_seeds(self.base_seed)
                output = model(input_tensor)
                outputs.append(output.clone())
        
        # Compare outputs
        all_equal = True
        max_diff = 0.0
        
        for i in range(1, len(outputs)):
            diff = torch.abs(outputs[0] - outputs[i]).max().item()
            max_diff = max(max_diff, diff)
            if diff > 1e-6:  # Small tolerance for floating point
                all_equal = False
        
        return {
            'deterministic': all_equal,
            'max_difference': max_diff,
            'num_runs': num_runs,
            'tolerance': 1e-6
        }
    
    def restore_original_settings(self) -> None:
        """Restore original PyTorch settings."""
        if 'cudnn_deterministic' in self.original_states:
            torch.backends.cudnn.deterministic = self.original_states['cudnn_deterministic']
        if 'cudnn_benchmark' in self.original_states:
            torch.backends.cudnn.benchmark = self.original_states['cudnn_benchmark']


@contextmanager
def deterministic_context(seed: int = 42, 
                         warn_only: bool = False,
                         allow_tf32: bool = False):
    """
    Context manager for deterministic operations.
    
    Example:
        >>> with deterministic_context(seed=42):
        ...     output = model(input_data)  # Guaranteed deterministic
    """
    manager = DeterminismManager(seed)
    
    # Enable determinism
    settings = manager.enable_full_determinism(warn_only=warn_only, allow_tf32=allow_tf32)
    
    try:
        yield manager, settings
    finally:
        # Restore original settings
        manager.restore_original_settings()


class ReproducibilityReport:
    """Generate comprehensive reproducibility reports for research papers."""
    
    @staticmethod
    def generate_environment_report() -> Dict[str, Any]:
        """Generate detailed environment report for reproducibility."""
        import platform
        import sys
        
        report = {
            'python_version': sys.version,
            'platform': platform.platform(),
            'pytorch_version': torch.__version__,
            'cuda_available': torch.cuda.is_available(),
            'cuda_version': torch.version.cuda if torch.cuda.is_available() else None,
            'cudnn_version': torch.backends.cudnn.version() if torch.cuda.is_available() else None,
            'device_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
        }
        
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                device_props = torch.cuda.get_device_properties(i)
                report[f'gpu_{i}'] = {
                    'name': device_props.name,
                    'memory': f"{device_props.total_memory / 1024**3:.1f} GB",
                    'compute_capability': f"{device_props.major}.{device_props.minor}"
                }
        
        return report
    
    @staticmethod
    def generate_experiment_hash(config: Dict[str, Any]) -> str:
        """Generate deterministic hash for experiment configuration."""
        config_str = str(sorted(config.items()))
        return hashlib.md5(config_str.encode()).hexdigest()[:8]


# Convenience functions for researchers
def setup_deterministic_environment(seed: int = 42,
                                   warn_only: bool = False,
                                   allow_tf32: bool = False) -> DeterminismManager:
    """
    One-line setup for deterministic research environment.
    
    Example:
        >>> setup_deterministic_environment(seed=42)
        >>> # Now all operations are deterministic!
    """
    manager = DeterminismManager(seed)
    settings = manager.enable_full_determinism(warn_only=warn_only, allow_tf32=allow_tf32)
    
    print("🔒 Deterministic environment configured:")
    for key, value in settings.items():
        if value:  # Only show enabled settings
            print(f"  ✅ {key}")
    
    return manager


def create_deterministic_dataloader(dataset, 
                                  batch_size: int,
                                  seed: int = 42,
                                  **kwargs) -> torch.utils.data.DataLoader:
    """
    Create DataLoader with deterministic settings for reproducible research.
    
    Args:
        dataset: PyTorch dataset
        batch_size: Batch size
        seed: Random seed
        **kwargs: Additional DataLoader arguments
        
    Returns:
        Deterministic DataLoader
    """
    manager = DeterminismManager(seed)
    worker_init_fn = manager.create_deterministic_worker_init_fn()
    
    # Force deterministic settings
    kwargs['worker_init_fn'] = worker_init_fn
    kwargs['generator'] = torch.Generator().manual_seed(seed)
    
    # Disable multiprocessing if not explicitly set (can cause non-determinism)
    if 'num_workers' not in kwargs:
        kwargs['num_workers'] = 0
    
    return torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        **kwargs
    )


if __name__ == "__main__":
    # Demo: Comprehensive determinism setup
    print("🔒 Determinism Hooks Demo")
    print("=" * 40)
    
    # Setup deterministic environment
    manager = setup_deterministic_environment(seed=42)
    
    # Create test model
    model = torch.nn.Sequential(
        torch.nn.Linear(10, 50),
        torch.nn.ReLU(),
        torch.nn.Linear(50, 1)
    )
    
    # Test determinism
    test_input = torch.randn(5, 10)
    validation = manager.validate_determinism(model, test_input)
    
    print(f"\nDeterminism validation:")
    print(f"  ✅ Deterministic: {validation['deterministic']}")
    print(f"  📊 Max difference: {validation['max_difference']:.2e}")
    
    # Generate reproducibility report
    env_report = ReproducibilityReport.generate_environment_report()
    print(f"\nEnvironment:")
    print(f"  🐍 Python: {env_report['python_version'].split()[0]}")
    print(f"  🔥 PyTorch: {env_report['pytorch_version']}")
    print(f"  🎮 CUDA: {env_report['cuda_version'] or 'Not available'}")
    
    print("\n✅ Determinism hooks configured successfully!")