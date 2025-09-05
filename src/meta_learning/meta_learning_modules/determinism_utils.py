"""
Determinism Utilities for Reproducible Meta-Learning
====================================================

Author: Benedict Chen (benedict@benedictchen.com)

Comprehensive determinism utilities for reproducible meta-learning experiments.
Critical for research where episode sampling and model behavior must be 
exactly reproducible across runs.

Key Components:
1. Complete seeding of all random sources
2. Deterministic CUDA operations (with performance warnings)
3. DataLoader worker seeding
4. Verification utilities to check determinism
5. Context managers for scoped deterministic operations

Research Importance:
Reproducibility is fundamental to scientific validity. In meta-learning,
non-deterministic episode sampling can make results unreproducible even
with identical hyperparameters.
"""

import torch
import random
import numpy as np
import os
import hashlib
from typing import Optional, Dict, Any, Callable, List, Tuple
from contextlib import contextmanager
import warnings
import logging
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)


@dataclass
class DeterminismConfig:
    """Configuration for deterministic operations."""
    seed: int = 42
    cuda_deterministic: bool = True
    cuda_benchmark: bool = False
    torch_use_deterministic_algorithms: bool = True
    warn_performance_impact: bool = True
    
    def __post_init__(self):
        if self.cuda_deterministic and self.warn_performance_impact:
            warnings.warn(
                "Deterministic CUDA operations may significantly reduce performance. "
                "Set warn_performance_impact=False to disable this warning."
            )


def seed_everything(
    seed: int, 
    cuda_deterministic: bool = True,
    benchmark: bool = False,
    use_deterministic_algorithms: bool = True
) -> None:
    """
    Set seeds for all random number generators for perfect reproducibility.
    
    This is the gold-standard seeding function that covers all sources of
    randomness in PyTorch-based meta-learning experiments.
    
    Args:
        seed: Random seed to set everywhere
        cuda_deterministic: Use deterministic CUDA operations (slower)
        benchmark: Enable CuDNN benchmark (faster but non-deterministic)
        use_deterministic_algorithms: Force PyTorch to use deterministic algorithms
    """
    # Python random module
    random.seed(seed)
    
    # NumPy random
    np.random.seed(seed)
    
    # PyTorch random
    torch.manual_seed(seed)
    
    # PyTorch CUDA random (if available)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # For multi-GPU setups
        
        # CuDNN settings for determinism
        if cuda_deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = benchmark
        else:
            torch.backends.cudnn.deterministic = False
            torch.backends.cudnn.benchmark = True
    
    # PyTorch deterministic algorithms (requires PyTorch 1.8+)
    if use_deterministic_algorithms and hasattr(torch, 'use_deterministic_algorithms'):
        try:
            torch.use_deterministic_algorithms(True)
        except RuntimeError as e:
            warnings.warn(f"Could not enable deterministic algorithms: {e}")
    
    # Environment variables for additional determinism
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    logger.info(f"Set global seed to {seed} with cuda_deterministic={cuda_deterministic}")


def create_worker_init_fn(base_seed: int) -> Callable[[int], None]:
    """
    Create worker initialization function for DataLoader determinism.
    
    Critical for reproducible episode generation when using multiple
    DataLoader workers. Each worker gets a unique but deterministic seed.
    
    Args:
        base_seed: Base seed to derive worker seeds from
        
    Returns:
        Worker initialization function for DataLoader
        
    Usage:
        >>> worker_init = create_worker_init_fn(42)
        >>> dataloader = DataLoader(dataset, num_workers=4, worker_init_fn=worker_init)
    """
    def worker_init_fn(worker_id: int) -> None:
        # Each worker gets a unique but deterministic seed
        worker_seed = base_seed + worker_id
        np.random.seed(worker_seed)
        random.seed(worker_seed)
        torch.manual_seed(worker_seed)
    
    return worker_init_fn


class DeterminismChecker:
    """
    Utility to verify that operations are actually deterministic.
    
    Records intermediate states and can verify that repeated runs
    produce identical results.
    """
    
    def __init__(self, tolerance: float = 1e-8):
        self.tolerance = tolerance
        self.recorded_states = {}
        self.run_count = 0
        
    def record_state(self, name: str, tensor: torch.Tensor) -> None:
        """Record tensor state for later comparison."""
        if name not in self.recorded_states:
            self.recorded_states[name] = []
        
        # Store detached copy
        self.recorded_states[name].append(tensor.detach().cpu().clone())
        
    def verify_determinism(self) -> Dict[str, bool]:
        """
        Verify that all recorded states are deterministic across runs.
        
        Returns:
            Dictionary mapping state names to whether they're deterministic
        """
        results = {}
        
        for name, states in self.recorded_states.items():
            if len(states) < 2:
                results[name] = None  # Can't verify with < 2 runs
                continue
                
            # Compare first state with all others
            first_state = states[0]
            is_deterministic = all(
                torch.allclose(first_state, state, atol=self.tolerance)
                for state in states[1:]
            )
            results[name] = is_deterministic
            
        return results
    
    def assert_deterministic(self) -> None:
        """Assert that all recorded states are deterministic."""
        results = self.verify_determinism()
        
        failed_states = [name for name, is_det in results.items() 
                        if is_det is False]
        
        if failed_states:
            raise AssertionError(
                f"Non-deterministic behavior detected in states: {failed_states}"
            )
            
        logger.info(f"Determinism verified for {len(results)} states")


@contextmanager
def deterministic_context(config: DeterminismConfig):
    """
    Context manager for scoped deterministic operations.
    
    Applies deterministic settings within the context and restores
    original settings when exiting.
    
    Usage:
        >>> config = DeterminismConfig(seed=42, cuda_deterministic=True)
        >>> with deterministic_context(config):
        ...     # All operations here are deterministic
        ...     result = run_experiment()
    """
    # Store original settings
    original_state = {}
    
    if torch.cuda.is_available():
        original_state['cudnn_deterministic'] = torch.backends.cudnn.deterministic
        original_state['cudnn_benchmark'] = torch.backends.cudnn.benchmark
    
    # Store original algorithm mode if available
    if hasattr(torch, 'are_deterministic_algorithms_enabled'):
        original_state['deterministic_algorithms'] = torch.are_deterministic_algorithms_enabled()
    
    try:
        # Apply deterministic settings
        seed_everything(
            config.seed,
            config.cuda_deterministic,
            config.cuda_benchmark,
            config.torch_use_deterministic_algorithms
        )
        
        yield config
        
    finally:
        # Restore original settings
        if torch.cuda.is_available():
            if 'cudnn_deterministic' in original_state:
                torch.backends.cudnn.deterministic = original_state['cudnn_deterministic']
            if 'cudnn_benchmark' in original_state:
                torch.backends.cudnn.benchmark = original_state['cudnn_benchmark']
        
        if ('deterministic_algorithms' in original_state and 
            hasattr(torch, 'use_deterministic_algorithms')):
            torch.use_deterministic_algorithms(original_state['deterministic_algorithms'])


def compute_tensor_hash(tensor: torch.Tensor) -> str:
    """
    Compute deterministic hash of tensor contents.
    
    Useful for verifying that tensors are identical across runs
    without storing the entire tensor.
    """
    # Convert to numpy for consistent hashing
    np_array = tensor.detach().cpu().numpy()
    return hashlib.md5(np_array.tobytes()).hexdigest()


def verify_episode_determinism(
    episode_generator: Callable[[int], Any],
    seed: int,
    num_episodes: int = 100
) -> Tuple[bool, List[str]]:
    """
    Verify that episode generation is deterministic.
    
    Args:
        episode_generator: Function that takes seed and generates episodes
        seed: Seed to test with
        num_episodes: Number of episodes to generate for testing
        
    Returns:
        (is_deterministic, episode_hashes) where episode_hashes can be
        compared across runs to verify determinism
    """
    # Generate episodes twice with same seed
    seed_everything(seed)
    episodes_1 = [episode_generator(i) for i in range(num_episodes)]
    
    seed_everything(seed)  # Reset to same seed
    episodes_2 = [episode_generator(i) for i in range(num_episodes)]
    
    # Compare episode hashes
    hashes_1 = []
    hashes_2 = []
    
    for ep1, ep2 in zip(episodes_1, episodes_2):
        # Convert episode to tensors if needed
        if hasattr(ep1, '__iter__') and not isinstance(ep1, (str, torch.Tensor)):
            # Assume episode is tuple/list of tensors
            ep1_tensors = [t for t in ep1 if isinstance(t, torch.Tensor)]
            ep2_tensors = [t for t in ep2 if isinstance(t, torch.Tensor)]
            
            hash1 = hashlib.md5(
                b''.join(compute_tensor_hash(t).encode() for t in ep1_tensors)
            ).hexdigest()
            hash2 = hashlib.md5(
                b''.join(compute_tensor_hash(t).encode() for t in ep2_tensors)
            ).hexdigest()
        else:
            # Single tensor episode
            hash1 = compute_tensor_hash(ep1)
            hash2 = compute_tensor_hash(ep2)
            
        hashes_1.append(hash1)
        hashes_2.append(hash2)
    
    # Check if all hashes match
    is_deterministic = hashes_1 == hashes_2
    
    return is_deterministic, hashes_1


class ReproducibilityManager:
    """
    High-level manager for experiment reproducibility.
    
    Handles seeding, configuration storage, and verification
    of reproducible meta-learning experiments.
    """
    
    def __init__(self, config: DeterminismConfig):
        self.config = config
        self.experiment_hash = self._compute_config_hash()
        
    def _compute_config_hash(self) -> str:
        """Compute hash of configuration for experiment identification."""
        config_str = str(sorted(asdict(self.config).items()))
        return hashlib.md5(config_str.encode()).hexdigest()[:8]
    
    def setup_experiment(self, extra_info: Optional[Dict[str, Any]] = None) -> str:
        """
        Setup experiment with full reproducibility.
        
        Returns:
            Experiment ID for tracking and reproduction
        """
        # Apply seeding
        seed_everything(
            self.config.seed,
            self.config.cuda_deterministic,
            not self.config.cuda_deterministic,  # benchmark opposite of deterministic
            self.config.torch_use_deterministic_algorithms
        )
        
        # Log experiment setup
        experiment_id = f"exp_{self.experiment_hash}_{self.config.seed}"
        
        logger.info(f"Setup reproducible experiment: {experiment_id}")
        logger.info(f"Config: {asdict(self.config)}")
        
        if extra_info:
            logger.info(f"Extra info: {extra_info}")
            
        return experiment_id
    
    def verify_setup(self, operation: Callable[[], torch.Tensor], 
                    num_runs: int = 3) -> bool:
        """
        Verify that the reproducibility setup works correctly.
        
        Runs the given operation multiple times and checks for identical results.
        """
        results = []
        
        for run in range(num_runs):
            # Re-setup for each run
            self.setup_experiment()
            result = operation()
            results.append(compute_tensor_hash(result))
        
        # All runs should produce identical hashes
        is_reproducible = len(set(results)) == 1
        
        if is_reproducible:
            logger.info(f"Reproducibility verification passed ({num_runs} runs)")
        else:
            logger.warning(f"Reproducibility verification failed! Got {len(set(results))} different results")
            
        return is_reproducible


if __name__ == "__main__":
    # Test determinism utilities
    print("Determinism Utilities Test")
    print("=" * 40)
    
    # Test basic seeding
    config = DeterminismConfig(seed=42, warn_performance_impact=False)
    
    def test_operation():
        return torch.randn(10)
    
    # Test with reproducibility manager
    manager = ReproducibilityManager(config)
    experiment_id = manager.setup_experiment()
    print(f"Experiment ID: {experiment_id}")
    
    # Verify reproducibility
    is_reproducible = manager.verify_setup(test_operation, num_runs=5)
    print(f"Reproducibility test: {'✓ PASS' if is_reproducible else '✗ FAIL'}")
    
    # Test determinism checker
    checker = DeterminismChecker()
    
    # Record states from multiple runs
    for run in range(3):
        seed_everything(42)
        result = torch.randn(5)
        checker.record_state("test_tensor", result)
    
    # Verify determinism
    determinism_results = checker.verify_determinism()
    print(f"Determinism check: {determinism_results}")
    
    try:
        checker.assert_deterministic()
        print("✓ Determinism assertion passed")
    except AssertionError as e:
        print(f"✗ Determinism assertion failed: {e}")
    
    # Test episode determinism verification
    def dummy_episode_generator(episode_id):
        return torch.randn(10), torch.randint(0, 5, (10,))
    
    is_det, hashes = verify_episode_determinism(
        dummy_episode_generator, seed=42, num_episodes=10
    )
    print(f"Episode determinism: {'✓ PASS' if is_det else '✗ FAIL'}")
    
    # Test context manager
    print("\nTesting deterministic context manager...")
    
    with deterministic_context(config):
        result1 = torch.randn(5)
        
    with deterministic_context(config):  # Same seed
        result2 = torch.randn(5)
        
    context_deterministic = torch.allclose(result1, result2)
    print(f"Context manager test: {'✓ PASS' if context_deterministic else '✗ FAIL'}")
    
    print("\n✓ All determinism utilities tested")