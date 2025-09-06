"""
Global pytest configuration and fixtures for meta-learning library.

This file contains fixtures that are available to all test modules.
Following 2024 best practices for AI/ML library testing.
"""

import pytest
import torch
import numpy as np
import tempfile
import os
from pathlib import Path
from typing import Tuple, Optional
import warnings

# Import core components
import meta_learning as ml
from meta_learning.data import SyntheticFewShotDataset, make_episodes
from meta_learning.core.episode import Episode


# =============================================================================
# TEST CONFIGURATION & SETUP
# =============================================================================

def pytest_configure(config):
    """Configure pytest with custom markers and settings."""
    config.addinivalue_line(
        "markers", 
        "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", 
        "performance: performance benchmark tests"
    )
    config.addinivalue_line(
        "markers", 
        "integration: integration tests"
    )
    config.addinivalue_line(
        "markers", 
        "unit: unit tests"
    )
    config.addinivalue_line(
        "markers", 
        "regression: regression tests"
    )


@pytest.fixture(scope="session", autouse=True)
def setup_test_environment():
    """Set up consistent test environment for reproducibility."""
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Configure PyTorch for deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Suppress warnings during testing
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    
    # Use CPU for consistent testing across environments
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
        torch.cuda.manual_seed_all(42)


# =============================================================================
# CORE FIXTURES
# =============================================================================

@pytest.fixture(scope="session")
def random_seed():
    """Fixed seed for reproducible tests across all test modules."""
    return 42


@pytest.fixture(scope="session")
def torch_device():
    """Consistent device for all tests (CPU for deterministic behavior)."""
    return torch.device("cpu")


@pytest.fixture(scope="session")
def test_temp_dir():
    """Session-scoped temporary directory for test files."""
    with tempfile.TemporaryDirectory(prefix="meta_learning_tests_") as temp_dir:
        yield Path(temp_dir)


# =============================================================================
# DATA FIXTURES
# =============================================================================

@pytest.fixture
def synthetic_dataset(random_seed):
    """Standard synthetic dataset for general testing."""
    return SyntheticFewShotDataset(
        n_classes=10, 
        dim=64, 
        noise=0.1
    )


@pytest.fixture
def small_synthetic_dataset(random_seed):
    """Small synthetic dataset for quick tests."""
    return SyntheticFewShotDataset(
        n_classes=5, 
        dim=32, 
        noise=0.05
    )


@pytest.fixture
def large_synthetic_dataset(random_seed):
    """Large synthetic dataset for performance testing."""
    return SyntheticFewShotDataset(
        n_classes=20, 
        dim=128, 
        noise=0.1
    )


@pytest.fixture
def image_mode_dataset(random_seed):
    """Image mode synthetic dataset for testing Conv4 compatibility."""
    return SyntheticFewShotDataset(
        n_classes=8,
        dim=64,
        noise=0.1,
        image_mode=True
    )


# =============================================================================
# EPISODE FIXTURES
# =============================================================================

@pytest.fixture
def standard_episode(synthetic_dataset, random_seed):
    """Standard 5-way 3-shot 4-query episode for general testing."""
    support_x, support_y, query_x, query_y = synthetic_dataset.sample_support_query(
        n_way=5, k_shot=3, m_query=4, seed=random_seed
    )
    return Episode(support_x, support_y, query_x, query_y)


@pytest.fixture
def small_episode(random_seed):
    """Minimal 2-way 2-shot 2-query episode for edge case testing."""
    torch.manual_seed(random_seed)
    support_x = torch.randn(4, 16)  # 2 classes * 2 shots
    support_y = torch.tensor([0, 0, 1, 1])
    query_x = torch.randn(4, 16)    # 2 classes * 2 queries
    query_y = torch.tensor([0, 0, 1, 1])
    return Episode(support_x, support_y, query_x, query_y)


@pytest.fixture
def large_episode(random_seed):
    """Large 10-way 5-shot 10-query episode for performance testing."""
    n_way, k_shot, m_query = 10, 5, 10
    dim = 128
    
    torch.manual_seed(random_seed)
    support_x = torch.randn(n_way * k_shot, dim)
    support_y = torch.arange(n_way).repeat_interleave(k_shot)
    query_x = torch.randn(n_way * m_query, dim)
    query_y = torch.arange(n_way).repeat_interleave(m_query)
    
    return Episode(support_x, support_y, query_x, query_y)


@pytest.fixture
def edge_case_episode(random_seed):
    """Episode with edge case scenarios (single shot, etc.)."""
    torch.manual_seed(random_seed)
    # 3-way 1-shot 1-query (minimal viable episode)
    support_x = torch.randn(3, 8)   # 1 shot per class
    support_y = torch.tensor([0, 1, 2])
    query_x = torch.randn(3, 8)     # 1 query per class  
    query_y = torch.tensor([0, 1, 2])
    return Episode(support_x, support_y, query_x, query_y)


@pytest.fixture(params=[2, 5, 10])
def parametrized_n_way_episode(request, random_seed):
    """Parametrized fixture for testing different n-way scenarios."""
    n_way = request.param
    k_shot, m_query = 2, 3
    dim = 32
    
    torch.manual_seed(random_seed)
    support_x = torch.randn(n_way * k_shot, dim)
    support_y = torch.arange(n_way).repeat_interleave(k_shot)
    query_x = torch.randn(n_way * m_query, dim)
    query_y = torch.arange(n_way).repeat_interleave(m_query)
    
    return Episode(support_x, support_y, query_x, query_y)


# =============================================================================
# MODEL FIXTURES  
# =============================================================================

@pytest.fixture
def simple_linear_model(torch_device):
    """Simple linear model for basic testing."""
    model = torch.nn.Sequential(
        torch.nn.Linear(64, 32),
        torch.nn.ReLU(),
        torch.nn.Linear(32, 16),
        torch.nn.ReLU(),
        torch.nn.Linear(16, 5)
    ).to(torch_device)
    return model


@pytest.fixture
def conv4_model(torch_device):
    """Conv4 model for image-based testing."""
    try:
        model = ml.Conv4(num_classes=5).to(torch_device)
        return model
    except (AttributeError, ImportError):
        pytest.skip("Conv4 model not available")


@pytest.fixture
def proto_head():
    """ProtoNet head for prototypical network testing."""
    try:
        return ml.ProtoHead(distance="sqeuclidean", tau=1.0)
    except (AttributeError, ImportError):
        pytest.skip("ProtoHead not available")


# =============================================================================
# ALGORITHM FIXTURES
# =============================================================================

@pytest.fixture
def continual_maml_model(simple_linear_model):
    """ContinualMAML model for continual learning testing."""
    try:
        return ml.ContinualMAML(
            model=simple_linear_model,
            memory_size=100,
            consolidation_strength=1000.0,
            fisher_samples=50
        )
    except (AttributeError, ImportError):
        pytest.skip("ContinualMAML not available")


# =============================================================================
# UTILITY FIXTURES
# =============================================================================

@pytest.fixture
def loss_function():
    """Standard loss function for testing."""
    return torch.nn.CrossEntropyLoss()


@pytest.fixture
def optimizer_factory():
    """Factory function for creating optimizers."""
    def create_optimizer(model, lr=0.001):
        return torch.optim.Adam(model.parameters(), lr=lr)
    return create_optimizer


@pytest.fixture
def mock_dataloader():
    """Mock dataloader for testing training loops."""
    class MockDataLoader:
        def __init__(self, num_batches=3, batch_size=4, dim=32, num_classes=3):
            self.num_batches = num_batches
            self.batch_size = batch_size
            self.dim = dim
            self.num_classes = num_classes
            torch.manual_seed(42)  # Deterministic data
            
        def __iter__(self):
            for _ in range(self.num_batches):
                x = torch.randn(self.batch_size, self.dim)
                y = torch.randint(0, self.num_classes, (self.batch_size,))
                yield x, y
                
        def __len__(self):
            return self.num_batches
    
    return MockDataLoader()


# =============================================================================
# TESTING UTILITIES
# =============================================================================

@pytest.fixture
def assert_tensor_properties():
    """Utility function for asserting tensor properties."""
    def _assert_tensor_properties(tensor, expected_shape=None, expected_dtype=None, 
                                  check_finite=True, check_non_negative=False):
        """Assert common tensor properties."""
        assert isinstance(tensor, torch.Tensor), f"Expected tensor, got {type(tensor)}"
        
        if expected_shape is not None:
            assert tensor.shape == expected_shape, f"Expected shape {expected_shape}, got {tensor.shape}"
            
        if expected_dtype is not None:
            assert tensor.dtype == expected_dtype, f"Expected dtype {expected_dtype}, got {tensor.dtype}"
            
        if check_finite:
            assert torch.isfinite(tensor).all(), "Tensor contains non-finite values"
            
        if check_non_negative:
            assert torch.all(tensor >= 0), "Tensor contains negative values"
            
    return _assert_tensor_properties


@pytest.fixture
def assert_episode_properties():
    """Utility function for asserting episode properties."""
    def _assert_episode_properties(episode, expected_n_way=None, expected_k_shot=None, 
                                   expected_m_query=None, expected_dim=None):
        """Assert common episode properties."""
        assert isinstance(episode, Episode), f"Expected Episode, got {type(episode)}"
        
        # Basic shapes match
        assert episode.support_x.shape[0] == episode.support_y.shape[0]
        assert episode.query_x.shape[0] == episode.query_y.shape[0]
        
        # Same feature dimension
        if episode.support_x.dim() > 1:
            assert episode.support_x.shape[1:] == episode.query_x.shape[1:]
            
        # Labels are proper integers
        assert episode.support_y.dtype == torch.int64
        assert episode.query_y.dtype == torch.int64
        
        # Expected dimensions
        if expected_dim is not None:
            if episode.support_x.dim() == 2:  # Vector data
                assert episode.support_x.shape[1] == expected_dim
            elif episode.support_x.dim() == 4:  # Image data
                assert episode.support_x.shape[1:] == expected_dim  # (C, H, W)
                
        # Expected episode configuration
        if expected_n_way is not None:
            n_classes = len(torch.unique(episode.support_y))
            assert n_classes == expected_n_way
            
        if expected_k_shot is not None and expected_n_way is not None:
            assert episode.support_x.shape[0] == expected_n_way * expected_k_shot
            
        if expected_m_query is not None and expected_n_way is not None:
            assert episode.query_x.shape[0] == expected_n_way * expected_m_query
            
    return _assert_episode_properties


# =============================================================================
# PERFORMANCE TESTING FIXTURES
# =============================================================================

@pytest.fixture
def benchmark_runner():
    """Simple benchmark runner for performance tests."""
    import time
    
    def _benchmark(func, *args, **kwargs):
        """Run function multiple times and return timing statistics."""
        times = []
        for _ in range(5):  # Run 5 times
            start_time = time.time()
            result = func(*args, **kwargs)
            end_time = time.time()
            times.append(end_time - start_time)
        
        return {
            'result': result,
            'mean_time': np.mean(times),
            'std_time': np.std(times),
            'min_time': np.min(times),
            'max_time': np.max(times)
        }
    
    return _benchmark


# =============================================================================
# CLEANUP AND TEARDOWN
# =============================================================================

@pytest.fixture(autouse=True)
def cleanup_after_test():
    """Clean up after each test to prevent state leakage."""
    yield  # Run the test
    
    # Clean up PyTorch state
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    
    # Reset random states (though they should be set in each test)
    torch.manual_seed(42)
    np.random.seed(42)


# =============================================================================
# CONDITIONAL FIXTURES (based on availability)
# =============================================================================

@pytest.fixture
def hardware_config():
    """Hardware configuration if available."""
    try:
        return ml.create_hardware_config(device="cpu")
    except (AttributeError, ImportError):
        pytest.skip("Hardware configuration not available")


@pytest.fixture  
def leakage_guard():
    """Leakage guard if available."""
    try:
        return ml.create_leakage_guard(strict_mode=False)
    except (AttributeError, ImportError):
        pytest.skip("Leakage guard not available")


# =============================================================================
# INTEGRATION TEST FIXTURES
# =============================================================================

@pytest.fixture
def complete_few_shot_setup(synthetic_dataset, proto_head, random_seed):
    """Complete setup for few-shot learning integration tests."""
    # Generate episodes
    episodes = list(make_episodes(
        synthetic_dataset, 
        n_way=5, 
        k_shot=3, 
        m_query=4, 
        episodes=10
    ))
    
    # Simple evaluation function
    def evaluate_fn(episode):
        support_x, support_y, query_x, query_y = episode
        n_classes = len(torch.unique(support_y))
        return torch.randn(query_x.shape[0], n_classes)  # Mock predictions
    
    return {
        'episodes': episodes,
        'model': proto_head,
        'evaluate_fn': evaluate_fn
    }


# =============================================================================
# DEBUGGING AND DEVELOPMENT FIXTURES
# =============================================================================

@pytest.fixture
def debug_mode():
    """Enable debug mode with extra logging and assertions."""
    import logging
    logging.basicConfig(level=logging.DEBUG)
    
    # Return debug utilities
    class DebugUtils:
        @staticmethod
        def print_tensor_stats(tensor, name="tensor"):
            print(f"\n{name} stats:")
            print(f"  Shape: {tensor.shape}")
            print(f"  Dtype: {tensor.dtype}")
            print(f"  Min: {tensor.min().item():.6f}")
            print(f"  Max: {tensor.max().item():.6f}")
            print(f"  Mean: {tensor.mean().item():.6f}")
            print(f"  Std: {tensor.std().item():.6f}")
            print(f"  Finite: {torch.isfinite(tensor).all().item()}")
    
    return DebugUtils()


# =============================================================================
# PYTEST PLUGINS CONFIGURATION
# =============================================================================

# Enable pytest plugins if available
pytest_plugins = [
    "pytest_benchmark",  # For performance testing
    "pytest_mock",       # For mocking
    "pytest_randomly",   # For random test order
]

# Filter out unavailable plugins
available_plugins = []
for plugin in pytest_plugins:
    try:
        __import__(plugin.replace("pytest_", ""))
        available_plugins.append(plugin)
    except ImportError:
        pass

pytest_plugins = available_plugins