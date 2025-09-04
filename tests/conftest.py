"""
🧪 Meta-Learning Test Configuration & Research Validation Suite
==============================================================

This module provides comprehensive testing infrastructure for validating research-accurate
meta-learning algorithm implementations against theoretical properties and empirical benchmarks.

💰 Donations: Help support this research!
   PayPal: https://www.paypal.com/cgi-bin/webscr?cmd=_s-xclick&hosted_button_id=WXQKYYKPHWXHS
   💖 Please consider recurring donations to support continued research validation

🔬 Research Foundation:
======================
Testing framework validates core theoretical properties from:
- Finn et al. (2017): MAML convergence properties and gradient computation
- Snell et al. (2017): Prototypical Networks distance metrics and few-shot accuracy  
- Snell et al. (2024): Test-Time Compute Scaling computational budgets and allocation
- Vinyals et al. (2016): Matching Networks attention mechanisms and embedding quality

🎯 Key Testing Categories:
=========================
• **Property-Based Testing**: Validates theoretical guarantees using Hypothesis
• **Research Accuracy Validation**: Ensures implementations match paper specifications
• **Statistical Rigor**: Proper confidence intervals and significance testing
• **Performance Benchmarking**: Computational complexity and memory usage validation
• **Cross-Algorithm Integration**: Tests for consistent interfaces and data flows

ELI5 Explanation:
================
Think of this like a quality control factory for meta-learning algorithms! 

Just like how a car factory tests every component (brakes, engine, steering) before 
the car hits the road, we test every meta-learning algorithm component to make sure:
- The math works exactly like the research papers said it should
- The algorithms learn quickly from just a few examples (that's the "meta" part!)
- The statistical analysis gives reliable results researchers can trust
- Everything runs fast enough for real-world use

ASCII Test Architecture:
========================
    Research Paper    Implementation    Test Suite
    ┌─────────────┐   ┌─────────────┐   ┌─────────────┐
    │ Finn (2017) │──▶│ MAML Learner│──▶│Property Tests│
    │ θ'=θ-α∇L_τ  │   │     Code    │   │   ✓ Grad OK │
    └─────────────┘   └─────────────┘   └─────────────┘
          │                 │                 │
          ▼                 ▼                 ▼
    ┌─────────────┐   ┌─────────────┐   ┌─────────────┐
    │Theoretical  │   │ Test Data   │   │ Statistical │
    │ Properties  │   │ Generators  │   │ Validation  │
    └─────────────┘   └─────────────┘   └─────────────┘

📊 Testing Framework Architecture:
=================================
1. **Fixture Factories**: Generate realistic meta-learning tasks and datasets
2. **Property Validators**: Ensure mathematical properties hold (e.g., gradient norms)
3. **Statistical Analyzers**: Compute proper confidence intervals for few-shot accuracy
4. **Performance Monitors**: Track computational costs and memory usage
5. **Research Reproducibility**: Validate against published benchmark results

⚡ Test Execution Flow:
======================
Property Test → Algorithm → Statistical Analysis → Research Validation
     │              │            │                      │
     ├─ Generate    ├─ Execute   ├─ Confidence         ├─ Compare vs
     │  test cases  │  algorithm │  intervals          │  paper results
     └─ Validate    └─ Record    └─ Significance       └─ Pass/Fail
        invariants     metrics     testing

Author: Benedict Chen (benedict@benedictchen.com)
Testing Framework: pytest + hypothesis + coverage (2024 research-grade practices)
Research Validation: Mathematical property verification + empirical benchmarking
"""

import pytest
import torch
import numpy as np
from typing import Dict, List, Any, Callable
from unittest.mock import Mock, patch
import tempfile
import shutil
from pathlib import Path
import json
import pickle
from hypothesis import strategies as st, settings, HealthCheck

# Import our modules for testing
import sys
sys.path.insert(0, 'src')

from meta_learning import *
from meta_learning.meta_learning_modules.test_time_compute import TestTimeComputeScaler, TestTimeComputeConfig
from meta_learning.meta_learning_modules.maml_variants import MAMLLearner, MAMLConfig
from meta_learning.meta_learning_modules.few_shot_learning import PrototypicalNetworks, PrototypicalConfig
from meta_learning.meta_learning_modules.continual_meta_learning import OnlineMetaLearner, ContinualMetaConfig
from meta_learning.meta_learning_modules.utils import MetaLearningDataset, TaskConfiguration, EvaluationConfig


# =============================================================================
# PYTEST CONFIGURATION HOOKS
# =============================================================================

def pytest_configure(config):
    """Configure pytest with custom markers and settings."""
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Configure Hypothesis settings
    settings.register_profile("fast", max_examples=20, deadline=1000)
    settings.register_profile("thorough", max_examples=100, deadline=5000)
    settings.register_profile("ci", max_examples=50, deadline=2000)
    
    # Use appropriate profile based on environment
    try:
        profile = config.getoption("--hypothesis-profile", default=None)
        if profile:
            settings.load_profile(profile)
        else:
            settings.load_profile("fast")
    except Exception:
        # Fallback to fast profile if option parsing fails
        settings.load_profile("fast")


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers and organize tests."""
    for item in items:
        # Add markers based on test names and paths
        if "property" in str(item.fspath) or "property" in item.name:
            item.add_marker(pytest.mark.property)
            
        if "integration" in str(item.fspath) or "integration" in item.name:
            item.add_marker(pytest.mark.integration)
            
        if "slow" in item.name or "benchmark" in item.name:
            item.add_marker(pytest.mark.slow)
            
        if "fixme" in item.name.lower() or "solution" in item.name.lower():
            item.add_marker(pytest.mark.fixme_solution)
            
        # Auto-add unit marker for tests in unit directory
        if "unit" in str(item.fspath):
            item.add_marker(pytest.mark.unit)


# =============================================================================
# DEVICE AND HARDWARE FIXTURES
# =============================================================================

@pytest.fixture(scope="session")
def device():
    """Provide computational device (CPU/GPU) for tests."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")  # Apple Silicon
    else:
        return torch.device("cpu")


@pytest.fixture(scope="session")
def use_gpu():
    """Boolean fixture indicating if GPU is available."""
    return torch.cuda.is_available()


# =============================================================================
# BASIC TEST DATA FIXTURES
# =============================================================================

@pytest.fixture
def sample_data():
    """Generate sample data for basic testing."""
    torch.manual_seed(42)
    return {
        'support_data': torch.randn(25, 84),  # 5-way 5-shot
        'support_labels': torch.repeat_interleave(torch.arange(5), 5),
        'query_data': torch.randn(75, 84),    # 5-way 15-query
        'query_labels': torch.repeat_interleave(torch.arange(5), 15),
        'feature_dim': 84,
        'n_way': 5,
        'k_shot': 5,
        'q_query': 15
    }


@pytest.fixture(scope="function")
def synthetic_dataset():
    """Create synthetic dataset for meta-learning tests."""
    # Create synthetic data: 20 classes, 100 samples per class
    n_classes = 20
    samples_per_class = 100
    feature_dim = 64
    
    data_list = []
    labels_list = []
    
    for class_id in range(n_classes):
        # Generate class-specific data with some structure
        class_center = torch.randn(feature_dim) * 2
        class_data = class_center.unsqueeze(0) + torch.randn(samples_per_class, feature_dim) * 0.5
        class_labels = torch.full((samples_per_class,), class_id)
        
        data_list.append(class_data)
        labels_list.append(class_labels)
    
    data = torch.cat(data_list, dim=0)
    labels = torch.cat(labels_list, dim=0)
    
    return data, labels


@pytest.fixture(params=[1, 3, 5, 10])
def k_shot_values(request):
    """Parametrized fixture for different k-shot values."""
    return request.param


@pytest.fixture(params=[3, 5, 10, 20])
def n_way_values(request):
    """Parametrized fixture for different n-way values."""
    return request.param


# =============================================================================
# CONFIGURATION FIXTURES
# =============================================================================

@pytest.fixture
def test_time_compute_config():
    """Basic test-time compute configuration."""
    return TestTimeComputeConfig(
        max_compute_steps=5,
        temperature_scaling=0.1,
        ensemble_size=3,
        compute_strategy="basic"
    )


@pytest.fixture(params=["basic", "snell2024", "akyurek2024", "openai_o1", "hybrid"])
def all_compute_strategies(request):
    """Parametrized fixture for all compute strategies."""
    return TestTimeComputeConfig(
        compute_strategy=request.param,
        max_compute_steps=3,  # Keep low for testing speed
        ensemble_size=2
    )


@pytest.fixture
def maml_config():
    """Basic MAML configuration."""
    return MAMLConfig(
        inner_lr=0.01,
        outer_lr=0.001,
        inner_steps=3,  # Keep low for testing
        meta_batch_size=4
    )


@pytest.fixture(params=["standard", "fomaml", "reptile", "anil", "boil"])
def all_maml_variants(request):
    """Parametrized fixture for all MAML variants."""
    return MAMLConfig(
        maml_variant=request.param,
        inner_steps=2,  # Keep low for testing
        meta_batch_size=2
    )


@pytest.fixture
def prototypical_config():
    """Basic prototypical networks configuration."""
    return PrototypicalConfig(
        embedding_dim=64,
        protonet_variant="research_accurate"
    )


@pytest.fixture(params=["original", "research_accurate", "simple", "enhanced"])
def all_prototypical_variants(request):
    """Parametrized fixture for all prototypical variants."""
    return PrototypicalConfig(
        embedding_dim=32,  # Keep small for testing
        protonet_variant=request.param
    )


@pytest.fixture
def continual_config():
    """Basic continual meta-learning configuration.""" 
    return ContinualMetaConfig(
        memory_size=100,  # Keep small for testing
        adaptation_lr=0.01,
        ewc_method="diagonal",
        fisher_estimation_method="empirical"
    )


@pytest.fixture(params=["diagonal", "full", "evcl", "none"])
def all_ewc_methods(request):
    """Parametrized fixture for all EWC methods."""
    return ContinualMetaConfig(
        ewc_method=request.param,
        memory_size=50,  # Keep small for testing
        adaptation_lr=0.01
    )


# =============================================================================
# MODEL FIXTURES
# =============================================================================

@pytest.fixture
def simple_model():
    """Simple neural network model for testing."""
    return torch.nn.Sequential(
        torch.nn.Linear(84, 64),
        torch.nn.ReLU(),
        torch.nn.Linear(64, 32),
        torch.nn.ReLU(), 
        torch.nn.Linear(32, 5)  # 5-way classification
    )


@pytest.fixture
def encoder_model():
    """Encoder model for few-shot learning."""
    return torch.nn.Sequential(
        torch.nn.Linear(84, 64),
        torch.nn.ReLU(),
        torch.nn.Linear(64, 32)  # Output embedding dimension
    )


@pytest.fixture(scope="session")
def mock_tokenizer():
    """Mock tokenizer for language model tests."""
    tokenizer = Mock()
    tokenizer.encode.return_value = [1, 2, 3, 4, 5]
    tokenizer.decode.return_value = "mock output"
    tokenizer.pad_token_id = 0
    tokenizer.return_value = {
        'input_ids': torch.tensor([[1, 2, 3, 4, 5]]),
        'attention_mask': torch.tensor([[1, 1, 1, 1, 1]])
    }
    return tokenizer


# =============================================================================
# FACTORY FUNCTION FIXTURES
# =============================================================================

@pytest.fixture
def config_factory():
    """Factory for creating various configurations."""
    def _create_config(config_type: str, **kwargs):
        factories = {
            'task': create_basic_task_config,
            'research_task': create_research_accurate_task_config,
            'eval': create_basic_evaluation_config, 
            'research_eval': create_research_accurate_evaluation_config,
            'standard_eval': create_meta_learning_standard_evaluation_config
        }
        
        if config_type not in factories:
            raise ValueError(f"Unknown config type: {config_type}")
            
        return factories[config_type](**kwargs)
    
    return _create_config


@pytest.fixture
def algorithm_factory():
    """Factory for creating algorithm instances."""
    def _create_algorithm(algorithm_type: str, model=None, config=None, **kwargs):
        if model is None:
            model = torch.nn.Sequential(
                torch.nn.Linear(84, 64),
                torch.nn.ReLU(),
                torch.nn.Linear(64, 5)
            )
            
        algorithms = {
            'test_time_compute': lambda: TestTimeComputeScaler(
                model, config or TestTimeComputeConfig(**kwargs)
            ),
            'maml': lambda: MAMLLearner(
                model, config or MAMLConfig(**kwargs)
            ),
            'prototypical': lambda: PrototypicalNetworks(
                model, config or PrototypicalConfig(**kwargs)
            ),
            'continual': lambda: OnlineMetaLearner(
                model, config or ContinualMetaConfig(**kwargs)
            )
        }
        
        if algorithm_type not in algorithms:
            raise ValueError(f"Unknown algorithm type: {algorithm_type}")
            
        return algorithms[algorithm_type]()
    
    return _create_algorithm


# =============================================================================
# TEMPORARY RESOURCES FIXTURES
# =============================================================================

@pytest.fixture
def temp_dir():
    """Temporary directory that gets cleaned up after test."""
    temp_path = tempfile.mkdtemp()
    yield Path(temp_path)
    shutil.rmtree(temp_path)


@pytest.fixture
def temp_file():
    """Temporary file that gets cleaned up after test."""
    fd, temp_path = tempfile.mkstemp(suffix='.json')
    yield Path(temp_path)
    Path(temp_path).unlink(missing_ok=True)


# =============================================================================
# HYPOTHESIS STRATEGIES FOR PROPERTY-BASED TESTING
# =============================================================================

@pytest.fixture(scope="session")
def meta_learning_strategies():
    """Hypothesis strategies for meta-learning property tests."""
    return {
        # Basic data strategies
        'feature_dim': st.integers(min_value=16, max_value=128),
        'n_way': st.integers(min_value=2, max_value=10),
        'k_shot': st.integers(min_value=1, max_value=10),
        'q_query': st.integers(min_value=1, max_value=20),
        
        # Learning rate strategies
        'learning_rates': st.floats(min_value=1e-5, max_value=1e-1),
        
        # Configuration strategies
        'compute_strategies': st.sampled_from([
            "basic", "snell2024", "akyurek2024", "openai_o1", "hybrid"
        ]),
        'maml_variants': st.sampled_from([
            "standard", "fomaml", "reptile", "anil", "boil"
        ]),
        'prototypical_variants': st.sampled_from([
            "original", "research_accurate", "simple", "enhanced"
        ]),
        'ewc_methods': st.sampled_from([
            "diagonal", "full", "evcl", "none"
        ]),
        'fisher_methods': st.sampled_from([
            "empirical", "exact", "kfac"
        ]),
        'ci_methods': st.sampled_from([
            "bootstrap", "t_distribution", "meta_learning_standard", "bca_bootstrap"
        ]),
        'difficulty_methods': st.sampled_from([
            "pairwise_distance", "silhouette", "entropy", "knn"
        ]),
        
        # Tensor strategies
        'small_tensors': st.integers(1, 10).flatmap(
            lambda n: st.integers(1, 20).flatmap(
                lambda d: st.just(torch.randn(n, d))
            )
        ),
        'accuracy_values': st.lists(
            st.floats(min_value=0.0, max_value=1.0), 
            min_size=3, max_size=100
        )
    }


# =============================================================================
# PERFORMANCE AND BENCHMARKING FIXTURES  
# =============================================================================

@pytest.fixture
def benchmark_config():
    """Configuration for performance benchmarking."""
    return {
        'warmup_rounds': 3,
        'measurement_rounds': 10,
        'timeout_seconds': 30,
        'memory_profiling': True
    }


@pytest.fixture
def performance_thresholds():
    """Performance thresholds for regression testing."""
    return {
        'max_training_time': 60.0,  # seconds
        'max_inference_time': 1.0,   # seconds  
        'max_memory_usage': 500,     # MB
        'min_accuracy': 0.6,         # minimum acceptable accuracy
        'max_loss': 2.0              # maximum acceptable loss
    }


# =============================================================================
# MOCK AND STUB FIXTURES
# =============================================================================

@pytest.fixture
def mock_external_dependencies():
    """Mock external dependencies like web APIs, file systems, etc."""
    mocks = {}
    
    # Mock sklearn if not available
    with patch.dict('sys.modules', {
        'sklearn': Mock(),
        'sklearn.metrics': Mock(),
        'sklearn.neighbors': Mock(),
        'sklearn.model_selection': Mock()
    }):
        mocks['sklearn'] = sys.modules['sklearn']
        yield mocks


@pytest.fixture
def research_paper_mocks():
    """Mock research paper validation functions."""
    def mock_validate_research_accuracy(algorithm_name: str, implementation: Any) -> bool:
        # Always return True for testing - would validate against papers in real implementation
        return True
    
    def mock_check_citation_accuracy(citation: str) -> bool:
        # Mock citation checking
        return "et al." in citation and "2017" in citation or "2024" in citation
        
    return {
        'validate_research_accuracy': mock_validate_research_accuracy,
        'check_citation_accuracy': mock_check_citation_accuracy
    }


# =============================================================================
# INTEGRATION TEST FIXTURES
# =============================================================================

@pytest.fixture(scope="session")
def integration_test_suite():
    """End-to-end integration test configuration."""
    return {
        'test_episodes': 10,  # Reduced for testing speed
        'confidence_level': 0.95,
        'min_tasks_per_algorithm': 5,
        'evaluation_metrics': ['accuracy', 'loss', 'adaptation_speed'],
        'algorithms_to_test': [
            'test_time_compute', 'maml', 'prototypical', 'continual'
        ]
    }


# =============================================================================
# CLEAN UP AND FINALIZATION
# =============================================================================

@pytest.fixture(autouse=True)
def reset_random_seeds():
    """Reset random seeds before each test for reproducibility."""
    torch.manual_seed(42)
    np.random.seed(42)
    yield
    # Cleanup after test if needed


@pytest.fixture(autouse=True, scope="function")
def clear_cuda_cache():
    """Clear CUDA cache between tests if using GPU."""
    yield
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# =============================================================================
# HYPOTHESIS CONFIGURATION
# =============================================================================

# Configure Hypothesis for consistent test behavior
settings.register_profile("test", max_examples=20, deadline=2000)
settings.load_profile("test")

# Suppress health checks that may be problematic in testing environment
settings.register_profile("ci", 
    max_examples=10, 
    deadline=1000,
    suppress_health_check=[
        HealthCheck.too_slow,
        HealthCheck.data_too_large
    ]
)