"""
Deterministic Few-Shot Learning Benchmarks
==========================================

Reproducible benchmarks for few-shot learning algorithms with frozen datasets
and deterministic evaluation metrics. Ensures consistent results across runs
for research reproducibility.

Based on user feedback: "Add deterministic benchmarks for reproducible evaluation."
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass
import random
from pathlib import Path

# Import when used as module, fallback for direct execution
try:
    from .reference_kernels import (
        Episode, create_episode_from_raw, reference_prototypical_episode,
        reference_maml_step, ReferenceMAMLLearner
    )
    from .maml_core import create_maml_trainer, InnerLoopConfig
except ImportError:
    # For direct execution
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent))
    from reference_kernels import (
        Episode, create_episode_from_raw, reference_prototypical_episode,
        reference_maml_step, ReferenceMAMLLearner
    )
    from maml_core import create_maml_trainer, InnerLoopConfig


@dataclass
class BenchmarkConfig:
    """Configuration for deterministic benchmarks."""
    n_way: int = 5
    n_shot: int = 1  
    n_query: int = 15
    n_episodes: int = 100
    input_dim: int = 64
    random_seed: int = 42
    dataset_name: str = "synthetic"
    algorithm: str = "protonet"
    
    def __post_init__(self):
        """Validate configuration."""
        assert self.n_way > 0, "n_way must be positive"
        assert self.n_shot > 0, "n_shot must be positive" 
        assert self.n_query > 0, "n_query must be positive"
        assert self.n_episodes > 0, "n_episodes must be positive"


@dataclass 
class BenchmarkResult:
    """Results from benchmark evaluation."""
    accuracy: float
    confidence_interval: Tuple[float, float]
    episode_accuracies: List[float]
    mean_loss: float
    std_loss: float
    config: BenchmarkConfig
    algorithm_name: str
    
    def summary(self) -> str:
        """Generate human-readable summary."""
        return f"""
        Benchmark Results - {self.algorithm_name}
        =======================================
        Dataset: {self.config.dataset_name}
        Task: {self.config.n_way}-way {self.config.n_shot}-shot
        Episodes: {self.config.n_episodes}
        
        Accuracy: {self.accuracy:.3f} ± {(self.confidence_interval[1] - self.confidence_interval[0])/2:.3f}
        95% CI: [{self.confidence_interval[0]:.3f}, {self.confidence_interval[1]:.3f}]
        Mean Loss: {self.mean_loss:.4f} ± {self.std_loss:.4f}
        """


class DeterministicDataGenerator:
    """
    Deterministic synthetic data generator for few-shot benchmarks.
    
    Uses fixed random seeds to ensure identical data across runs.
    """
    
    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.rng = np.random.RandomState(config.random_seed)
        self.torch_gen = torch.Generator()
        self.torch_gen.manual_seed(config.random_seed)
        
    def generate_class_centers(self) -> torch.Tensor:
        """Generate deterministic class centers."""
        # Use fixed seed for reproducible class centers
        centers = self.rng.randn(self.config.n_way, self.config.input_dim).astype(np.float32)
        return torch.from_numpy(centers)
    
    def generate_episode(self, episode_id: int) -> Episode:
        """
        Generate deterministic episode with fixed seed.
        
        Args:
            episode_id: Unique episode identifier for reproducible generation
            
        Returns:
            Episode with deterministic data
        """
        # Set episode-specific seed for reproducibility
        episode_seed = self.config.random_seed + episode_id
        episode_rng = np.random.RandomState(episode_seed)
        
        # Generate class centers
        centers = episode_rng.randn(self.config.n_way, self.config.input_dim)
        
        # Generate support set
        support_x = []
        support_y = []
        
        for class_id in range(self.config.n_way):
            for shot in range(self.config.n_shot):
                # Add Gaussian noise around class center
                noise = episode_rng.randn(self.config.input_dim) * 0.1
                example = centers[class_id] + noise
                support_x.append(example)
                support_y.append(class_id)
        
        # Generate query set
        query_x = []
        query_y = []
        
        for class_id in range(self.config.n_way):
            for query in range(self.config.n_query):
                # Add different Gaussian noise for query examples
                noise = episode_rng.randn(self.config.input_dim) * 0.1  
                example = centers[class_id] + noise
                query_x.append(example)
                query_y.append(class_id)
        
        # Convert to tensors
        support_x = torch.tensor(np.array(support_x), dtype=torch.float32)
        support_y = torch.tensor(support_y, dtype=torch.long)
        query_x = torch.tensor(np.array(query_x), dtype=torch.float32)
        query_y = torch.tensor(query_y, dtype=torch.long)
        
        return create_episode_from_raw(support_x, support_y, query_x, query_y)


class DeterministicEncoder(nn.Module):
    """Deterministic encoder with fixed initialization."""
    
    def __init__(self, input_dim: int, output_dim: int, seed: int = 123):
        super().__init__()
        # Deterministic initialization
        torch.manual_seed(seed)
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, output_dim * 2),
            nn.ReLU(),
            nn.Linear(output_dim * 2, output_dim),
            nn.ReLU(),
            nn.Linear(output_dim, output_dim)
        )
        
    def forward(self, x):
        return self.encoder(x)


def compute_accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    """Compute classification accuracy."""
    predictions = torch.argmax(logits, dim=1)
    correct = (predictions == labels).float()
    return correct.mean().item()


def compute_confidence_interval(accuracies: List[float], confidence: float = 0.95) -> Tuple[float, float]:
    """Compute confidence interval for accuracies."""
    accuracies = np.array(accuracies)
    mean_acc = np.mean(accuracies)
    std_acc = np.std(accuracies)
    n = len(accuracies)
    
    # Use t-distribution for small samples
    from scipy import stats
    t_val = stats.t.ppf((1 + confidence) / 2, n - 1)
    margin = t_val * std_acc / np.sqrt(n)
    
    return (mean_acc - margin, mean_acc + margin)


def benchmark_prototypical_networks(config: BenchmarkConfig) -> BenchmarkResult:
    """
    Benchmark Prototypical Networks with deterministic evaluation.
    
    Uses reference implementation for mathematical correctness.
    """
    # Deterministic setup
    torch.manual_seed(config.random_seed)
    np.random.seed(config.random_seed)
    random.seed(config.random_seed)
    
    # Create deterministic encoder
    encoder = DeterministicEncoder(
        input_dim=config.input_dim,
        output_dim=64,
        seed=config.random_seed
    )
    
    # Generate deterministic dataset
    data_generator = DeterministicDataGenerator(config)
    
    episode_accuracies = []
    episode_losses = []
    
    for episode_id in range(config.n_episodes):
        # Generate deterministic episode
        episode = data_generator.generate_episode(episode_id)
        
        # Run reference prototypical networks
        logits = reference_prototypical_episode(
            support_x=episode.support_x,
            support_y=episode.support_y,
            query_x=episode.query_x,
            encoder=encoder,
            distance="sqeuclidean",
            tau=1.0
        )
        
        # Compute metrics
        accuracy = compute_accuracy(logits, episode.query_y)
        loss = F.cross_entropy(logits, episode.query_y).item()
        
        episode_accuracies.append(accuracy)
        episode_losses.append(loss)
    
    # Compute statistics
    mean_accuracy = np.mean(episode_accuracies)
    confidence_interval = compute_confidence_interval(episode_accuracies)
    mean_loss = np.mean(episode_losses)
    std_loss = np.std(episode_losses)
    
    return BenchmarkResult(
        accuracy=mean_accuracy,
        confidence_interval=confidence_interval,
        episode_accuracies=episode_accuracies,
        mean_loss=mean_loss,
        std_loss=std_loss,
        config=config,
        algorithm_name="Prototypical Networks (Reference)"
    )


def benchmark_maml(config: BenchmarkConfig) -> BenchmarkResult:
    """
    Benchmark MAML with deterministic evaluation.
    
    Uses reference implementation for mathematical correctness.
    """
    # Deterministic setup
    torch.manual_seed(config.random_seed)
    np.random.seed(config.random_seed)
    random.seed(config.random_seed)
    
    # Create deterministic model
    model = nn.Sequential(
        nn.Linear(config.input_dim, 64),
        nn.ReLU(),
        nn.Linear(64, 32),
        nn.ReLU(),
        nn.Linear(32, config.n_way)
    )
    
    # Initialize with fixed seed
    torch.manual_seed(config.random_seed)
    for layer in model:
        if isinstance(layer, nn.Linear):
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)
    
    # Generate deterministic dataset
    data_generator = DeterministicDataGenerator(config)
    
    episode_accuracies = []
    episode_losses = []
    
    for episode_id in range(config.n_episodes):
        # Generate deterministic episode
        episode = data_generator.generate_episode(episode_id)
        
        # Run reference MAML inner loop
        query_loss = reference_maml_step(
            model=model,
            support_x=episode.support_x,
            support_y=episode.support_y,
            query_x=episode.query_x,
            query_y=episode.query_y,
            inner_lr=0.01,
            first_order=False
        )
        
        # For accuracy, we need to get the adapted model's logits
        # This requires re-implementing the inner step with logit access
        # Inner adaptation (requires grad)
        support_logits = model(episode.support_x)
        support_loss = F.cross_entropy(support_logits, episode.support_y)
        
        # Compute gradients
        grads = torch.autograd.grad(
            support_loss, model.parameters(),
            create_graph=True, allow_unused=True
        )
        
        # Functional update
        adapted_params = {}
        for (name, param), grad in zip(model.named_parameters(), grads):
            if grad is not None:
                adapted_params[name] = param - 0.01 * grad
            else:
                adapted_params[name] = param
        
        # Query evaluation with adapted parameters
        with torch.no_grad():
            query_logits = torch.func.functional_call(model, adapted_params, episode.query_x)
            
        # Compute metrics
        accuracy = compute_accuracy(query_logits, episode.query_y)
        loss = query_loss.item()
        
        episode_accuracies.append(accuracy)
        episode_losses.append(loss)
    
    # Compute statistics
    mean_accuracy = np.mean(episode_accuracies)
    confidence_interval = compute_confidence_interval(episode_accuracies)
    mean_loss = np.mean(episode_losses)
    std_loss = np.std(episode_losses)
    
    return BenchmarkResult(
        accuracy=mean_accuracy,
        confidence_interval=confidence_interval,
        episode_accuracies=episode_accuracies,
        mean_loss=mean_loss,
        std_loss=std_loss,
        config=config,
        algorithm_name="MAML (Reference)"
    )


def run_comprehensive_benchmark(
    algorithms: Optional[List[str]] = None,
    configs: Optional[List[BenchmarkConfig]] = None
) -> Dict[str, List[BenchmarkResult]]:
    """
    Run comprehensive deterministic benchmark suite.
    
    Args:
        algorithms: List of algorithms to benchmark ["protonet", "maml"]
        configs: List of benchmark configurations to test
        
    Returns:
        Dictionary mapping algorithm names to results lists
    """
    if algorithms is None:
        algorithms = ["protonet", "maml"]
    
    if configs is None:
        # Standard benchmark configurations
        configs = [
            BenchmarkConfig(n_way=5, n_shot=1, n_episodes=100),
            BenchmarkConfig(n_way=5, n_shot=5, n_episodes=100),
            BenchmarkConfig(n_way=10, n_shot=1, n_episodes=100),
        ]
    
    results = {}
    
    for algorithm in algorithms:
        results[algorithm] = []
        
        for config in configs:
            config.algorithm = algorithm
            
            print(f"Running {algorithm} benchmark: {config.n_way}-way {config.n_shot}-shot...")
            
            if algorithm == "protonet":
                result = benchmark_prototypical_networks(config)
            elif algorithm == "maml":
                result = benchmark_maml(config)
            else:
                raise ValueError(f"Unknown algorithm: {algorithm}")
            
            results[algorithm].append(result)
            print(f"✅ {algorithm}: {result.accuracy:.3f} ± {(result.confidence_interval[1] - result.confidence_interval[0])/2:.3f}")
    
    return results


def save_benchmark_results(results: Dict[str, List[BenchmarkResult]], filename: str):
    """Save benchmark results to file."""
    import json
    
    # Convert results to JSON-serializable format
    json_results = {}
    for algorithm, result_list in results.items():
        json_results[algorithm] = []
        for result in result_list:
            json_results[algorithm].append({
                'accuracy': result.accuracy,
                'confidence_interval': result.confidence_interval,
                'mean_loss': result.mean_loss,
                'std_loss': result.std_loss,
                'config': {
                    'n_way': result.config.n_way,
                    'n_shot': result.config.n_shot,
                    'n_query': result.config.n_query,
                    'n_episodes': result.config.n_episodes,
                    'random_seed': result.config.random_seed,
                    'algorithm': result.config.algorithm
                }
            })
    
    with open(filename, 'w') as f:
        json.dump(json_results, f, indent=2)


if __name__ == "__main__":
    # Quick validation of deterministic benchmarks
    print("🔬 Running deterministic few-shot benchmark validation...")
    
    # Test deterministic data generation
    config = BenchmarkConfig(n_way=3, n_shot=2, n_episodes=5)
    generator = DeterministicDataGenerator(config)
    
    # Generate same episode twice - should be identical
    episode1 = generator.generate_episode(0)
    episode2 = generator.generate_episode(0)
    
    assert torch.allclose(episode1.support_x, episode2.support_x), "Episode data not deterministic!"
    assert torch.equal(episode1.support_y, episode2.support_y), "Episode labels not deterministic!"
    print("✅ Deterministic data generation validated")
    
    # Test prototypical networks benchmark
    proto_result = benchmark_prototypical_networks(config)
    assert 0.0 <= proto_result.accuracy <= 1.0, "Invalid accuracy range"
    assert len(proto_result.episode_accuracies) == config.n_episodes
    print("✅ Prototypical Networks benchmark validated")
    
    # Test MAML benchmark  
    maml_result = benchmark_maml(config)
    assert 0.0 <= maml_result.accuracy <= 1.0, "Invalid accuracy range"
    assert len(maml_result.episode_accuracies) == config.n_episodes
    print("✅ MAML benchmark validated")
    
    print("🎉 Deterministic benchmarks validated!")
    print(f"ProtoNet accuracy: {proto_result.accuracy:.3f}")
    print(f"MAML accuracy: {maml_result.accuracy:.3f}")