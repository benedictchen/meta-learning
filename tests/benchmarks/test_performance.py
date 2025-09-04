"""
🚀 Performance Benchmarking Tests for Meta-Learning Package
===========================================================

These tests benchmark the performance of meta-learning algorithms across
different configurations, hardware setups, and dataset sizes.

Benchmark Categories:
- Algorithm speed comparisons
- Memory usage profiling
- Scalability testing
- Hardware acceleration benchmarks
- Cross-algorithm performance analysis
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
import time
import psutil
import gc
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass, asdict
import json

# Meta-learning components
from src.meta_learning.meta_learning_modules.few_shot_learning import (
    PrototypicalLearner, PrototypicalConfig
)
from src.meta_learning.meta_learning_modules.maml_variants import (
    MAML, FOMAML, Reptile, MAMLConfig
)
from src.meta_learning.meta_learning_modules.test_time_compute import (
    TestTimeComputeScaler, TestTimeComputeConfig
)
from src.meta_learning.meta_learning_modules.continual_meta_learning import (
    ContinualMetaLearner, OnlineMetaLearner, ContinualConfig, OnlineConfig
)
from src.meta_learning.meta_learning_modules.utils import (
    MetaLearningDataset, DatasetConfig, EvaluationMetrics
)
from src.meta_learning.meta_learning_modules.hardware_utils import (
    HardwareManager, HardwareConfig, get_optimal_batch_size, log_hardware_info
)


@dataclass
class BenchmarkResult:
    """Structured benchmark result."""
    algorithm: str
    configuration: str
    episodes_per_second: float
    mean_accuracy: float
    memory_usage_mb: float
    peak_memory_mb: float
    gpu_utilization: float
    total_time: float
    n_episodes: int
    hardware_config: Dict[str, Any]
    

class PerformanceBenchmarkSuite:
    """Comprehensive performance benchmarking suite."""
    
    def __init__(self):
        self.results: List[BenchmarkResult] = []
        self.baseline_configs = self._create_baseline_configs()
    
    def _create_baseline_configs(self) -> Dict[str, Dict]:
        """Create standardized benchmark configurations."""
        return {
            'small': {
                'n_way': 5, 'k_shot': 5, 'n_query': 10,
                'n_episodes': 20, 'feature_dim': 64, 'hidden_dim': 128
            },
            'medium': {
                'n_way': 10, 'k_shot': 10, 'n_query': 15,
                'n_episodes': 50, 'feature_dim': 128, 'hidden_dim': 256
            },
            'large': {
                'n_way': 20, 'k_shot': 15, 'n_query': 25,
                'n_episodes': 100, 'feature_dim': 256, 'hidden_dim': 512
            }
        }
    
    def _create_encoder(self, config: Dict) -> nn.Module:
        """Create encoder network for benchmarking."""
        return nn.Sequential(
            nn.Linear(config['feature_dim'], config['hidden_dim']),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(config['hidden_dim'], config['hidden_dim']),
            nn.ReLU(),
            nn.Linear(config['hidden_dim'], config['hidden_dim'])
        )
    
    def _create_dataset(self, config: Dict) -> MetaLearningDataset:
        """Create dataset for benchmarking."""
        dataset_config = DatasetConfig(
            n_way=config['n_way'],
            k_shot=config['k_shot'],
            n_query=config['n_query'],
            feature_dim=config['feature_dim'],
            num_classes=config['n_way'] * 4,  # Larger class pool
            episode_length=config['n_episodes']
        )
        return MetaLearningDataset(dataset_config)
    
    def _measure_memory_usage(self) -> Tuple[float, float]:
        """Measure current and peak memory usage in MB."""
        process = psutil.Process()
        memory_info = process.memory_info()
        current_mb = memory_info.rss / 1024 / 1024
        
        # Peak memory approximation
        peak_mb = current_mb  # Simplified for this implementation
        
        return current_mb, peak_mb
    
    def _get_gpu_utilization(self) -> float:
        """Get GPU utilization percentage."""
        if torch.cuda.is_available():
            # Simplified GPU utilization (would need nvidia-ml-py for real implementation)
            return 0.0  # Placeholder
        return 0.0
    
    def benchmark_algorithm(self, algorithm_name: str, learner: Any, config: Dict, 
                          dataset: MetaLearningDataset, hw_manager: HardwareManager) -> BenchmarkResult:
        """Benchmark a specific algorithm."""
        # Clear memory
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Start measurements
        start_memory, _ = self._measure_memory_usage()
        start_time = time.time()
        
        accuracies = []
        
        # Benchmark loop
        for episode_idx in range(config['n_episodes']):
            # Generate episode
            episode_data = dataset.generate_episode()
            support_x, support_y, query_x, query_y = episode_data
            
            # Prepare data
            support_x = hw_manager.prepare_data(support_x)
            support_y = hw_manager.prepare_data(support_y)
            query_x = hw_manager.prepare_data(query_x)
            query_y = hw_manager.prepare_data(query_y)
            
            # Algorithm-specific forward pass
            with hw_manager.autocast_context():
                if hasattr(learner, 'meta_forward'):  # MAML-style
                    loss, adapted_params = learner.meta_forward(support_x, support_y, query_x, query_y)
                    logits = learner.forward_with_params(query_x, adapted_params)
                elif hasattr(learner, 'scale_compute'):  # Test-time compute
                    logits, _ = learner.scale_compute(
                        None, support_x, support_y, query_x  # Placeholder base learner
                    )
                else:  # Standard learner
                    logits = learner(support_x, support_y, query_x)
                
                # Compute accuracy
                predictions = torch.argmax(logits, dim=1)
                accuracy = (predictions == query_y).float().mean().item()
                accuracies.append(accuracy)
        
        # End measurements
        end_time = time.time()
        end_memory, peak_memory = self._measure_memory_usage()
        gpu_util = self._get_gpu_utilization()
        
        # Calculate metrics
        total_time = end_time - start_time
        episodes_per_second = config['n_episodes'] / total_time
        mean_accuracy = np.mean(accuracies)
        memory_usage = end_memory - start_memory
        
        return BenchmarkResult(
            algorithm=algorithm_name,
            configuration=f"{config['n_way']}way_{config['k_shot']}shot",
            episodes_per_second=episodes_per_second,
            mean_accuracy=mean_accuracy,
            memory_usage_mb=memory_usage,
            peak_memory_mb=peak_memory,
            gpu_utilization=gpu_util,
            total_time=total_time,
            n_episodes=config['n_episodes'],
            hardware_config=asdict(hw_manager.config) if hw_manager.config else {}
        )


@pytest.mark.benchmark
class TestAlgorithmBenchmarks:
    """Test suite for algorithm performance benchmarks."""
    
    @pytest.fixture(scope="class")
    def benchmark_suite(self):
        """Initialize benchmark suite."""
        return PerformanceBenchmarkSuite()
    
    @pytest.fixture(params=['small', 'medium'])
    def benchmark_config(self, request, benchmark_suite):
        """Parameterized benchmark configurations."""
        return benchmark_suite.baseline_configs[request.param]
    
    def test_prototypical_networks_benchmark(self, benchmark_suite, benchmark_config):
        """Benchmark Prototypical Networks with different configurations."""
        # Create components
        encoder = benchmark_suite._create_encoder(benchmark_config)
        dataset = benchmark_suite._create_dataset(benchmark_config)
        
        # Hardware configurations to test
        hw_configs = [
            HardwareConfig(use_mixed_precision=False, memory_efficient=True),
            HardwareConfig(use_mixed_precision=False, memory_efficient=False, compile_model=False)
        ]
        
        # Prototypical configurations
        proto_configs = [
            PrototypicalConfig(distance_metric='euclidean'),
            PrototypicalConfig(distance_metric='cosine'),
            PrototypicalConfig(
                distance_metric='euclidean',
                use_uncertainty_aware_distances=True,
                use_task_adaptive_prototypes=True
            )
        ]
        
        results = []
        
        for hw_config in hw_configs:
            hw_manager = HardwareManager(hw_config)
            prepared_encoder = hw_manager.prepare_model(encoder)
            
            for proto_config in proto_configs:
                learner = PrototypicalLearner(prepared_encoder, proto_config)
                config_name = f"proto_{proto_config.distance_metric}"
                if proto_config.use_uncertainty_aware_distances:
                    config_name += "_enhanced"
                
                result = benchmark_suite.benchmark_algorithm(
                    config_name, learner, benchmark_config, dataset, hw_manager
                )
                results.append(result)
                benchmark_suite.results.append(result)
        
        # Validate results
        assert len(results) > 0
        for result in results:
            assert result.episodes_per_second > 0
            assert 0.0 <= result.mean_accuracy <= 1.0
            assert result.total_time > 0
            
        # Print benchmark summary
        print(f"\n🚀 Prototypical Networks Benchmark ({benchmark_config['n_way']}-way):")
        for result in results:
            print(f"  {result.algorithm}: {result.episodes_per_second:.1f} eps/sec, "
                  f"{result.mean_accuracy:.3f} acc, {result.memory_usage_mb:.1f}MB")
    
    def test_maml_variants_benchmark(self, benchmark_suite, benchmark_config):
        """Benchmark MAML and its variants."""
        encoder = benchmark_suite._create_encoder(benchmark_config)
        dataset = benchmark_suite._create_dataset(benchmark_config)
        
        # Hardware setup
        hw_config = HardwareConfig(memory_efficient=True)
        hw_manager = HardwareManager(hw_config)
        prepared_encoder = hw_manager.prepare_model(encoder)
        
        # MAML variants to test
        maml_configs = [
            ('MAML', MAMLConfig(maml_variant='maml', num_inner_steps=3)),
            ('FOMAML', MAMLConfig(maml_variant='fomaml', num_inner_steps=3)),
            ('Reptile', MAMLConfig(maml_variant='reptile', num_inner_steps=3)),
            ('MAML_Enhanced', MAMLConfig(
                maml_variant='maml', 
                num_inner_steps=3,
                use_adaptive_lr=True,
                use_memory_efficient=True
            ))
        ]
        
        results = []
        
        for variant_name, maml_config in maml_configs:
            learner = MAML(prepared_encoder, maml_config)
            
            result = benchmark_suite.benchmark_algorithm(
                variant_name, learner, benchmark_config, dataset, hw_manager
            )
            results.append(result)
            benchmark_suite.results.append(result)
        
        # Validate and compare results
        assert len(results) == len(maml_configs)
        for result in results:
            assert result.episodes_per_second > 0
            assert 0.0 <= result.mean_accuracy <= 1.0
        
        # Performance analysis
        sorted_results = sorted(results, key=lambda x: x.episodes_per_second, reverse=True)
        fastest = sorted_results[0]
        
        print(f"\n🧠 MAML Variants Benchmark ({benchmark_config['n_way']}-way):")
        for result in results:
            speedup = result.episodes_per_second / sorted_results[-1].episodes_per_second
            print(f"  {result.algorithm}: {result.episodes_per_second:.1f} eps/sec "
                  f"({speedup:.1f}x), {result.mean_accuracy:.3f} acc")
    
    def test_test_time_compute_benchmark(self, benchmark_suite, benchmark_config):
        """Benchmark Test-Time Compute scaling strategies."""
        encoder = benchmark_suite._create_encoder(benchmark_config)
        dataset = benchmark_suite._create_dataset(benchmark_config)
        
        # Hardware setup
        hw_config = HardwareConfig(memory_efficient=True)
        hw_manager = HardwareManager(hw_config)
        prepared_encoder = hw_manager.prepare_model(encoder)
        
        # Test-Time Compute configurations
        ttc_configs = [
            ('TTC_Basic', TestTimeComputeConfig(
                compute_strategy='basic',
                base_compute_steps=1,
                max_compute_steps=3
            )),
            ('TTC_Snell2024', TestTimeComputeConfig(
                compute_strategy='snell2024',
                base_compute_steps=3,
                max_compute_steps=8,
                use_process_reward_model=True
            )),
            ('TTC_Hybrid', TestTimeComputeConfig(
                compute_strategy='hybrid',
                use_test_time_training=True,
                use_chain_of_thought=True
            ))
        ]
        
        # Create base learner for TTC
        base_config = PrototypicalConfig(distance_metric='euclidean')
        base_learner = PrototypicalLearner(prepared_encoder, base_config)
        
        results = []
        
        for config_name, ttc_config in ttc_configs:
            scaler = TestTimeComputeScaler(ttc_config)
            # Wrap scaler to make it compatible with benchmark interface
            class TTCWrapper:
                def __init__(self, scaler, base_learner):
                    self.scaler = scaler
                    self.base_learner = base_learner
                
                def scale_compute(self, _, support_x, support_y, query_x):
                    return self.scaler.scale_compute(
                        self.base_learner, support_x, support_y, query_x
                    )
            
            wrapper = TTCWrapper(scaler, base_learner)
            
            result = benchmark_suite.benchmark_algorithm(
                config_name, wrapper, benchmark_config, dataset, hw_manager
            )
            results.append(result)
            benchmark_suite.results.append(result)
        
        # Validate results
        for result in results:
            assert result.episodes_per_second > 0
            assert 0.0 <= result.mean_accuracy <= 1.0
        
        print(f"\n⚡ Test-Time Compute Benchmark ({benchmark_config['n_way']}-way):")
        for result in results:
            print(f"  {result.algorithm}: {result.episodes_per_second:.1f} eps/sec, "
                  f"{result.mean_accuracy:.3f} acc, {result.memory_usage_mb:.1f}MB")
    
    def test_continual_learning_benchmark(self, benchmark_suite, benchmark_config):
        """Benchmark continual learning approaches."""
        # Reduce episodes for continual learning (more expensive)
        smaller_config = benchmark_config.copy()
        smaller_config['n_episodes'] = min(20, benchmark_config['n_episodes'])
        
        encoder = benchmark_suite._create_encoder(smaller_config)
        dataset = benchmark_suite._create_dataset(smaller_config)
        
        # Hardware setup
        hw_config = HardwareConfig(memory_efficient=True)
        hw_manager = HardwareManager(hw_config)
        prepared_encoder = hw_manager.prepare_model(encoder)
        
        # Continual learning configurations
        continual_configs = [
            ('Continual_Basic', ContinualConfig(use_ewc=False, use_memory_bank=False)),
            ('Continual_EWC', ContinualConfig(use_ewc=True, ewc_lambda=1000.0)),
            ('Continual_Memory', ContinualConfig(
                use_ewc=True, 
                use_memory_bank=True,
                memory_size=50
            )),
            ('Online_Meta', OnlineConfig(
                learning_rate=0.01,
                memory_size=30,
                update_frequency=5
            ))
        ]
        
        results = []
        
        for config_name, config in continual_configs:
            if isinstance(config, OnlineConfig):
                learner = OnlineMetaLearner(prepared_encoder, config)
            else:
                learner = ContinualMetaLearner(prepared_encoder, config)
            
            result = benchmark_suite.benchmark_algorithm(
                config_name, learner, smaller_config, dataset, hw_manager
            )
            results.append(result)
            benchmark_suite.results.append(result)
        
        # Validate results
        for result in results:
            assert result.episodes_per_second > 0
            assert 0.0 <= result.mean_accuracy <= 1.0
        
        print(f"\n🌊 Continual Learning Benchmark ({smaller_config['n_way']}-way):")
        for result in results:
            print(f"  {result.algorithm}: {result.episodes_per_second:.1f} eps/sec, "
                  f"{result.mean_accuracy:.3f} acc")


@pytest.mark.benchmark  
@pytest.mark.slow
class TestScalabilityBenchmarks:
    """Test scalability across different problem sizes."""
    
    def test_scalability_across_ways(self, tmp_path):
        """Test how algorithms scale with increasing number of ways."""
        benchmark_suite = PerformanceBenchmarkSuite()
        
        # Test different n-way configurations
        way_configs = [
            {'n_way': 5, 'k_shot': 5, 'n_query': 10, 'n_episodes': 20},
            {'n_way': 10, 'k_shot': 5, 'n_query': 10, 'n_episodes': 20},
            {'n_way': 20, 'k_shot': 5, 'n_query': 10, 'n_episodes': 20}
        ]
        
        # Add consistent dimensions
        for config in way_configs:
            config.update({'feature_dim': 128, 'hidden_dim': 256})
        
        scalability_results = []
        
        for config in way_configs:
            encoder = benchmark_suite._create_encoder(config)
            dataset = benchmark_suite._create_dataset(config)
            
            hw_manager = HardwareManager(HardwareConfig(memory_efficient=True))
            prepared_encoder = hw_manager.prepare_model(encoder)
            
            # Test Prototypical Networks scaling
            proto_config = PrototypicalConfig(distance_metric='euclidean')
            learner = PrototypicalLearner(prepared_encoder, proto_config)
            
            result = benchmark_suite.benchmark_algorithm(
                f"Proto_{config['n_way']}way", learner, config, dataset, hw_manager
            )
            scalability_results.append(result)
        
        # Analyze scaling behavior
        assert len(scalability_results) == len(way_configs)
        
        # Performance should degrade gracefully with scale
        for i in range(1, len(scalability_results)):
            current = scalability_results[i]
            previous = scalability_results[i-1]
            
            # Speed may decrease with more ways, but should still be reasonable
            assert current.episodes_per_second > 0
            # Memory usage should scale reasonably
            assert current.memory_usage_mb >= previous.memory_usage_mb
        
        # Save scalability report
        scalability_data = [asdict(result) for result in scalability_results]
        report_file = tmp_path / "scalability_report.json"
        with open(report_file, 'w') as f:
            json.dump(scalability_data, f, indent=2)
        
        print(f"\n📈 Scalability Analysis:")
        for result in scalability_results:
            n_way = result.configuration.split('_')[0]
            print(f"  {n_way}: {result.episodes_per_second:.1f} eps/sec, "
                  f"{result.memory_usage_mb:.1f}MB")
    
    def test_memory_efficiency_benchmark(self):
        """Test memory efficiency across different configurations."""
        benchmark_suite = PerformanceBenchmarkSuite()
        
        # Memory efficiency configurations
        memory_configs = [
            ('Standard', HardwareConfig(memory_efficient=False)),
            ('Memory_Efficient', HardwareConfig(memory_efficient=True)),
            ('Gradient_Checkpointing', HardwareConfig(
                memory_efficient=True,
                gradient_checkpointing=True
            ))
        ]
        
        # Use medium-sized problem
        config = benchmark_suite.baseline_configs['medium']
        encoder = benchmark_suite._create_encoder(config)
        dataset = benchmark_suite._create_dataset(config)
        
        memory_results = []
        
        for config_name, hw_config in memory_configs:
            hw_manager = HardwareManager(hw_config)
            prepared_encoder = hw_manager.prepare_model(encoder)
            
            proto_config = PrototypicalConfig(distance_metric='euclidean')
            learner = PrototypicalLearner(prepared_encoder, proto_config)
            
            result = benchmark_suite.benchmark_algorithm(
                f"Memory_{config_name}", learner, config, dataset, hw_manager
            )
            memory_results.append(result)
        
        # Validate memory efficiency improvements
        standard = memory_results[0]
        efficient = memory_results[1]
        
        assert len(memory_results) == len(memory_configs)
        # Memory efficient version should use less or similar memory
        assert efficient.memory_usage_mb <= standard.memory_usage_mb * 1.1  # Allow 10% variance
        
        print(f"\n💾 Memory Efficiency Benchmark:")
        for result in memory_results:
            print(f"  {result.algorithm}: {result.memory_usage_mb:.1f}MB, "
                  f"{result.episodes_per_second:.1f} eps/sec")


@pytest.mark.benchmark
class TestHardwareBenchmarks:
    """Hardware-specific performance benchmarks."""
    
    def test_device_comparison_benchmark(self):
        """Compare performance across different device configurations."""
        benchmark_suite = PerformanceBenchmarkSuite()
        config = benchmark_suite.baseline_configs['small']
        
        # Device configurations to test
        device_configs = []
        
        # Always test CPU
        device_configs.append(('CPU', HardwareConfig(device='cpu')))
        
        # Test CUDA if available
        if torch.cuda.is_available():
            device_configs.extend([
                ('CUDA', HardwareConfig(device='cuda')),
                ('CUDA_Mixed_Precision', HardwareConfig(
                    device='cuda', 
                    use_mixed_precision=True
                ))
            ])
        
        # Test MPS if available
        if torch.backends.mps.is_available():
            device_configs.append(('MPS', HardwareConfig(device='mps')))
        
        encoder = benchmark_suite._create_encoder(config)
        dataset = benchmark_suite._create_dataset(config)
        
        device_results = []
        
        for device_name, hw_config in device_configs:
            try:
                hw_manager = HardwareManager(hw_config)
                prepared_encoder = hw_manager.prepare_model(encoder)
                
                proto_config = PrototypicalConfig(distance_metric='euclidean')
                learner = PrototypicalLearner(prepared_encoder, proto_config)
                
                result = benchmark_suite.benchmark_algorithm(
                    f"Proto_{device_name}", learner, config, dataset, hw_manager
                )
                device_results.append(result)
                
            except Exception as e:
                print(f"Warning: {device_name} benchmark failed: {e}")
                continue
        
        # Validate device comparison
        assert len(device_results) > 0
        for result in device_results:
            assert result.episodes_per_second > 0
            assert 0.0 <= result.mean_accuracy <= 1.0
        
        print(f"\n⚡ Hardware Device Benchmark:")
        for result in device_results:
            device = result.algorithm.split('_')[-1]
            print(f"  {device}: {result.episodes_per_second:.1f} eps/sec, "
                  f"{result.mean_accuracy:.3f} acc")
    
    def test_compilation_benchmark(self):
        """Test PyTorch 2.0 compilation performance."""
        if not hasattr(torch, 'compile'):
            pytest.skip("PyTorch compilation not available")
        
        benchmark_suite = PerformanceBenchmarkSuite()
        config = benchmark_suite.baseline_configs['medium']
        
        # Compilation configurations
        compile_configs = [
            ('No_Compile', HardwareConfig(compile_model=False)),
            ('Compiled', HardwareConfig(compile_model=True))
        ]
        
        encoder = benchmark_suite._create_encoder(config)
        dataset = benchmark_suite._create_dataset(config)
        
        compile_results = []
        
        for config_name, hw_config in compile_configs:
            try:
                hw_manager = HardwareManager(hw_config)
                prepared_encoder = hw_manager.prepare_model(encoder)
                
                proto_config = PrototypicalConfig(distance_metric='euclidean')
                learner = PrototypicalLearner(prepared_encoder, proto_config)
                
                result = benchmark_suite.benchmark_algorithm(
                    f"Proto_{config_name}", learner, config, dataset, hw_manager
                )
                compile_results.append(result)
                
            except Exception as e:
                print(f"Warning: {config_name} compilation benchmark failed: {e}")
                continue
        
        print(f"\n🔥 Compilation Benchmark:")
        for result in compile_results:
            mode = result.algorithm.split('_')[-1]
            print(f"  {mode}: {result.episodes_per_second:.1f} eps/sec")


if __name__ == "__main__":
    # Run with: pytest tests/benchmarks/test_performance.py -v -m benchmark
    pytest.main([__file__, "-v", "-m", "benchmark"])