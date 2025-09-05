"""
Test script for deterministic benchmarks - standalone validation.
"""
import sys
from pathlib import Path

# Add the few_shot_modules directory to path
few_shot_modules_path = Path(__file__).parent.parent / "src" / "meta_learning" / "meta_learning_modules" / "few_shot_modules"
sys.path.insert(0, str(few_shot_modules_path))

from benchmarks import BenchmarkConfig, benchmark_prototypical_networks, benchmark_maml

def test_deterministic_benchmarks():
    """Test that benchmarks produce deterministic, reproducible results."""
    print("🔬 Testing deterministic few-shot benchmarks...")
    
    # Small config for fast testing
    config = BenchmarkConfig(
        n_way=3, 
        n_shot=2, 
        n_episodes=10,
        random_seed=42
    )
    
    # Test Prototypical Networks
    print("Testing Prototypical Networks...")
    proto_result1 = benchmark_prototypical_networks(config)
    proto_result2 = benchmark_prototypical_networks(config)
    
    # Results should be identical (deterministic)
    assert abs(proto_result1.accuracy - proto_result2.accuracy) < 1e-10, "ProtoNet not deterministic!"
    assert abs(proto_result1.mean_loss - proto_result2.mean_loss) < 1e-10, "ProtoNet loss not deterministic!"
    print(f"✅ ProtoNet deterministic: {proto_result1.accuracy:.4f} accuracy")
    
    # Test MAML
    print("Testing MAML...")
    maml_result1 = benchmark_maml(config)  
    maml_result2 = benchmark_maml(config)
    
    # Results should be identical (deterministic)
    assert abs(maml_result1.accuracy - maml_result2.accuracy) < 1e-10, "MAML not deterministic!"
    assert abs(maml_result1.mean_loss - maml_result2.mean_loss) < 1e-10, "MAML loss not deterministic!"
    print(f"✅ MAML deterministic: {maml_result1.accuracy:.4f} accuracy")
    
    # Test different configuration
    config_alt = BenchmarkConfig(
        n_way=5,  # More challenging
        n_shot=1, 
        n_episodes=5,
        input_dim=32,
        random_seed=9999  # Very different seed
    )
    
    proto_alt = benchmark_prototypical_networks(config_alt)
    print(f"✅ Different configurations tested: {config.n_way}-way vs {config_alt.n_way}-way")
    
    print("🎉 All deterministic benchmark tests passed!")
    
    # Show sample results
    print("\nSample Benchmark Results:")
    print(f"ProtoNet: {proto_result1.accuracy:.3f} ± {(proto_result1.confidence_interval[1] - proto_result1.confidence_interval[0])/2:.3f}")
    print(f"MAML: {maml_result1.accuracy:.3f} ± {(maml_result1.confidence_interval[1] - maml_result1.confidence_interval[0])/2:.3f}")


if __name__ == "__main__":
    test_deterministic_benchmarks()