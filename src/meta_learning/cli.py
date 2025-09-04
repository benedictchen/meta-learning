#!/usr/bin/env python3
"""
📋 Cli
=======

🔬 Research Foundation:  
======================
Based on meta-learning and few-shot learning research:
- Finn, C., Abbeel, P. & Levine, S. (2017). "Model-Agnostic Meta-Learning for Fast Adaptation"
- Snell, J., Swersky, K. & Zemel, R. (2017). "Prototypical Networks for Few-shot Learning"
- Nichol, A., Achiam, J. & Schulman, J. (2018). "On First-Order Meta-Learning Algorithms"
🎯 ELI5 Summary:
This file is an important component in our AI research system! Like different organs 
in your body that work together to keep you healthy, this file has a specific job that 
helps the overall algorithm work correctly and efficiently.

🧪 Technical Details:
===================
Implementation details and technical specifications for this component.
Designed to work seamlessly within the research framework while
maintaining high performance and accuracy standards.

📋 Component Integration:
========================
    ┌──────────┐
    │   This   │
    │Component │ ←→ Other Components
    └──────────┘
         ↑↓
    System Integration

"""
"""
Meta-Learning Research CLI

Command-line interface for running meta-learning algorithm benchmarks
on standard few-shot learning datasets (Omniglot, miniImageNet, CIFAR-FS).

Provides research-accurate implementations following established protocols
from MAML (Finn et al. 2017), Prototypical Networks (Snell et al. 2017),
and related meta-learning literature.
"""

import argparse
import sys
import torch
import torch.nn as nn
from typing import Dict, Any, Optional
import numpy as np

try:
    from .meta_learning_modules import (
        TestTimeComputeScaler,
        MAMLLearner,
        OnlineMetaLearner,
        MetaLearningDataset,
        TaskConfiguration,
        TestTimeComputeConfig,
        MAMLConfig,
        OnlineMetaConfig
    )
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Ensure meta-learning package is properly installed")
    sys.exit(1)


def load_benchmark_dataset(dataset_name: str, n_way: int = 5, k_shot: int = 1) -> tuple:
    """
    Load standard few-shot learning benchmark datasets.
    
    Args:
        dataset_name: One of ['omniglot', 'miniimagenet', 'cifar_fs', 'tiered_imagenet']
        n_way: Number of classes per episode (standard: 5)
        k_shot: Number of shots per class (standard: 1 or 5)
    
    Returns:
        (support_set, support_labels, query_set, query_labels)
    """
    # Try torchmeta first (most research-accurate)
    try:
        import torchmeta
        from torchmeta.datasets import Omniglot, MiniImagenet, CIFAR_FS, TieredImagenet
        from torchmeta.transforms import Categorical, ClassSplitter
        from torchmeta.utils.data import BatchMetaDataLoader
        
        dataset_map = {
            'omniglot': Omniglot,
            'miniimagenet': MiniImagenet, 
            'cifar_fs': CIFAR_FS,
            'tiered_imagenet': TieredImagenet
        }
        
        if dataset_name not in dataset_map:
            raise ValueError(f"Unsupported dataset: {dataset_name}")
            
        dataset_class = dataset_map[dataset_name]
        
        # Standard few-shot learning episode configuration
        dataset = dataset_class(
            "data", num_classes_per_task=n_way, meta_train=True,
            transform=None, target_transform=Categorical(n_way),
            download=True
        )
        
        # Use standard meta-learning protocol
        dataset = ClassSplitter(dataset, shuffle=True, num_train_per_class=k_shot, num_test_per_class=15)
        dataloader = BatchMetaDataLoader(dataset, batch_size=1, num_workers=0)
        
        # Get one episode
        batch = next(iter(dataloader))
        train_inputs, train_targets = batch['train']
        test_inputs, test_targets = batch['test']
        
        # Reshape for episode format
        support_set = train_inputs.squeeze(0)  # [n_way * k_shot, ...]
        support_labels = train_targets.squeeze(0)
        query_set = test_inputs.squeeze(0) 
        query_labels = test_targets.squeeze(0)
        
        print(f"✅ Loaded {dataset_name}: {n_way}-way {k_shot}-shot episode")
        return support_set, support_labels, query_set, query_labels
        
    except ImportError:
        print("⚠️  torchmeta not available. Install: pip install torchmeta")
        print("Falling back to torchvision approximation...")
        
        # Fallback to torchvision with episode simulation
        return _load_torchvision_episode(dataset_name, n_way, k_shot)


def _load_torchvision_episode(dataset_name: str, n_way: int, k_shot: int) -> tuple:
    """Approximate few-shot episode using torchvision datasets."""
    import torchvision
    import torchvision.transforms as transforms
    
    transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    dataset_map = {
        'omniglot': None,  # Not available in torchvision
        'miniimagenet': None,  # Not available in torchvision  
        'cifar_fs': torchvision.datasets.CIFAR10,
        'tiered_imagenet': None
    }
    
    if dataset_name not in dataset_map or dataset_map[dataset_name] is None:
        raise RuntimeError(f"Dataset {dataset_name} requires torchmeta. Install: pip install torchmeta")
    
    dataset_class = dataset_map[dataset_name]
    dataset = dataset_class(root='data', train=True, transform=transform, download=True)
    
    # Simulate episode by sampling classes
    unique_classes = list(range(10))  # CIFAR-10 classes
    selected_classes = np.random.choice(unique_classes, n_way, replace=False)
    
    support_data, support_labels = [], []
    query_data, query_labels = [], []
    
    for new_label, orig_class in enumerate(selected_classes):
        # Get samples from this class
        class_indices = [i for i, (_, label) in enumerate(dataset) if label == orig_class]
        selected_indices = np.random.choice(class_indices, k_shot + 15, replace=False)
        
        # Support set (k_shot samples)
        for idx in selected_indices[:k_shot]:
            img, _ = dataset[idx]
            support_data.append(img)
            support_labels.append(new_label)
            
        # Query set (15 samples following standard protocol)
        for idx in selected_indices[k_shot:]:
            img, _ = dataset[idx]
            query_data.append(img)
            query_labels.append(new_label)
    
    support_set = torch.stack(support_data)
    support_labels = torch.tensor(support_labels)
    query_set = torch.stack(query_data)
    query_labels = torch.tensor(query_labels)
    
    print(f"✅ Simulated {dataset_name} episode: {n_way}-way {k_shot}-shot")
    return support_set, support_labels, query_set, query_labels


def run_prototypical_benchmark(support_set, support_labels, query_set, query_labels) -> Dict[str, float]:
    """
    Run Prototypical Networks benchmark (Snell et al. 2017, NIPS).
    
    Implements the standard protocol: c_k = 1/|S_k| Σ f_φ(x_i) for x_i ∈ S_k
    """
    from .meta_learning_modules.few_shot_learning import PrototypicalNetworks
    from .meta_learning_modules.few_shot_modules.configurations import PrototypicalConfig
    
    config = PrototypicalConfig(
        protonet_variant="research_accurate",
        use_squared_euclidean=True,  # Following Snell et al. 2017
        distance_temperature=1.0
    )
    
    # Simple CNN feature extractor for demonstration
    feature_extractor = nn.Sequential(
        nn.Conv2d(support_set.size(1), 64, 3, padding=1),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d((4, 4)),
        nn.Flatten(),
        nn.Linear(64 * 4 * 4, 128)
    )
    
    protonet = PrototypicalNetworks(feature_extractor, config)
    
    # Run evaluation
    with torch.no_grad():
        logits = protonet(support_set, support_labels, query_set)
        predictions = logits.argmax(dim=-1)
        accuracy = (predictions == query_labels).float().mean().item()
    
    return {"accuracy": accuracy, "method": "Prototypical Networks (Snell et al. 2017)"}


def run_maml_benchmark(support_set, support_labels, query_set, query_labels) -> Dict[str, float]:
    """
    Run MAML benchmark (Finn et al. 2017, ICML).
    
    Implements gradient-based meta-learning: φ_i = θ - α∇_θ L_T_i(f_θ)
    """
    from .meta_learning_modules.maml_variants import MAMLLearner
    from .meta_learning_modules.maml_variants import MAMLConfig
    
    config = MAMLConfig(
        inner_lr=0.01,
        inner_steps=5,
        maml_variant="standard"  # Original MAML
    )
    
    # Simple model for demonstration
    model = nn.Sequential(
        nn.Conv2d(support_set.size(1), 32, 3, padding=1),
        nn.ReLU(), 
        nn.AdaptiveAvgPool2d((4, 4)),
        nn.Flatten(),
        nn.Linear(32 * 4 * 4, len(torch.unique(support_labels)))
    )
    
    maml = MAMLLearner(model, config)
    
    # Run inner loop adaptation
    adapted_model = maml.adapt_to_task(support_set, support_labels)
    
    # Test on query set
    with torch.no_grad():
        logits = adapted_model(query_set)
        predictions = logits.argmax(dim=-1)
        accuracy = (predictions == query_labels).float().mean().item()
    
    return {"accuracy": accuracy, "method": "MAML (Finn et al. 2017)"}


def run_test_time_compute_benchmark(support_set, support_labels, query_set, query_labels) -> Dict[str, float]:
    """Run test-time compute scaling benchmark following recent research."""
    config = TestTimeComputeConfig(
        use_process_reward_model=True,
        max_compute_budget=10,
        confidence_threshold=0.8
    )
    
    # Simple base model
    base_model = nn.Sequential(
        nn.Conv2d(support_set.size(1), 32, 3, padding=1),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d((4, 4)), 
        nn.Flatten(),
        nn.Linear(32 * 4 * 4, len(torch.unique(support_labels)))
    )
    
    scaler = TestTimeComputeScaler(base_model, config)
    
    # Run test-time scaling
    predictions, metrics = scaler.scale_compute(support_set, support_labels, query_set)
    pred_labels = predictions.argmax(dim=-1)
    accuracy = (pred_labels == query_labels).float().mean().item()
    
    return {
        "accuracy": accuracy, 
        "compute_used": metrics.get("compute_steps_used", 0),
        "method": "Test-Time Compute Scaling"
    }


def main():
    """Main CLI entry point for meta-learning benchmarks."""
    parser = argparse.ArgumentParser(
        description="Meta-Learning Research CLI - Standard few-shot learning benchmarks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m meta_learning.cli --dataset omniglot --method prototypical
  python -m meta_learning.cli --dataset cifar_fs --method maml --shots 5
  python -m meta_learning.cli --method all  # Run all benchmarks
        """
    )
    
    parser.add_argument(
        "--dataset", 
        choices=["omniglot", "miniimagenet", "cifar_fs", "tiered_imagenet"],
        default="omniglot",
        help="Few-shot learning benchmark dataset"
    )
    
    parser.add_argument(
        "--method",
        choices=["prototypical", "maml", "test_time_compute", "all"],
        default="all", 
        help="Meta-learning method to benchmark"
    )
    
    parser.add_argument(
        "--n-way", type=int, default=5,
        help="Number of classes per episode (default: 5)"
    )
    
    parser.add_argument(
        "--k-shot", type=int, default=1,
        help="Number of shots per class (default: 1)"
    )
    
    args = parser.parse_args()
    
    print("🧠 Meta-Learning Research CLI")
    print(f"Dataset: {args.dataset} ({args.n_way}-way {args.k_shot}-shot)")
    print("-" * 50)
    
    try:
        # Load benchmark data
        support_set, support_labels, query_set, query_labels = load_benchmark_dataset(
            args.dataset, args.n_way, args.k_shot
        )
        
        # Run benchmarks
        results = []
        
        if args.method in ["prototypical", "all"]:
            print("\n🎯 Running Prototypical Networks...")
            result = run_prototypical_benchmark(support_set, support_labels, query_set, query_labels)
            results.append(result)
            print(f"   Accuracy: {result['accuracy']:.3f}")
        
        if args.method in ["maml", "all"]:
            print("\n🔄 Running MAML...")
            result = run_maml_benchmark(support_set, support_labels, query_set, query_labels)
            results.append(result)
            print(f"   Accuracy: {result['accuracy']:.3f}")
        
        if args.method in ["test_time_compute", "all"]:
            print("\n⚡ Running Test-Time Compute...")
            result = run_test_time_compute_benchmark(support_set, support_labels, query_set, query_labels)
            results.append(result)
            print(f"   Accuracy: {result['accuracy']:.3f} (Compute: {result['compute_used']})")
        
        # Summary
        print("\n" + "=" * 50)
        print("📊 BENCHMARK RESULTS")
        print("=" * 50)
        for result in results:
            print(f"{result['method']:.<40} {result['accuracy']:.3f}")
        
        return 0
        
    except Exception as e:
        print(f"❌ Benchmark failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())