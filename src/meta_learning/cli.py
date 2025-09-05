#!/usr/bin/env python3
"""
Meta-Learning CLI Tool (mlfew)
=============================

Production-ready command-line interface for few-shot learning research.

Usage:
    mlfew fit --dataset omniglot --algorithm protonet --n-way 5 --k-shot 1
    mlfew eval --model checkpoints/protonet_omniglot.pt --dataset omniglot
    mlfew benchmark --datasets omniglot,miniimagenet --algorithms protonet,maml
"""

import argparse
import sys
import os
from pathlib import Path
from typing import Dict, Any, Optional, List
import json
import time

import torch
import torch.nn as nn
import numpy as np
from rich.console import Console
from rich.table import Table
from rich.progress import track
from rich import print as rprint

# Try to import core functionality
try:
    from . import __version__
    from .algos.protonet import ProtoHead, fit_episode, make_episode
    from .core.math_utils import pairwise_sqeuclidean
    from .meta_learning_modules.few_shot_learning import PrototypicalNetworks
    from .meta_learning_modules.maml_variants import MAMLLearner
    from .meta_learning_modules.utils_modules import few_shot_accuracy, TaskConfiguration
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Ensure meta-learning-toolkit is properly installed")
    sys.exit(1)

console = Console()


def create_dataset_loader(dataset_name: str, split: str = "train") -> Any:
    """Create dataset loader for standard few-shot datasets."""
    
    if dataset_name == "omniglot":
        try:
            from torchvision import datasets, transforms
            
            transform = transforms.Compose([
                transforms.Resize((28, 28)),
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])
            
            # Use MNIST as Omniglot proxy for demo
            dataset = datasets.MNIST(
                root="./data", 
                train=(split == "train"), 
                download=True, 
                transform=transform
            )
            return dataset
            
        except ImportError:
            console.print("❌ torchvision required for dataset loading", style="red")
            return None
            
    elif dataset_name == "miniimagenet":
        console.print("⚠️  miniImageNet requires manual download", style="yellow")
        console.print("Using CIFAR-10 as proxy...")
        
        try:
            from torchvision import datasets, transforms
            
            transform = transforms.Compose([
                transforms.Resize((84, 84)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
            
            dataset = datasets.CIFAR10(
                root="./data",
                train=(split == "train"),
                download=True, 
                transform=transform
            )
            return dataset
            
        except ImportError:
            console.print("❌ torchvision required for dataset loading", style="red")
            return None
    
    else:
        console.print(f"❌ Unknown dataset: {dataset_name}", style="red")
        return None


def sample_episode(dataset: Any, n_way: int = 5, k_shot: int = 1, n_query: int = 15) -> tuple:
    """Sample a few-shot learning episode from dataset."""
    
    # Group samples by class
    class_to_samples = {}
    for i, (data, label) in enumerate(dataset):
        if label not in class_to_samples:
            class_to_samples[label] = []
        class_to_samples[label].append((data, label))
    
    # Select n_way classes
    available_classes = list(class_to_samples.keys())
    selected_classes = np.random.choice(available_classes, n_way, replace=False)
    
    support_data, support_labels = [], []
    query_data, query_labels = [], []
    
    for new_label, orig_class in enumerate(selected_classes):
        class_samples = class_to_samples[orig_class]
        
        # Sample support and query
        total_needed = k_shot + n_query
        if len(class_samples) < total_needed:
            selected = np.random.choice(len(class_samples), total_needed, replace=True)
        else:
            selected = np.random.choice(len(class_samples), total_needed, replace=False)
        
        # Add to support set
        for i in selected[:k_shot]:
            data, _ = class_samples[i]
            support_data.append(data)
            support_labels.append(new_label)
        
        # Add to query set
        for i in selected[k_shot:]:
            data, _ = class_samples[i] 
            query_data.append(data)
            query_labels.append(new_label)
    
    return (
        torch.stack(support_data),
        torch.tensor(support_labels),
        torch.stack(query_data), 
        torch.tensor(query_labels)
    )


def create_model(algorithm: str, input_shape: tuple, n_way: int) -> nn.Module:
    """Create model for specified algorithm."""
    
    if algorithm == "protonet":
        # Simple feature extractor
        if len(input_shape) == 3:  # RGB
            feature_extractor = nn.Sequential(
                nn.Conv2d(input_shape[0], 64, 3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.MaxPool2d(2),
                
                nn.Conv2d(64, 64, 3, padding=1),
                nn.BatchNorm2d(64), 
                nn.ReLU(),
                nn.MaxPool2d(2),
                
                nn.Conv2d(64, 64, 3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
            )
        else:  # Grayscale
            feature_extractor = nn.Sequential(
                nn.Conv2d(1, 64, 3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.MaxPool2d(2),
                
                nn.Conv2d(64, 64, 3, padding=1), 
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
            )
        
        return ProtoHead(feature_extractor)
        
    elif algorithm == "maml":
        # Simple CNN for MAML
        if len(input_shape) == 3:
            model = nn.Sequential(
                nn.Conv2d(input_shape[0], 32, 3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(32, 32, 3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Flatten(),
                nn.Linear(32 * (input_shape[1]//4) * (input_shape[2]//4), 128),
                nn.ReLU(),
                nn.Linear(128, n_way)
            )
        else:
            model = nn.Sequential(
                nn.Conv2d(1, 32, 3, padding=1),
                nn.ReLU(), 
                nn.MaxPool2d(2),
                nn.Flatten(),
                nn.Linear(32 * 14 * 14, 128),
                nn.ReLU(),
                nn.Linear(128, n_way)
            )
        
        return model
    
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")


def fit_command(args: argparse.Namespace) -> Dict[str, Any]:
    """Execute fit command - train a model."""
    console.print(f"🏋️ Training {args.algorithm} on {args.dataset}", style="bold blue")
    
    # Load dataset
    dataset = create_dataset_loader(args.dataset, "train")
    if dataset is None:
        return {"success": False, "error": "Failed to load dataset"}
    
    # Sample episode
    support_x, support_y, query_x, query_y = sample_episode(
        dataset, args.n_way, args.k_shot, args.n_query
    )
    
    console.print(f"📊 Episode: {args.n_way}-way {args.k_shot}-shot")
    console.print(f"   Support: {support_x.shape}, Query: {query_x.shape}")
    
    # Create model
    input_shape = support_x.shape[1:]
    model = create_model(args.algorithm, input_shape, args.n_way)
    
    # Train model
    start_time = time.time()
    
    if args.algorithm == "protonet":
        # Use fit_episode from algos.protonet
        try:
            logits = model(support_x, support_y, query_x)
            accuracy = (logits.argmax(-1) == query_y).float().mean().item()
        except Exception as e:
            console.print(f"❌ Training failed: {e}", style="red")
            return {"success": False, "error": str(e)}
    
    elif args.algorithm == "maml":
        # Simple MAML training
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()
        
        model.train()
        for _ in range(10):  # Inner steps
            optimizer.zero_grad()
            logits = model(support_x)
            loss = criterion(logits, support_y)
            loss.backward()
            optimizer.step()
        
        # Evaluate
        model.eval()
        with torch.no_grad():
            logits = model(query_x)
            accuracy = (logits.argmax(-1) == query_y).float().mean().item()
    
    train_time = time.time() - start_time
    
    # Save model
    os.makedirs("checkpoints", exist_ok=True)
    checkpoint_path = f"checkpoints/{args.algorithm}_{args.dataset}.pt"
    torch.save({
        "model_state_dict": model.state_dict(),
        "algorithm": args.algorithm,
        "dataset": args.dataset,
        "n_way": args.n_way,
        "k_shot": args.k_shot,
        "accuracy": accuracy,
    }, checkpoint_path)
    
    # Report results
    result_table = Table(title="Training Results")
    result_table.add_column("Metric", style="cyan")
    result_table.add_column("Value", style="green")
    
    result_table.add_row("Algorithm", args.algorithm)
    result_table.add_row("Dataset", args.dataset)
    result_table.add_row("N-way", str(args.n_way))
    result_table.add_row("K-shot", str(args.k_shot))
    result_table.add_row("Accuracy", f"{accuracy:.3f}")
    result_table.add_row("Training Time", f"{train_time:.2f}s")
    result_table.add_row("Model Saved", checkpoint_path)
    
    console.print(result_table)
    
    return {
        "success": True,
        "accuracy": accuracy,
        "train_time": train_time,
        "checkpoint": checkpoint_path
    }


def eval_command(args: argparse.Namespace) -> Dict[str, Any]:
    """Execute eval command - evaluate a trained model."""
    console.print(f"🧪 Evaluating {args.model} on {args.dataset}", style="bold green")
    
    # Load checkpoint
    if not os.path.exists(args.model):
        console.print(f"❌ Model not found: {args.model}", style="red")
        return {"success": False, "error": "Model not found"}
    
    checkpoint = torch.load(args.model, map_location="cpu")
    
    # Load dataset  
    dataset = create_dataset_loader(args.dataset, "test")
    if dataset is None:
        return {"success": False, "error": "Failed to load dataset"}
    
    # Sample episode
    support_x, support_y, query_x, query_y = sample_episode(
        dataset, checkpoint["n_way"], checkpoint["k_shot"], args.n_query
    )
    
    # Create and load model
    input_shape = support_x.shape[1:]
    model = create_model(checkpoint["algorithm"], input_shape, checkpoint["n_way"])
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    
    # Evaluate
    with torch.no_grad():
        if checkpoint["algorithm"] == "protonet":
            logits = model(support_x, support_y, query_x)
        else:  # MAML
            logits = model(query_x)
        
        accuracy = (logits.argmax(-1) == query_y).float().mean().item()
    
    # Report results
    result_table = Table(title="Evaluation Results") 
    result_table.add_column("Metric", style="cyan")
    result_table.add_column("Value", style="green")
    
    result_table.add_row("Model", args.model)
    result_table.add_row("Algorithm", checkpoint["algorithm"])
    result_table.add_row("Dataset", args.dataset) 
    result_table.add_row("Accuracy", f"{accuracy:.3f}")
    result_table.add_row("Training Accuracy", f"{checkpoint.get('accuracy', 0):.3f}")
    
    console.print(result_table)
    
    return {"success": True, "accuracy": accuracy}


def benchmark_command(args: argparse.Namespace) -> Dict[str, Any]:
    """Execute benchmark command - compare multiple algorithms."""
    console.print("🏆 Running benchmarks", style="bold magenta")
    
    datasets = args.datasets.split(",")
    algorithms = args.algorithms.split(",")
    
    results = []
    
    for dataset in datasets:
        for algorithm in algorithms:
            console.print(f"\n📈 Benchmarking {algorithm} on {dataset}")
            
            # Create temporary args for fit command
            temp_args = argparse.Namespace(
                dataset=dataset,
                algorithm=algorithm,
                n_way=args.n_way,
                k_shot=args.k_shot,
                n_query=args.n_query
            )
            
            result = fit_command(temp_args)
            if result["success"]:
                results.append({
                    "dataset": dataset,
                    "algorithm": algorithm,
                    "accuracy": result["accuracy"],
                    "train_time": result["train_time"]
                })
    
    # Create benchmark table
    benchmark_table = Table(title="Benchmark Results")
    benchmark_table.add_column("Dataset", style="cyan")
    benchmark_table.add_column("Algorithm", style="yellow") 
    benchmark_table.add_column("Accuracy", style="green")
    benchmark_table.add_column("Time (s)", style="blue")
    
    for result in results:
        benchmark_table.add_row(
            result["dataset"],
            result["algorithm"],
            f"{result['accuracy']:.3f}",
            f"{result['train_time']:.2f}"
        )
    
    console.print(benchmark_table)
    
    # Save results
    with open("benchmark_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    console.print("📊 Results saved to benchmark_results.json")
    
    return {"success": True, "results": results}


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="mlfew",
        description="Meta-Learning Few-Shot CLI Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  mlfew fit --dataset omniglot --algorithm protonet --n-way 5 --k-shot 1
  mlfew eval --model checkpoints/protonet_omniglot.pt --dataset omniglot  
  mlfew benchmark --datasets omniglot,miniimagenet --algorithms protonet,maml
        """
    )
    
    parser.add_argument("--version", action="version", version=f"mlfew {__version__}")
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Fit command
    fit_parser = subparsers.add_parser("fit", help="Train a model")
    fit_parser.add_argument("--dataset", required=True, 
                           choices=["omniglot", "miniimagenet"],
                           help="Dataset to use")
    fit_parser.add_argument("--algorithm", required=True,
                           choices=["protonet", "maml"],  
                           help="Algorithm to use")
    fit_parser.add_argument("--n-way", type=int, default=5,
                           help="Number of classes (default: 5)")
    fit_parser.add_argument("--k-shot", type=int, default=1,
                           help="Number of shots (default: 1)")
    fit_parser.add_argument("--n-query", type=int, default=15,
                           help="Query samples per class (default: 15)")
    
    # Eval command
    eval_parser = subparsers.add_parser("eval", help="Evaluate a model")
    eval_parser.add_argument("--model", required=True,
                            help="Path to model checkpoint") 
    eval_parser.add_argument("--dataset", required=True,
                            choices=["omniglot", "miniimagenet"],
                            help="Dataset to evaluate on")
    eval_parser.add_argument("--n-query", type=int, default=15,
                            help="Query samples per class (default: 15)")
    
    # Benchmark command
    benchmark_parser = subparsers.add_parser("benchmark", help="Run benchmarks")
    benchmark_parser.add_argument("--datasets", required=True,
                                 help="Comma-separated datasets (e.g. omniglot,miniimagenet)")
    benchmark_parser.add_argument("--algorithms", required=True,
                                 help="Comma-separated algorithms (e.g. protonet,maml)")
    benchmark_parser.add_argument("--n-way", type=int, default=5,
                                 help="Number of classes (default: 5)")
    benchmark_parser.add_argument("--k-shot", type=int, default=1,
                                 help="Number of shots (default: 1)")
    benchmark_parser.add_argument("--n-query", type=int, default=15,
                                 help="Query samples per class (default: 15)")
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    console.print(f"🧠 Meta-Learning CLI v{__version__}", style="bold blue")
    console.print("-" * 40)
    
    try:
        if args.command == "fit":
            result = fit_command(args)
        elif args.command == "eval":
            result = eval_command(args)
        elif args.command == "benchmark":
            result = benchmark_command(args)
        else:
            console.print(f"❌ Unknown command: {args.command}", style="red")
            return 1
        
        if result.get("success", False):
            console.print("\n✅ Command completed successfully", style="green")
            return 0
        else:
            console.print(f"\n❌ Command failed: {result.get('error', 'Unknown error')}", style="red")
            return 1
            
    except KeyboardInterrupt:
        console.print("\n🛑 Interrupted by user", style="yellow")
        return 1
    except Exception as e:
        console.print(f"\n💥 Unexpected error: {e}", style="red")
        return 1


if __name__ == "__main__":
    sys.exit(main())