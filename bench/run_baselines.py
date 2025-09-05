#!/usr/bin/env python3
"""
Reproducible benchmarks for meta-learning baselines.

Definition of Done:
- MiniImageNet AND CIFAR-FS, 5-way {1,5}-shot; ≥10,000 episodes per setting
- Report mean ± 95% CI, runtime (eps/s), backbone, transforms, BN policy, exact command + seed + env hash
- Results table checked into docs/results.md and artifacts (CSV/JSON) stored in runs/
- Single script reproduces tables on CPU (smoke) and GPU (full)

Usage:
    python bench/run_baselines.py --mode smoke  # CPU, 100 episodes
    python bench/run_baselines.py --mode full   # GPU, 10k episodes
"""

import argparse
import csv
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Any
import warnings

import torch
import numpy as np

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore', category=UserWarning)

@dataclass
class BenchmarkConfig:
    """Benchmark configuration with all reproducibility details."""
    dataset: str
    n_way: int
    n_shot: int
    n_query: int
    algorithm: str
    backbone: str
    n_episodes: int
    seed: int
    device: str
    batch_norm_policy: str
    transforms: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict for JSON serialization."""
        return asdict(self)

@dataclass  
class BenchmarkResult:
    """Benchmark result with all metrics and metadata."""
    config: BenchmarkConfig
    accuracy_mean: float
    accuracy_std: float
    accuracy_ci95: tuple
    runtime_seconds: float
    episodes_per_second: float
    command: str
    env_hash: str
    timestamp: str
    system_info: Dict[str, str]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict for JSON serialization."""
        result = {
            'config': self.config.to_dict(),
            'accuracy_mean': self.accuracy_mean,
            'accuracy_std': self.accuracy_std,
            'accuracy_ci95_lower': self.accuracy_ci95[0],
            'accuracy_ci95_upper': self.accuracy_ci95[1],
            'runtime_seconds': self.runtime_seconds,
            'episodes_per_second': self.episodes_per_second,
            'command': self.command,
            'env_hash': self.env_hash,
            'timestamp': self.timestamp,
            'system_info': self.system_info
        }
        return result

def get_env_hash() -> str:
    """Get reproducible environment hash."""
    import torch
    import numpy as np
    
    env_info = {
        'python_version': sys.version,
        'torch_version': torch.__version__,
        'numpy_version': np.__version__,
        'cuda_available': torch.cuda.is_available(),
        'cuda_version': torch.version.cuda if torch.cuda.is_available() else None,
        'platform': platform.platform(),
        'hostname': platform.node()
    }
    
    # Create hash from sorted key-value pairs
    env_str = json.dumps(env_info, sort_keys=True)
    return hashlib.md5(env_str.encode()).hexdigest()[:8]

def get_system_info() -> Dict[str, str]:
    """Get detailed system information."""
    return {
        'python_version': sys.version.split()[0],
        'torch_version': torch.__version__,
        'numpy_version': np.__version__,
        'cuda_available': str(torch.cuda.is_available()),
        'cuda_version': torch.version.cuda if torch.cuda.is_available() else 'None',
        'platform': platform.platform(),
        'processor': platform.processor(),
        'hostname': platform.node(),
        'num_threads': str(torch.get_num_threads())
    }

def run_benchmark(config: BenchmarkConfig) -> BenchmarkResult:
    """Run a single benchmark configuration."""
    print(f"Running {config.algorithm} on {config.dataset} "
          f"{config.n_way}w{config.n_shot}s ({config.n_episodes} episodes)...")
    
    # Import here to avoid issues if meta_learning not installed
    try:
        from meta_learning import ProtoNet, MAML, make_episodes
        from meta_learning.datasets import MiniImageNet, CIFARFS
    except ImportError as e:
        print(f"Error importing meta_learning: {e}")
        print("Please install with: pip install -e .")
        sys.exit(1)
    
    # Set deterministic seeds
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config.seed)
    
    # Build command for reproducibility
    command = (f"python bench/run_baselines.py --dataset {config.dataset} "
              f"--algorithm {config.algorithm} --n_way {config.n_way} "
              f"--n_shot {config.n_shot} --n_episodes {config.n_episodes} "
              f"--seed {config.seed} --device {config.device}")
    
    # Load dataset
    if config.dataset == 'miniImageNet':
        dataset = MiniImageNet(split='test')
    elif config.dataset == 'CIFAR-FS':
        dataset = CIFARFS(split='test')
    else:
        raise ValueError(f"Unknown dataset: {config.dataset}")
    
    # Create episode sampler
    episodes = make_episodes(
        dataset=dataset,
        n_way=config.n_way,
        n_shot=config.n_shot,
        n_query=config.n_query,
        n_episodes=config.n_episodes
    )
    
    # Initialize algorithm
    if config.algorithm == 'ProtoNet':
        model = ProtoNet(backbone=config.backbone).to(config.device)
    elif config.algorithm == 'MAML':
        model = MAML(backbone=config.backbone, inner_lr=0.01).to(config.device)
    else:
        raise ValueError(f"Unknown algorithm: {config.algorithm}")
    
    # Run evaluation
    start_time = time.time()
    accuracies = []
    
    for i, episode in enumerate(episodes):
        if i % 1000 == 0 and i > 0:
            elapsed = time.time() - start_time
            eps_per_sec = i / elapsed
            print(f"  Episode {i}/{config.n_episodes}, {eps_per_sec:.1f} eps/sec")
        
        # Move episode to device
        support_x = episode['support_x'].to(config.device)
        support_y = episode['support_y'].to(config.device)
        query_x = episode['query_x'].to(config.device)
        query_y = episode['query_y'].to(config.device)
        
        # Evaluate episode
        with torch.no_grad():
            if config.algorithm == 'ProtoNet':
                logits = model(support_x, support_y, query_x)
            elif config.algorithm == 'MAML':
                logits = model.forward_adapted(support_x, support_y, query_x)
            
            # Compute accuracy
            pred = torch.argmax(logits, dim=1)
            acc = (pred == query_y).float().mean().item()
            accuracies.append(acc)
    
    runtime = time.time() - start_time
    eps_per_sec = config.n_episodes / runtime
    
    # Compute statistics
    accuracies = np.array(accuracies)
    acc_mean = np.mean(accuracies)
    acc_std = np.std(accuracies)
    
    # 95% confidence interval
    ci_margin = 1.96 * acc_std / np.sqrt(len(accuracies))
    ci_lower = acc_mean - ci_margin
    ci_upper = acc_mean + ci_margin
    
    return BenchmarkResult(
        config=config,
        accuracy_mean=acc_mean,
        accuracy_std=acc_std,
        accuracy_ci95=(ci_lower, ci_upper),
        runtime_seconds=runtime,
        episodes_per_second=eps_per_sec,
        command=command,
        env_hash=get_env_hash(),
        timestamp=time.strftime('%Y-%m-%d %H:%M:%S'),
        system_info=get_system_info()
    )

def get_benchmark_configs(mode: str) -> List[BenchmarkConfig]:
    """Get benchmark configurations for given mode."""
    device = 'cuda' if torch.cuda.is_available() and mode == 'full' else 'cpu'
    n_episodes = 10000 if mode == 'full' else 100
    
    configs = []
    
    # Standard benchmarks
    for dataset in ['miniImageNet', 'CIFAR-FS']:
        for algorithm in ['ProtoNet', 'MAML']:
            for n_shot in [1, 5]:
                config = BenchmarkConfig(
                    dataset=dataset,
                    n_way=5,
                    n_shot=n_shot,
                    n_query=15,
                    algorithm=algorithm,
                    backbone='Conv4',
                    n_episodes=n_episodes,
                    seed=42,
                    device=device,
                    batch_norm_policy='episodic',
                    transforms='standard'
                )
                configs.append(config)
    
    return configs

def save_results(results: List[BenchmarkResult], mode: str):
    """Save results to CSV and JSON files."""
    runs_dir = Path('runs')
    runs_dir.mkdir(exist_ok=True)
    
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    
    # Save JSON (full details)
    json_path = runs_dir / f'benchmark_results_{mode}_{timestamp}.json'
    with open(json_path, 'w') as f:
        json.dump([result.to_dict() for result in results], f, indent=2)
    
    # Save CSV (summary table)  
    csv_path = runs_dir / f'benchmark_summary_{mode}_{timestamp}.csv'
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        
        # Header
        writer.writerow([
            'Dataset', 'Algorithm', 'N-Way', 'N-Shot', 'N-Episodes',
            'Accuracy', 'Std', 'CI95_Lower', 'CI95_Upper', 
            'Runtime(s)', 'Eps/Sec', 'Device', 'Seed', 'Env_Hash'
        ])
        
        # Data rows
        for result in results:
            c = result.config
            writer.writerow([
                c.dataset, c.algorithm, c.n_way, c.n_shot, c.n_episodes,
                f"{result.accuracy_mean:.4f}", f"{result.accuracy_std:.4f}",
                f"{result.accuracy_ci95[0]:.4f}", f"{result.accuracy_ci95[1]:.4f}",
                f"{result.runtime_seconds:.1f}", f"{result.episodes_per_second:.1f}",
                c.device, c.seed, result.env_hash
            ])
    
    print(f"Results saved:")
    print(f"  JSON: {json_path}")
    print(f"  CSV:  {csv_path}")
    
    return csv_path, json_path

def update_results_doc(csv_path: Path):
    """Update docs/results.md with latest results."""
    docs_dir = Path('docs')
    docs_dir.mkdir(exist_ok=True)
    
    results_path = docs_dir / 'results.md'
    
    # Read CSV data
    with open(csv_path, 'r') as f:
        reader = csv.reader(f)
        rows = list(reader)
    
    # Generate markdown table
    header = rows[0]
    data_rows = rows[1:]
    
    md_content = f"""# Benchmark Results

Last updated: {time.strftime('%Y-%m-%d %H:%M:%S')}

## Reproducibility Command

```bash
python bench/run_baselines.py --mode full
```

## Results Summary

| Dataset | Algorithm | Setup | Accuracy | CI95 | Eps/Sec | Device |
|---------|-----------|-------|----------|------|---------|--------|
"""
    
    for row in data_rows:
        dataset, algorithm, n_way, n_shot = row[0], row[1], row[2], row[3]
        accuracy, ci_lower, ci_upper = row[5], row[7], row[8]
        eps_sec, device = row[10], row[11]
        
        setup = f"{n_way}w{n_shot}s"
        ci_str = f"[{ci_lower}, {ci_upper}]"
        
        md_content += f"| {dataset} | {algorithm} | {setup} | {accuracy} | {ci_str} | {eps_sec} | {device} |\n"
    
    md_content += f"""
## Environment Details

- **Python**: {sys.version.split()[0]}
- **PyTorch**: {torch.__version__}  
- **CUDA**: {'Available' if torch.cuda.is_available() else 'Not Available'}
- **Platform**: {platform.platform()}

## Files

- **Raw results**: `{csv_path}`
- **Full details**: `{str(csv_path).replace('.csv', '.json')}`

Generated by `bench/run_baselines.py`
"""
    
    with open(results_path, 'w') as f:
        f.write(md_content)
    
    print(f"Updated: {results_path}")

def main():
    parser = argparse.ArgumentParser(description='Run reproducible meta-learning benchmarks')
    parser.add_argument('--mode', choices=['smoke', 'full'], default='smoke',
                       help='Benchmark mode: smoke (CPU, 100 eps) or full (GPU, 10k eps)')
    parser.add_argument('--dataset', help='Override dataset')
    parser.add_argument('--algorithm', help='Override algorithm')
    parser.add_argument('--n_way', type=int, help='Override n_way')
    parser.add_argument('--n_shot', type=int, help='Override n_shot')
    parser.add_argument('--n_episodes', type=int, help='Override n_episodes')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--device', help='Override device')
    
    args = parser.parse_args()
    
    print(f"🚀 Meta-Learning Benchmarks ({args.mode} mode)")
    print(f"Environment hash: {get_env_hash()}")
    print()
    
    if all(x is None for x in [args.dataset, args.algorithm, args.n_way, args.n_shot]):
        # Run full benchmark suite
        configs = get_benchmark_configs(args.mode)
        print(f"Running {len(configs)} benchmark configurations...")
    else:
        # Run single configuration
        device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
        config = BenchmarkConfig(
            dataset=args.dataset or 'miniImageNet',
            n_way=args.n_way or 5,
            n_shot=args.n_shot or 1,
            n_query=15,
            algorithm=args.algorithm or 'ProtoNet',
            backbone='Conv4',
            n_episodes=args.n_episodes or (10000 if args.mode == 'full' else 100),
            seed=args.seed,
            device=device,
            batch_norm_policy='episodic',
            transforms='standard'
        )
        configs = [config]
        print(f"Running single benchmark configuration...")
    
    # Run benchmarks
    results = []
    start_time = time.time()
    
    for i, config in enumerate(configs):
        print(f"\n[{i+1}/{len(configs)}]", end=" ")
        try:
            result = run_benchmark(config)
            results.append(result)
            
            print(f"✅ {result.accuracy_mean:.3f} ± {result.accuracy_std:.3f} "
                  f"({result.episodes_per_second:.1f} eps/sec)")
        except Exception as e:
            print(f"❌ Failed: {e}")
    
    total_time = time.time() - start_time
    print(f"\n🏁 Completed {len(results)}/{len(configs)} benchmarks in {total_time:.1f}s")
    
    if results:
        csv_path, json_path = save_results(results, args.mode)
        update_results_doc(csv_path)
    
    return 0

if __name__ == '__main__':
    exit(main())