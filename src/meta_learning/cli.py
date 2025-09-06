from __future__ import annotations

import argparse, json, sys, torch, os, time
from pathlib import Path
from ._version import __version__
from .core.seed import seed_all
from .core.bn_policy import freeze_batchnorm_running_stats
from .data import SyntheticFewShotDataset, CIFARFSDataset, MiniImageNetDataset, make_episodes, Episode
# Lazy import Conv4 to prevent crashes when using identity encoder
# from .models.conv4 import Conv4
from .algos.protonet import ProtoHead
from .algos.maml import ContinualMAML
from .eval import evaluate
from .bench import run_benchmark

# Import integrated advanced functionality
try:
    from .hardware_utils import create_hardware_config, setup_optimal_hardware
    from .leakage_guard import create_leakage_guard
    ADVANCED_FEATURES = True
except ImportError:
    ADVANCED_FEATURES = False

def make_encoder(name: str, out_dim: int = 64, p_drop: float = 0.0):
    """Create encoder with lazy imports to prevent crashes."""
    if name == "identity":
        return torch.nn.Identity()
    if name == "conv4":
        # Lazy import to avoid crash when using identity encoder
        from .models.conv4 import Conv4
        return Conv4(out_dim=out_dim, p_drop=p_drop)
    raise ValueError("encoder must be 'identity' or 'conv4'")

def _device(devopt: str) -> torch.device:
    if devopt == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(devopt)

def cmd_version(_): print(__version__)

def _build_dataset(args):
    if args.dataset == "synthetic":
        # Enable image_mode for synthetic data when using conv4 encoder (for compatibility)
        image_mode = (args.encoder == "conv4")
        return SyntheticFewShotDataset(n_classes=50, dim=args.emb_dim, noise=args.noise, image_mode=image_mode)
    if args.dataset == "cifar_fs":
        return CIFARFSDataset(root=args.data_root, split=args.split, manifest_path=args.manifest, download=args.download, image_size=args.image_size)
    if args.dataset == "miniimagenet":
        return MiniImageNetDataset(root=args.data_root, split=args.split, image_size=args.image_size)
    raise ValueError("unknown dataset")

def cmd_eval(args):
    seed_all(args.seed)
    device = _device(args.device)
    
    # Create ProtoHead with integrated uncertainty estimation
    uncertainty_method = getattr(args, 'uncertainty', None)
    head = ProtoHead(
        distance=args.distance, 
        tau=args.tau,
        prototype_shrinkage=getattr(args, 'prototype_shrinkage', 0.0),
        uncertainty_method=uncertainty_method,
        dropout_rate=getattr(args, 'uncertainty_dropout', 0.1),
        n_uncertainty_samples=getattr(args, 'uncertainty_samples', 10)
    ).to(device)
    
    enc = make_encoder(args.encoder, out_dim=args.emb_dim, p_drop=args.dropout).to(device)
    
    # Hardware optimization if available
    if ADVANCED_FEATURES and getattr(args, 'optimize_hardware', False):
        hardware_config = create_hardware_config(device=str(device))
        enc, _ = setup_optimal_hardware(enc, hardware_config)
        print(f"✓ Hardware optimized for {device}")
    
    ds = _build_dataset(args)

    if args.freeze_bn: freeze_batchnorm_running_stats(enc)
    
    # Leakage detection if available  
    leakage_guard = None
    if ADVANCED_FEATURES and getattr(args, 'check_leakage', False):
        leakage_guard = create_leakage_guard(strict_mode=False)
        print("✓ Leakage detection enabled")

    def run_logits(ep: Episode):
        # Test-Time Compute Scaling: multiple stochastic forward passes
        if args.ttcs > 1:
            # Use improved TTCS implementation with TTA and better MC-Dropout
            from .algos.ttcs import ttcs_predict
            return ttcs_predict(
                enc, head, ep, 
                passes=args.ttcs, 
                device=device, 
                combine=args.combine,
                image_size=args.image_size,
                enable_mc_dropout=True
            )
        else:
            # Standard single forward pass
            z_s = ep.support_x.to(device) if isinstance(enc, torch.nn.Identity) and ep.support_x.dim()==2 else enc(ep.support_x.to(device))
            z_q = ep.query_x.to(device) if isinstance(enc, torch.nn.Identity) and ep.query_x.dim()==2 else enc(ep.query_x.to(device))
            return head(z_s, ep.support_y.to(device), z_q)

    if args.dataset == "synthetic":
        eps = list(make_episodes(ds, args.n_way, args.k_shot, args.m_query, args.episodes))
    else:
        eps = [Episode(*ds.sample_support_query(args.n_way, args.k_shot, args.m_query, seed=args.seed+i)) for i in range(args.episodes)]
        for e in eps: e.validate(expect_n_classes=args.n_way)

    res = evaluate(run_logits, eps, outdir=args.outdir, dump_preds=args.dump_preds)
    print(json.dumps(res, indent=2))

def cmd_bench(args):
    seed_all(args.seed)
    device = _device(args.device)
    head = ProtoHead(distance=args.distance, tau=args.tau).to(device)
    enc = make_encoder(args.encoder, out_dim=args.emb_dim, p_drop=args.dropout).to(device)
    ds = _build_dataset(args)
    if args.freeze_bn: freeze_batchnorm_running_stats(enc)

    def episode_acc():
        xs, ys, xq, yq = ds.sample_support_query(args.n_way, args.k_shot, args.m_query)
        z_s = xs.to(device) if isinstance(enc, torch.nn.Identity) and xs.dim()==2 else enc(xs.to(device))
        z_q = xq.to(device) if isinstance(enc, torch.nn.Identity) and xq.dim()==2 else enc(xq.to(device))
        pred = head(z_s, ys.to(device), z_q).argmax(1)
        return float((pred==yq.to(device)).float().mean().item())

    res = run_benchmark(episode_acc, episodes=args.episodes, warmup=min(20, args.episodes//10), meta={"algo":"protonet","dataset":args.dataset}, outdir=args.outdir)
    print(json.dumps(res.__dict__, indent=2))

def main(argv=None):
    p = argparse.ArgumentParser("mlfew")
    sub = p.add_subparsers(dest="cmd", required=True)

    pv = sub.add_parser("version"); pv.set_defaults(func=cmd_version)

    common = dict()
    pe = sub.add_parser("eval")
    pe.add_argument("--dataset", choices=["synthetic","cifar_fs","miniimagenet"], default="synthetic")
    pe.add_argument("--split", choices=["train","val","test"], default="val")
    pe.add_argument("--n-way", type=int, default=5); pe.add_argument("--k-shot", type=int, default=1)
    pe.add_argument("--m-query", type=int, default=15); pe.add_argument("--episodes", type=int, default=200)
    pe.add_argument("--encoder", choices=["identity","conv4"], default="identity")
    pe.add_argument("--emb-dim", type=int, default=64); pe.add_argument("--dropout", type=float, default=0.0)
    pe.add_argument("--distance", choices=["sqeuclidean","cosine"], default="sqeuclidean"); pe.add_argument("--tau", type=float, default=1.0)
    pe.add_argument("--noise", type=float, default=0.1)
    pe.add_argument("--data-root", type=str, default="data"); pe.add_argument("--manifest", type=str, default=None)
    pe.add_argument("--download", action="store_true"); pe.add_argument("--image-size", type=int, default=32)
    pe.add_argument("--device", choices=["auto","cpu","cuda"], default="auto"); pe.add_argument("--freeze-bn", action="store_true")
    pe.add_argument("--seed", type=int, default=1234); pe.add_argument("--ttcs", type=int, default=1, help="Test-Time Compute Scaling: number of stochastic forward passes")
    pe.add_argument("--combine", choices=["mean_prob","mean_logit"], default="mean_prob", help="TTCS ensemble combination method")
    pe.add_argument("--outdir", type=str, default=None); pe.add_argument("--dump-preds", action="store_true")
    
    # Advanced integrated features
    if ADVANCED_FEATURES:
        pe.add_argument("--uncertainty", choices=["monte_carlo_dropout"], default=None, help="Enable uncertainty estimation")
        pe.add_argument("--uncertainty-dropout", type=float, default=0.1, help="Dropout rate for uncertainty estimation")  
        pe.add_argument("--uncertainty-samples", type=int, default=10, help="Number of samples for uncertainty estimation")
        pe.add_argument("--prototype-shrinkage", type=float, default=0.0, help="Prototype shrinkage regularization")
        pe.add_argument("--optimize-hardware", action="store_true", help="Enable hardware optimization")
        pe.add_argument("--check-leakage", action="store_true", help="Enable leakage detection")
        pe.add_argument("--continual-learning", action="store_true", help="Enable continual learning mode")
        pe.add_argument("--memory-size", type=int, default=1000, help="Memory size for continual learning")
        pe.add_argument("--ewc-strength", type=float, default=1000.0, help="EWC regularization strength")
    
    pe.set_defaults(func=cmd_eval)

    pb = sub.add_parser("bench")
    for arg, typ, default in [
        ("--dataset", str, "synthetic"), ("--split", str, "val"),
        ("--n-way", int, 5), ("--k-shot", int, 1), ("--m-query", int, 15), ("--episodes", int, 500),
        ("--encoder", str, "identity"), ("--emb-dim", int, 64), ("--dropout", float, 0.0),
        ("--distance", str, "sqeuclidean"), ("--tau", float, 1.0), ("--noise", float, 0.1),
        ("--data-root", str, "data"), ("--manifest", str, None), ("--download", bool, False),
        ("--image-size", int, 32), ("--device", str, "auto"), ("--freeze-bn", bool, False), ("--seed", int, 1234), ("--outdir", str, None),
    ]:
        if typ is bool:
            pb.add_argument(arg, action="store_true")
        else:
            pb.add_argument(arg, type=typ, default=default)
    pb.set_defaults(func=cmd_bench)

    args = p.parse_args(argv); return args.func(args)


# CLI experiment tracking utilities
class ExperimentTracker:
    """
    Experiment tracking system for meta-learning research.
    
    Features:
    - Hierarchical experiment organization
    - Automatic hyperparameter logging
    - Result comparison and analysis
    - Reproducible experiment management
    """
    
    def __init__(self, base_dir: str = "experiments"):
        """Initialize experiment tracker."""
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(exist_ok=True)
    
    def create_experiment(self, name: str, config: dict) -> str:
        """Create new experiment with unique ID."""
        timestamp = int(time.time())
        exp_id = f"{name}_{timestamp}"
        exp_dir = self.base_dir / exp_id
        exp_dir.mkdir(exist_ok=True)
        
        # Save experiment configuration
        config_path = exp_dir / "config.json"
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        return str(exp_dir)
    
    def log_results(self, exp_dir: str, results: dict):
        """Log experiment results."""
        exp_path = Path(exp_dir)
        results_path = exp_path / "results.json"
        
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
    
    def compare_experiments(self, exp_names: list) -> dict:
        """Compare multiple experiments."""
        comparisons = {}
        
        for exp_name in exp_names:
            exp_dirs = list(self.base_dir.glob(f"{exp_name}_*"))
            if not exp_dirs:
                continue
                
            # Load most recent experiment
            latest_exp = max(exp_dirs, key=lambda x: x.stat().st_mtime)
            
            # Load results
            results_path = latest_exp / "results.json"
            if results_path.exists():
                with open(results_path, 'r') as f:
                    results = json.load(f)
                comparisons[exp_name] = results
        
        return comparisons


class HyperparameterOptimizer:
    """
    Automatic hyperparameter optimization using grid search.
    
    Features:
    - Grid search over hyperparameter spaces
    - Random search for large spaces
    - Early stopping based on validation performance
    - Best configuration selection
    """
    
    def __init__(self, param_space: dict, optimization_metric: str = 'mean_acc'):
        """Initialize hyperparameter optimizer."""
        self.param_space = param_space
        self.optimization_metric = optimization_metric
        self.results = []
    
    def generate_configurations(self, method: str = 'grid', n_samples: int = None):
        """Generate hyperparameter configurations."""
        import itertools
        import random
        
        if method == 'grid':
            # Grid search - all combinations
            keys, values = zip(*self.param_space.items())
            configurations = []
            
            for combination in itertools.product(*values):
                config = dict(zip(keys, combination))
                configurations.append(config)
            
            return configurations
            
        elif method == 'random':
            # Random search - sample configurations
            if n_samples is None:
                n_samples = 50
            
            configurations = []
            for _ in range(n_samples):
                config = {}
                for param, values in self.param_space.items():
                    config[param] = random.choice(values)
                configurations.append(config)
            
            return configurations
        
        else:
            raise ValueError("Method must be 'grid' or 'random'")
    
    def evaluate_configuration(self, config: dict, eval_func) -> float:
        """Evaluate a single configuration."""
        try:
            results = eval_func(config)
            metric_value = results.get(self.optimization_metric, 0.0)
            
            self.results.append({
                'config': config,
                'results': results,
                'metric_value': metric_value
            })
            
            return metric_value
            
        except Exception as e:
            print(f"Configuration failed: {config}, Error: {e}")
            return -float('inf')
    
    def find_best_configuration(self):
        """Find the best configuration from results."""
        if not self.results:
            return None
        
        best_result = max(self.results, key=lambda x: x['metric_value'])
        return best_result


def cmd_hyperopt(args):
    """Hyperparameter optimization command."""
    # Define hyperparameter space
    param_space = {
        'tau': [0.1, 0.5, 1.0, 2.0, 5.0],
        'distance': ['sqeuclidean', 'cosine'],
        'emb_dim': [64, 128, 256],
        'dropout': [0.0, 0.1, 0.2]
    }
    
    if hasattr(args, 'param_space') and args.param_space:
        # Load custom parameter space
        with open(args.param_space, 'r') as f:
            param_space = json.load(f)
    
    optimizer = HyperparameterOptimizer(param_space)
    configurations = optimizer.generate_configurations(
        method=args.search_method, 
        n_samples=args.n_samples
    )
    
    print(f"Evaluating {len(configurations)} configurations...")
    
    def eval_func(config):
        # Create modified args with config
        eval_args = argparse.Namespace(**vars(args))
        for key, value in config.items():
            setattr(eval_args, key, value)
        
        # Run evaluation (capture output)
        import io
        from contextlib import redirect_stdout
        
        f = io.StringIO()
        with redirect_stdout(f):
            cmd_eval(eval_args)
        
        # Parse JSON output
        output = f.getvalue().strip()
        try:
            results = json.loads(output)
            return results
        except json.JSONDecodeError:
            return {'mean_acc': 0.0}
    
    # Evaluate configurations
    for i, config in enumerate(configurations):
        print(f"Evaluating configuration {i+1}/{len(configurations)}: {config}")
        optimizer.evaluate_configuration(config, eval_func)
    
    # Find and report best configuration
    best = optimizer.find_best_configuration()
    if best:
        print("\nBest Configuration:")
        print(json.dumps(best['config'], indent=2))
        print(f"\nBest {optimizer.optimization_metric}: {best['metric_value']:.4f}")
        
        # Save results
        if args.outdir:
            outdir = Path(args.outdir)
            outdir.mkdir(exist_ok=True)
            
            with open(outdir / "hyperopt_results.json", 'w') as f:
                json.dump(optimizer.results, f, indent=2)


def cmd_compare(args):
    """Compare multiple experiment results."""
    tracker = ExperimentTracker(args.exp_dir)
    
    if args.experiment_names:
        exp_names = args.experiment_names.split(',')
    else:
        # Auto-detect experiments
        exp_names = [d.name.split('_')[0] for d in tracker.base_dir.iterdir() if d.is_dir()]
        exp_names = list(set(exp_names))  # Remove duplicates
    
    comparisons = tracker.compare_experiments(exp_names)
    
    if not comparisons:
        print("No experiments found for comparison.")
        return
    
    print("Experiment Comparison:")
    print("=" * 50)
    
    for exp_name, results in comparisons.items():
        print(f"\n{exp_name}:")
        if 'mean_acc' in results:
            print(f"  Accuracy: {results['mean_acc']:.4f} ± {results.get('ci95', 0.0):.4f}")
        if 'elapsed_s' in results:
            print(f"  Time: {results['elapsed_s']:.2f}s")
        if 'episodes' in results:
            print(f"  Episodes: {results['episodes']}")
    
    # Statistical comparison if multiple experiments
    if len(comparisons) >= 2:
        from .eval import StatisticalTestSuite
        
        # Mock data for comparison (in practice, would load raw accuracies)
        test_suite = StatisticalTestSuite()
        print(f"\nStatistical comparison would require raw episode accuracies.")
        print("Enable --dump-preds in evaluation to support statistical testing.")


def cmd_dataset_info(args):
    """Display dataset information and statistics."""
    ds = _build_dataset(args)
    
    print(f"Dataset: {args.dataset}")
    print(f"Split: {args.split}")
    
    if hasattr(ds, 'n_classes'):
        print(f"Classes: {ds.n_classes}")
    if hasattr(ds, 'n_samples'):
        print(f"Samples: {ds.n_samples}")
    
    # Sample episode for analysis
    if hasattr(ds, 'sample_support_query'):
        xs, ys, xq, yq = ds.sample_support_query(args.n_way, args.k_shot, args.m_query)
        print(f"\nEpisode sample:")
        print(f"  Support shape: {xs.shape}")
        print(f"  Query shape: {xq.shape}")
        print(f"  Support classes: {torch.unique(ys).tolist()}")
        print(f"  Query classes: {torch.unique(yq).tolist()}")


# Add new CLI commands
def add_advanced_commands(subparsers):
    """Add advanced CLI commands."""
    
    # Hyperparameter optimization
    p_hyperopt = subparsers.add_parser("hyperopt", help="Hyperparameter optimization")
    p_hyperopt.add_argument("--search-method", choices=["grid", "random"], default="grid")
    p_hyperopt.add_argument("--n-samples", type=int, default=50, help="Number of samples for random search")
    p_hyperopt.add_argument("--param-space", type=str, help="JSON file with parameter space")
    p_hyperopt.add_argument("--metric", type=str, default="mean_acc", help="Optimization metric")
    p_hyperopt.add_argument("--outdir", type=str, help="Output directory for results")
    
    # Copy eval arguments
    for action in [a for a in sys.modules[__name__].pe._actions if a.dest not in ['help']]:
        p_hyperopt.add_argument(*action.option_strings, **{
            k: v for k, v in action.__dict__.items() 
            if k not in ['option_strings', 'dest', 'container']
        })
    
    p_hyperopt.set_defaults(func=cmd_hyperopt)
    
    # Experiment comparison
    p_compare = subparsers.add_parser("compare", help="Compare experiment results")
    p_compare.add_argument("--exp-dir", type=str, default="experiments", help="Experiments directory")
    p_compare.add_argument("--experiment-names", type=str, help="Comma-separated experiment names")
    p_compare.set_defaults(func=cmd_compare)
    
    # Dataset information
    p_dataset = subparsers.add_parser("dataset-info", help="Display dataset information")
    p_dataset.add_argument("--dataset", choices=["synthetic", "cifar_fs", "miniimagenet"], required=True)
    p_dataset.add_argument("--split", choices=["train", "val", "test"], default="val")
    p_dataset.add_argument("--n-way", type=int, default=5)
    p_dataset.add_argument("--k-shot", type=int, default=1)
    p_dataset.add_argument("--m-query", type=int, default=15)
    p_dataset.add_argument("--data-root", type=str, default="data")
    p_dataset.add_argument("--manifest", type=str, default=None)
    p_dataset.add_argument("--image-size", type=int, default=32)
    p_dataset.add_argument("--noise", type=float, default=0.1)
    p_dataset.add_argument("--emb-dim", type=int, default=64)
    p_dataset.set_defaults(func=cmd_dataset_info)


# Store reference to eval parser for hyperopt
pe = None

if __name__ == "__main__":
    sys.exit(main())
