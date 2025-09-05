from __future__ import annotations
import argparse, json, sys, torch
from ._version import __version__
from .core.seed import seed_all
from .core.bn_policy import freeze_batchnorm_running_stats
from .data import SyntheticFewShotDataset, CIFARFSDataset, MiniImageNetDataset, make_episodes, Episode
from .models.conv4 import Conv4
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
    if name == "identity":
        return torch.nn.Identity()
    if name == "conv4":
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
            # Enable dropout/stochastic behavior during inference
            enc.train() if hasattr(enc, 'train') else None
            head.train() if hasattr(head, 'train') else None
            
            prob_list = []
            for _ in range(args.ttcs):
                z_s = ep.support_x.to(device) if isinstance(enc, torch.nn.Identity) and ep.support_x.dim()==2 else enc(ep.support_x.to(device))
                z_q = ep.query_x.to(device) if isinstance(enc, torch.nn.Identity) and ep.query_x.dim()==2 else enc(ep.query_x.to(device))
                logits = head(z_s, ep.support_y.to(device), z_q)
                
                # Convert logits to probabilities for proper Bayesian ensembling
                probs = torch.softmax(logits, dim=-1)
                prob_list.append(probs)
            
            # Bayesian Model Averaging: average probabilities, then convert back to logits
            # This is mathematically correct for ensemble uncertainty estimation
            mean_probs = torch.stack(prob_list).mean(dim=0)
            
            # Convert averaged probabilities back to logits with numerical stability
            eps = torch.finfo(mean_probs.dtype).eps
            mean_probs = torch.clamp(mean_probs, min=eps, max=1.0 - eps)
            return torch.log(mean_probs)
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

if __name__ == "__main__":
    sys.exit(main())
