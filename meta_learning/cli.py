from __future__ import annotations
import argparse, json, sys, torch
from ._version import __version__
from .core.seed import seed_all
from .data import SyntheticFewShotDataset, CIFARFSDataset, make_episodes
from .models.conv4 import Conv4
from .algos.protonet import ProtoHead
from .eval import evaluate

def make_encoder(name: str, out_dim: int = 64):
    if name == "identity":
        return torch.nn.Identity()
    if name == "conv4":
        return Conv4(out_dim=out_dim)
    raise ValueError("encoder must be 'identity' or 'conv4'")

def cmd_version(_): print(__version__)

def _device(devopt: str) -> torch.device:
    if devopt == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(devopt)

def cmd_eval(args):
    seed_all(args.seed)
    device = _device(args.device)
    head = ProtoHead(distance=args.distance, tau=args.tau).to(device)
    enc = make_encoder(args.encoder, out_dim=args.emb_dim).to(device)

    # dataset selection
    if args.dataset == "synthetic":
        ds = SyntheticFewShotDataset(n_classes=50, dim=args.emb_dim, noise=args.noise)
        # For synthetic vectors, the "encoder" is identity: use features directly
        def run_logits(ep):
            z_s = ep.support_x.to(device) if args.encoder=="identity" else enc(ep.support_x.to(device))
            z_q = ep.query_x.to(device) if args.encoder=="identity" else enc(ep.query_x.to(device))
            return head(z_s, ep.support_y.to(device), z_q)
        eps = list(make_episodes(ds, args.n_way, args.k_shot, args.m_query, args.episodes))
    elif args.dataset == "cifar_fs":
        ds = CIFARFSDataset(root=args.data_root, split=args.split, download=args.download, image_size=args.image_size)
        def run_logits(ep):
            z_s = enc(ep.support_x.to(device))
            z_q = enc(ep.query_x.to(device))
            return head(z_s, ep.support_y.to(device), z_q)
        # Build episodes on-the-fly (avoid storing all in RAM)
        eps = (Episode(*ds.sample_support_query(args.n_way, args.k_shot, args.m_query, seed=args.seed+i)) for i in range(args.episodes))
        # Materialize to list for evaluate (small episode count recommended)
        eps = [e for e in eps]
        # Validate episodes
        for e in eps: e.validate(expect_n_classes=args.n_way)
    else:
        raise ValueError("dataset must be 'synthetic' or 'cifar_fs'")

    res = evaluate(run_logits, eps, outdir=args.outdir)
    print(json.dumps(res, indent=2))

def cmd_bench(args):
    from .bench import run_benchmark
    seed_all(args.seed)
    device = _device(args.device)
    head = ProtoHead(distance=args.distance, tau=args.tau).to(device)
    enc = make_encoder(args.encoder, out_dim=args.emb_dim).to(device)

    if args.dataset == "synthetic":
        ds = SyntheticFewShotDataset(n_classes=50, dim=args.emb_dim, noise=args.noise)
        def episode_acc():
            xs, ys, xq, yq = ds.sample_support_query(args.n_way, args.k_shot, args.m_query)
            z_s = xs.to(device) if args.encoder=="identity" else enc(xs.to(device))
            z_q = xq.to(device) if args.encoder=="identity" else enc(xq.to(device))
            pred = head(z_s, ys.to(device), z_q).argmax(1)
            return float((pred==yq.to(device)).float().mean().item())
    elif args.dataset == "cifar_fs":
        ds = CIFARFSDataset(root=args.data_root, split=args.split, download=args.download, image_size=args.image_size)
        def episode_acc():
            xs, ys, xq, yq = ds.sample_support_query(args.n_way, args.k_shot, args.m_query)
            pred = head(enc(xs.to(device)), ys.to(device), enc(xq.to(device))).argmax(1)
            return float((pred==yq.to(device)).float().mean().item())
    else:
        raise ValueError("dataset must be 'synthetic' or 'cifar_fs'")

    res = run_benchmark(episode_acc, episodes=args.episodes, warmup=min(20, args.episodes//10), meta={"algo":"protonet","dataset":args.dataset})
    print(json.dumps(res.__dict__, indent=2))

def main(argv=None):
    p = argparse.ArgumentParser("mlfew")
    sub = p.add_subparsers(dest="cmd", required=True)

    pv = sub.add_parser("version"); pv.set_defaults(func=cmd_version)

    pe = sub.add_parser("eval")
    pe.add_argument("--dataset", choices=["synthetic","cifar_fs"], default="synthetic")
    pe.add_argument("--split", choices=["train","val","test"], default="val")
    pe.add_argument("--n-way", type=int, default=5); pe.add_argument("--k-shot", type=int, default=1)
    pe.add_argument("--m-query", type=int, default=15); pe.add_argument("--episodes", type=int, default=200)
    pe.add_argument("--encoder", choices=["identity","conv4"], default="identity")
    pe.add_argument("--emb-dim", type=int, default=64)
    pe.add_argument("--distance", choices=["sqeuclidean","cosine"], default="sqeuclidean")
    pe.add_argument("--tau", type=float, default=1.0)
    pe.add_argument("--noise", type=float, default=0.1)             # synthetic only
    pe.add_argument("--data-root", type=str, default="data")        # torchvision root
    pe.add_argument("--download", action="store_true")
    pe.add_argument("--image-size", type=int, default=32)
    pe.add_argument("--device", choices=["auto","cpu","cuda"], default="auto")
    pe.add_argument("--seed", type=int, default=1234)
    pe.add_argument("--outdir", type=str, default="runs/eval")
    pe.set_defaults(func=cmd_eval)

    pb = sub.add_parser("bench")
    for arg, typ, default in [
        ("--dataset", str, "synthetic"),
        ("--split", str, "val"),
        ("--n-way", int, 5),
        ("--k-shot", int, 1),
        ("--m-query", int, 15),
        ("--episodes", int, 500),
        ("--encoder", str, "identity"),
        ("--emb-dim", int, 64),
        ("--distance", str, "sqeuclidean"),
        ("--tau", float, 1.0),
        ("--noise", float, 0.1),
        ("--data-root", str, "data"),
        ("--download", bool, False),
        ("--image-size", int, 32),
        ("--device", str, "auto"),
        ("--seed", int, 1234),
    ]:
        if typ is bool:
            pb.add_argument(arg, action="store_true")
        else:
            pb.add_argument(arg, type=typ, default=default)
    pb.set_defaults(func=cmd_bench)

    args = p.parse_args(argv); return args.func(args)

if __name__ == "__main__":
    sys.exit(main())
