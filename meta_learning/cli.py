from __future__ import annotations
import argparse, json, sys, torch
from ._version import __version__
from .core.seed import seed_all
from .core.bn_policy import freeze_batchnorm_running_stats
from .data import SyntheticFewShotDataset, CIFARFSDataset, MiniImageNetDataset, make_episodes, Episode
from .models.conv4 import Conv4
from .algos.protonet import ProtoHead
from .eval import evaluate
from .bench import run_benchmark

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
        return SyntheticFewShotDataset(n_classes=50, dim=args.emb_dim, noise=args.noise)
    if args.dataset == "cifar_fs":
        return CIFARFSDataset(root=args.data_root, split=args.split, manifest_path=args.manifest, download=args.download, image_size=args.image_size)
    if args.dataset == "miniimagenet":
        return MiniImageNetDataset(root=args.data_root, split=args.split, image_size=args.image_size)
    raise ValueError("unknown dataset")

def cmd_eval(args):
    seed_all(args.seed)
    device = _device(args.device)
    head = ProtoHead(distance=args.distance, tau=args.tau).to(device)
    enc = make_encoder(args.encoder, out_dim=args.emb_dim, p_drop=args.dropout).to(device)
    ds = _build_dataset(args)

    if args.freeze_bn: freeze_batchnorm_running_stats(enc)

    def run_logits(ep: Episode):
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
    pe.add_argument("--seed", type=int, default=1234); pe.add_argument("--outdir", type=str, default=None); pe.add_argument("--dump-preds", action="store_true")
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
