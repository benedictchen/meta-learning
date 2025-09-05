from __future__ import annotations
from typing import Callable, Dict, Any, Iterable, Optional
import time, json, os
import torch
from .core.episode import Episode

def evaluate(run_logits: Callable[[Episode], torch.Tensor], episodes: Iterable[Episode], *, outdir: Optional[str] = None, dump_preds: bool = False) -> Dict[str, Any]:
    accs = []; preds_dump = []
    t0 = time.time(); episodes = list(episodes)
    for ep in episodes:
        logits = run_logits(ep)
        pred = logits.argmax(dim=1)
        accs.append((pred == ep.query_y).float().mean().item())
        if dump_preds:
            preds_dump.append({"pred": pred.tolist(), "y": ep.query_y.tolist()})
    dt = time.time()-t0
    n = len(accs); mean = sum(accs)/n
    var = sum((x-mean)**2 for x in accs)/(n-1 if n>1 else 1); sd = var**0.5
    ci = 1.96*sd/(n**0.5) if n>1 else 0.0
    res = {"episodes": n, "mean_acc": mean, "ci95": ci, "elapsed_s": dt}
    if outdir:
        os.makedirs(outdir, exist_ok=True)
        with open(os.path.join(outdir, "metrics.json"), "w") as f:
            json.dump(res, f, indent=2)
        if dump_preds:
            with open(os.path.join(outdir, "preds.jsonl"), "w") as f:
                for p in preds_dump: f.write(json.dumps(p)+"\n")
    return res
