from __future__ import annotations
from typing import Callable, Dict, Any, Iterable
import time, json, os
import torch
from .core.episode import Episode

def evaluate(run_logits: Callable[[Episode], torch.Tensor], episodes: Iterable[Episode], *, outdir: str | None = None) -> Dict[str, Any]:
    accs = []; t0 = time.time()
    for ep in episodes:
        logits = run_logits(ep)
        pred = logits.argmax(dim=1)
        accs.append((pred == ep.query_y).float().mean().item())
    dt = time.time()-t0
    n = len(accs); mean = sum(accs)/n
    var = sum((x-mean)**2 for x in accs)/(n-1 if n>1 else 1); sd = var**0.5
    ci = 1.96*sd/(n**0.5) if n>1 else 0.0
    res = {"episodes": n, "mean_acc": mean, "ci95": ci, "elapsed_s": dt}
    if outdir:
        os.makedirs(outdir, exist_ok=True)
        with open(os.path.join(outdir, "metrics.json"), "w") as f:
            json.dump(res, f, indent=2)
    return res
