from __future__ import annotations
from typing import Callable, Dict, Any, Iterable, Optional
import time, json, os, math
import torch
from .core.episode import Episode

def _get_t_critical(df: int) -> float:
    """Get t-critical value for 95% confidence interval given degrees of freedom.
    
    This approximates scipy.stats.t.ppf(0.975, df) for common sample sizes.
    For research-grade statistical inference with small samples.
    """
    # Pre-computed t-values for 95% CI (two-tailed, α=0.05)
    t_table = {
        1: 12.71, 2: 4.30, 3: 3.18, 4: 2.78, 5: 2.57, 6: 2.45, 7: 2.36, 8: 2.31, 9: 2.26,
        10: 2.23, 11: 2.20, 12: 2.18, 13: 2.16, 14: 2.14, 15: 2.13, 16: 2.12, 17: 2.11, 
        18: 2.10, 19: 2.09, 20: 2.09, 21: 2.08, 22: 2.07, 23: 2.07, 24: 2.06, 25: 2.06,
        26: 2.06, 27: 2.05, 28: 2.05, 29: 2.05
    }
    
    if df in t_table:
        return t_table[df]
    elif df >= 30:
        return 1.96  # Normal approximation for large samples
    else:
        # Linear interpolation for missing values (simple approximation)
        return 2.78 - (df - 4) * 0.02  # Rough approximation

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
    
    # Use t-distribution for small sample sizes (research-grade statistics)
    # This provides more accurate confidence intervals than normal approximation
    if n > 1:
        # For small samples (n < 30), use t-distribution; for large samples, t ≈ normal
        # t.ppf(0.975, df=n-1) for 95% CI, but approximated here for common sample sizes
        if n < 30:
            # Common t-values for 95% CI: t(df=4)≈2.78, t(df=9)≈2.26, t(df=19)≈2.09, t(df=29)≈2.05
            t_critical = _get_t_critical(n-1) 
        else:
            t_critical = 1.96  # Normal approximation for large samples
        
        ci = t_critical * sd / math.sqrt(n)
        se = sd / math.sqrt(n)  # Standard error
    else:
        ci = 0.0
        se = 0.0
        
    res = {"episodes": n, "mean_acc": mean, "ci95": ci, "std_err": se, "elapsed_s": dt}
    if outdir:
        os.makedirs(outdir, exist_ok=True)
        with open(os.path.join(outdir, "metrics.json"), "w") as f:
            json.dump(res, f, indent=2)
        if dump_preds:
            with open(os.path.join(outdir, "preds.jsonl"), "w") as f:
                for p in preds_dump: f.write(json.dumps(p)+"\n")
    return res
