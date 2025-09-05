from __future__ import annotations
import time
from dataclasses import dataclass
from typing import Callable, Any, Dict, Optional, List

@dataclass
class BenchResult:
    mean_acc: float
    ci95: float
    episodes: int
    eps_per_sec: float
    meta: Dict[str, Any]

def _mean_ci95(xs: List[float]):
    n = len(xs); m = sum(xs)/n
    var = sum((x-m)**2 for x in xs)/(n-1 if n>1 else 1); sd = var**0.5
    return m, (1.96*sd/(n**0.5) if n>1 else 0.0)

def run_benchmark(run_episode: Callable[[], float], episodes: int = 1000, warmup: int = 20, meta: Optional[Dict[str, Any]] = None) -> BenchResult:
    meta = meta or {}
    for _ in range(max(0, warmup)): _ = run_episode()
    accs = []; t0 = time.perf_counter()
    for _ in range(episodes): accs.append(float(run_episode()))
    dt = time.perf_counter()-t0; m, ci = _mean_ci95(accs)
    eps = episodes / max(dt, 1e-9)
    return BenchResult(mean_acc=m, ci95=ci, episodes=episodes, eps_per_sec=eps, meta=meta)
