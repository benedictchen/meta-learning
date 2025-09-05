"""
Professional benchmarking utilities for few-shot learning evaluation.
"""
import time
import math
import json
from dataclasses import dataclass
from typing import Callable, Sequence, Any, Dict

@dataclass
class BenchResult:
    """Result of benchmark evaluation with statistical confidence."""
    mean_acc: float
    ci95: float
    episodes: int
    eps_per_sec: float
    meta: Dict[str, Any]

def mean_ci95(xs):
    """Calculate mean and 95% confidence interval."""
    n = len(xs)
    m = sum(xs) / n
    var = sum((x - m) ** 2 for x in xs) / (n - 1 if n > 1 else 1)
    sd = var ** 0.5
    ci = 1.96 * sd / (n ** 0.5) if n > 1 else 0.0
    return m, ci

def run_benchmark(
    run_episode: Callable[[], float], 
    episodes: int = 1000, 
    warmup: int = 20, 
    meta: Dict[str, Any] | None = None
) -> BenchResult:
    """
    Run benchmark with proper statistical evaluation.
    
    Based on standard few-shot learning evaluation protocols from:
    - Vinyals et al. (2016) "Matching Networks for One Shot Learning"
    - Snell et al. (2017) "Prototypical Networks for Few-shot Learning"  
    - Finn et al. (2017) "Model-Agnostic Meta-Learning"
    """
    meta = meta or {}
    
    # Warmup phase to stabilize performance
    for _ in range(warmup):
        _ = run_episode()
    
    # Timed evaluation with statistical rigor
    accs = []
    t0 = time.perf_counter()
    for _ in range(episodes):
        accs.append(float(run_episode()))
    dt = time.perf_counter() - t0
    
    eps = episodes / max(dt, 1e-9)
    m, ci = mean_ci95(accs)
    
    return BenchResult(
        mean_acc=m, 
        ci95=ci, 
        episodes=episodes, 
        eps_per_sec=eps, 
        meta=meta
    )

if __name__ == "__main__":
    # Demonstration of benchmark usage
    import random
    
    def toy_episode(): 
        return 0.5 + (random.random() - 0.5) * 0.05
    
    res = run_benchmark(toy_episode, episodes=200)
    print(json.dumps(res.__dict__, indent=2))