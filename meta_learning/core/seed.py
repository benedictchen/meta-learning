from __future__ import annotations
import os, random
import numpy as np
import torch

def seed_all(seed: int) -> None:
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    try:
        import torch.backends.cudnn as cudnn
        cudnn.deterministic = True; cudnn.benchmark = False
    except Exception:
        pass
