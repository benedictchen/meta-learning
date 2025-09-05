from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Iterable, List, Dict, Tuple
import random
import torch

from .core.episode import Episode, remap_labels

@dataclass
class SyntheticFewShotDataset:
    n_classes: int = 20
    dim: int = 32
    noise: float = 0.1

    def sample_support_query(self, n_way: int, k_shot: int, m_query: int, *, seed: Optional[int]=None):
        g = torch.Generator().manual_seed(seed) if seed is not None else None
        means = torch.randn(n_way, self.dim, generator=g)
        xs = means.repeat_interleave(k_shot, 0) + self.noise*torch.randn(n_way*k_shot, self.dim, generator=g)
        ys = torch.arange(n_way).repeat_interleave(k_shot)
        xq = means.repeat_interleave(m_query, 0) + self.noise*torch.randn(n_way*m_query, self.dim, generator=g)
        yq = torch.arange(n_way).repeat_interleave(m_query)
        return xs, ys.long(), xq, yq.long()

def make_episodes(dataset, n_way: int, k_shot: int, m_query: int, episodes: int) -> Iterable[Episode]:
    for i in range(episodes):
        xs, ys, xq, yq = dataset.sample_support_query(n_way, k_shot, m_query, seed=1337+i)
        ys_m, yq_m = remap_labels(ys, yq)
        ep = Episode(xs, ys_m, xq, yq_m); ep.validate(expect_n_classes=n_way)
        yield ep

# --- Image episodic dataset using torchvision CIFAR100 with CIFAR-FS-style splits
class CIFARFSDataset:
    """Few-shot episodic dataset constructed from CIFAR-100 with class splits.
    Requires torchvision to be installed. Splits can be provided as lists of class indices.
    """
    def __init__(self, root: str, split: str = "train", *, class_splits: Dict[str, List[int]] | None = None, download: bool = False, image_size: int = 32):
        try:
            from torchvision import datasets, transforms
        except Exception as e:
            raise RuntimeError("torchvision is required for CIFARFSDataset. Install extra: pip install 'meta-learning-toolkit[data]'") from e
        if class_splits is None:
            # Default deterministic 64/16/20 split (NOT canonical; replace with official if available)
            rng = random.Random(1234)
            all_classes = list(range(100))
            rng.shuffle(all_classes)
            class_splits = {
                "train": all_classes[:64],
                "val":   all_classes[64:80],
                "test":  all_classes[80:100],
            }
        if split not in class_splits: raise ValueError("split must be one of keys in class_splits")
        self.allowed_classes = set(class_splits[split])
        self.split = split
        self.image_size = image_size
        T = transforms
        mean = (0.5071, 0.4867, 0.4408); std = (0.2675, 0.2565, 0.2761)
        self.transform = T.Compose([T.Resize((image_size, image_size)), T.ToTensor(), T.Normalize(mean, std)])
        self.ds = datasets.CIFAR100(root=root, train=True, download=download)  # use train split; for simplicity

        # Build class -> indices map
        self.class_to_indices: Dict[int, List[int]] = {c: [] for c in self.allowed_classes}
        for idx, y in enumerate(self.ds.targets):
            if int(y) in self.allowed_classes:
                self.class_to_indices[int(y)].append(idx)

    def sample_support_query(self, n_way: int, k_shot: int, m_query: int, *, seed: Optional[int]=None):
        rng = random.Random(seed if seed is not None else 0)
        classes = rng.sample(sorted(self.allowed_classes), n_way)
        xs, ys, xq, yq = [], [], [], []
        for i, c in enumerate(classes):
            pool = self.class_to_indices[c]
            assert len(pool) >= k_shot + m_query, f"Not enough images in class {c}"
            idxs = rng.sample(pool, k_shot + m_query)
            for j in range(k_shot):
                img, _ = self.ds[idxs[j]]
                xs.append(self.transform(img)); ys.append(i)
            for j in range(k_shot, k_shot + m_query):
                img, _ = self.ds[idxs[j]]
                xq.append(self.transform(img)); yq.append(i)
        xs = torch.stack(xs); ys = torch.tensor(ys, dtype=torch.int64)
        xq = torch.stack(xq); yq = torch.tensor(yq, dtype=torch.int64)
        return xs, ys, xq, yq
