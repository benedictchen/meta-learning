from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Iterable, List, Dict, Tuple
import random, json, os
import torch
from PIL import Image

from .core.episode import Episode, remap_labels

# ---------- Synthetic vectors ----------
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

# ---------- CIFAR-FS (via torchvision CIFAR-100) with manifest splits ----------
class CIFARFSDataset:
    def __init__(self, root: str, split: str = "val", manifest_path: Optional[str] = None, download: bool = False, image_size: int = 32):
        try:
            from torchvision import datasets, transforms
        except Exception as e:
            raise RuntimeError("torchvision is required. Install 'meta-learning-toolkit[data]'") from e

        with open(manifest_path or os.path.join(os.path.dirname(__file__), "splits", "cifar_fs.json"), "r") as f:
            class_splits = json.load(f)
        if split not in class_splits: raise ValueError(f"split must be one of {list(class_splits.keys())}")
        allowed = set(class_splits[split])

        T = transforms
        mean = (0.5071, 0.4867, 0.4408); std = (0.2675, 0.2565, 0.2761)
        self.transform = T.Compose([T.Resize((image_size, image_size)), T.ToTensor(), T.Normalize(mean, std)])
        self.ds = datasets.CIFAR100(root=root, train=True, download=download)
        self.allowed_classes = allowed

        # class -> indices map
        self.class_to_indices: Dict[int, List[int]] = {c: [] for c in allowed}
        for idx, y in enumerate(self.ds.targets):
            if int(y) in allowed:
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

# ---------- miniImageNet (ImageFolder + CSV split manifests) ----------
class MiniImageNetDataset:
    """miniImageNet few-shot episoder using an ImageFolder dataset and CSV split manifests.
    Expects directory layout:
      root/
        images/               (all images)
        splits/
          train.csv           (filename,labelname per line, header: file,cls)
          val.csv
          test.csv
    """
    def __init__(self, root: str, split: str = "val", image_size: int = 84):
        import csv
        from torchvision import transforms
        self.root = root; self.split = split
        split_csv = os.path.join(root, "splits", f"{split}.csv")
        if not os.path.isfile(split_csv):
            raise FileNotFoundError(f"Missing split CSV: {split_csv}")
        files, labels = [], []
        with open(split_csv, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                files.append(row["file"]); labels.append(row["cls"])
        # Map label names to ids
        uniq = sorted(set(labels))
        self.label_to_id = {name: i for i, name in enumerate(uniq)}
        self.by_class: Dict[int, List[str]] = {self.label_to_id[l]: [] for l in uniq}
        for file, label in zip(files, labels):
            cid = self.label_to_id[label]; self.by_class[cid].append(os.path.join(root, "images", file))
        T = transforms
        self.transform = T.Compose([T.Resize((image_size, image_size)), T.ToTensor()])

    def _load(self, path: str):
        img = Image.open(path).convert("RGB"); return self.transform(img)

    def sample_support_query(self, n_way: int, k_shot: int, m_query: int, *, seed: Optional[int]=None):
        rng = random.Random(seed if seed is not None else 0)
        classes = rng.sample(sorted(self.by_class.keys()), n_way)
        xs, ys, xq, yq = [], [], [], []
        for i in classes:
            pool = self.by_class[i]
            assert len(pool) >= k_shot + m_query, f"Not enough images in class {i}"
            idxs = rng.sample(pool, k_shot + m_query)
            for p in idxs[:k_shot]:
                xs.append(self._load(p)); ys.append(classes.index(i))
            for p in idxs[k_shot:]:
                xq.append(self._load(p)); yq.append(classes.index(i))
        xs = torch.stack(xs); ys = torch.tensor(ys, dtype=torch.int64)
        xq = torch.stack(xq); yq = torch.tensor(yq, dtype=torch.int64)
        return xs, ys, xq, yq

# ---------- Episode generator ----------
def make_episodes(dataset, n_way: int, k_shot: int, m_query: int, episodes: int) -> Iterable[Episode]:
    for i in range(episodes):
        xs, ys, xq, yq = dataset.sample_support_query(n_way, k_shot, m_query, seed=1337+i)
        ys_m, yq_m = remap_labels(ys, yq)
        ep = Episode(xs, ys_m, xq, yq_m); ep.validate(expect_n_classes=n_way)
        yield ep
