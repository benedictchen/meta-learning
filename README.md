# Meta-Learning Toolkit (v3)

Professional, paper-faithful meta-learning with **real dataset support** and a clean API.

## Install
```bash
# minimal
pip install -e .
# with dev tools
pip install -e .[dev]
# add dataset extras (CIFAR-FS via torchvision)
pip install -e .[data]
```

## Quickstart (synthetic)
```bash
pytest -q
mlfew eval --dataset synthetic --episodes 200 --n-way 5 --k-shot 1 --m-query 15
```

## CIFAR-FS (few-shot from CIFAR-100)
```bash
# downloads CIFAR-100 to ./data by default
mlfew eval --dataset cifar_fs --split val --download --encoder conv4 --device auto   --image-size 32 --n-way 5 --k-shot 1 --m-query 15 --episodes 200
```

## Stable API
```python
from meta_learning import Episode, remap_labels
from meta_learning.algos.protonet import ProtoHead
from meta_learning.models.conv4 import Conv4
from meta_learning.data import SyntheticFewShotDataset, CIFARFSDataset, make_episodes
from meta_learning.eval import evaluate
```

## Notes
- CIFAR-FS splits here are **deterministic defaults** (64/16/20) if you don't provide official splits.
- For canonical results, replace with official splits and report **mean ± 95% CI** over ≥10k episodes.
