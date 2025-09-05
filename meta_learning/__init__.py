"""Meta-Learning Toolkit (v3 full)
No placeholders: manifests, real loaders, functional MAML, artifacted eval.
"""
from ._version import __version__
from .core.episode import Episode, remap_labels

__all__ = ["Episode", "remap_labels", "__version__"]
