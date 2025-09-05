"""Meta-Learning Toolkit (v3)
Real dataset support (CIFAR-FS), tiny Conv4 backbone, stable API.
"""
from ._version import __version__
from .core.episode import Episode, remap_labels

__all__ = ["Episode", "remap_labels", "__version__"]
