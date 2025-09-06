"""
Legacy Compatibility Layer.

This module provides compatibility for functions that were previously in the algos/ directory
and are now being consolidated into the algorithms/ directory.
"""

# Import the legacy functions from algos/ for now, we'll migrate them properly later
from ..algos.maml import inner_adapt_and_eval, meta_outer_step, ContinualMAML, DualModeMAML
from ..algos.protonet import ProtoHead
from ..algos.ttcs import ttcs_predict

# Re-export for compatibility
__all__ = [
    'inner_adapt_and_eval',
    'meta_outer_step', 
    'ContinualMAML',
    'DualModeMAML',
    'ProtoHead',
    'ttcs_predict'
]