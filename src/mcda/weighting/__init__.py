"""Weighting methods for MCDA.

This module contains implementations of various methods for determining criteria weights.
"""

from mcda.weighting.manual import ManualWeighting
from mcda.weighting.ahp import AHPWeighting
from mcda.weighting.entropy import EntropyWeighting
from mcda.weighting.equal import EqualWeighting

__all__ = [
    "ManualWeighting",
    "AHPWeighting",
    "EntropyWeighting",
    "EqualWeighting"
]
