"""Normalization methods for MCDA.

This module contains implementations of various normalization techniques for decision matrices.
"""

from mcda.normalization.vector import VectorNormalization
from mcda.normalization.linear_minmax import LinearMinMaxNormalization
from mcda.normalization.linear_max import LinearMaxNormalization
from mcda.normalization.linear_sum import LinearSumNormalization

__all__ = [
    "VectorNormalization",
    "LinearMinMaxNormalization",
    "LinearMaxNormalization",
    "LinearSumNormalization"
]
