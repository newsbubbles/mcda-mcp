"""MCDA methods implementations.

This module contains implementations of various MCDA methods for ranking alternatives.
"""

from mcda.methods.topsis import TOPSIS
from mcda.methods.ahp import AHP
from mcda.methods.vikor import VIKOR
from mcda.methods.promethee import PROMETHEE, PROMETHEE1, PROMETHEE2, PROMETHEE3, PROMETHEE4, PROMETHEE5, PROMETHEE6
from mcda.methods.wsm import WSM
from mcda.methods.wpm import WPM
from mcda.methods.waspas import WASPAS

__all__ = [
    "TOPSIS",
    "AHP",
    "VIKOR",
    "PROMETHEE",  # Backward compatibility, aliases to PROMETHEE2
    "PROMETHEE1",
    "PROMETHEE2",
    "PROMETHEE3",
    "PROMETHEE4",
    "PROMETHEE5",
    "PROMETHEE6",
    "WSM",
    "WPM",
    "WASPAS"
]
