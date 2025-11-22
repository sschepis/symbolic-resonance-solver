"""
Symbolic Resonance Solver (SRS)

Revolutionary NP-complete problem solver using symbolic entropy spaces 
and quantum resonance dynamics.
"""

from srs.core.engine import SRSSolver, Problem
from srs.types.config import SRSConfig
from srs.types.solution import Solution
from srs.version import __version__

__all__ = [
    "SRSSolver",
    "Problem",
    "SRSConfig",
    "Solution",
    "__version__",
]