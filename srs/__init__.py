"""
Symbolic Resonance Solver (SRS)

Revolutionary NP-complete problem solver using symbolic entropy spaces
and quantum resonance dynamics.
"""

from srs.core.engine import SRSSolver, Problem
from srs.types.config import SRSConfig
from srs.types.solution import Solution
from srs.version import __version__

# Export constraint classes for easy access
from srs.constraints.sat import SATClause, Literal, KSATClause
from srs.constraints.subset_sum import SubsetSumConstraint
from srs.constraints.graph import (
    HamiltonianConstraint,
    VertexCoverConstraint,
    CliqueConstraint,
    ExactCoverConstraint
)

__all__ = [
    # Core classes
    "SRSSolver",
    "Problem",
    "SRSConfig",
    "Solution",
    "__version__",
    # Constraint classes
    "SATClause",
    "Literal",
    "KSATClause",
    "SubsetSumConstraint",
    "HamiltonianConstraint",
    "VertexCoverConstraint",
    "CliqueConstraint",
    "ExactCoverConstraint",
]