"""Pytest configuration and fixtures for SRS tests."""

import numpy as np
import pytest

from srs.constraints import SATClause, SubsetSumConstraint
from srs.constraints.graph import (
    CliqueConstraint,
    ExactCoverConstraint,
    HamiltonianConstraint,
    VertexCoverConstraint,
)
from srs.core import HilbertSpace, ProblemSpace, QuantumState
from srs.types import SRSConfig


@pytest.fixture
def default_config():
    """Default SRS configuration for testing."""
    return SRSConfig()


@pytest.fixture
def hilbert_space():
    """Standard Hilbert space for testing."""
    return HilbertSpace(dimension=50)


@pytest.fixture
def small_hilbert_space():
    """Small Hilbert space for fast tests."""
    return HilbertSpace(dimension=10)


@pytest.fixture
def quantum_state(hilbert_space):
    """Random quantum state for testing."""
    return hilbert_space.create_state(random_init=True)


@pytest.fixture
def sat_problem_space():
    """Problem space for SAT problems."""
    return ProblemSpace(
        dimensions=5,
        variables=5,
        problem_type="sat",
        bounds=[(0.0, 1.0) for _ in range(5)]
    )


@pytest.fixture
def simple_sat_clause():
    """Simple SAT clause: (x0 ∨ ¬x1 ∨ x2)."""
    return SATClause([(0, False), (1, True), (2, False)])


@pytest.fixture
def satisfying_assignment():
    """Assignment that satisfies the simple clause."""
    return np.array([1, 0, 0], dtype=np.int32)


@pytest.fixture
def unsatisfying_assignment():
    """Assignment that does not satisfy the simple clause."""
    return np.array([0, 0, 0], dtype=np.int32)


@pytest.fixture
def subset_sum_problem():
    """Simple subset sum problem."""
    numbers = [3.0, 34.0, 4.0, 12.0, 5.0, 2.0]
    target = 9.0
    return SubsetSumConstraint(numbers, target)


@pytest.fixture
def subset_sum_solution():
    """Solution to subset sum problem: {3, 4, 2} = 9."""
    return np.array([1, 0, 1, 0, 0, 1], dtype=np.int32)


@pytest.fixture
def triangle_graph():
    """Simple triangle graph for testing."""
    nodes = 3
    edges = [(0, 1), (1, 2), (2, 0)]
    return nodes, edges


@pytest.fixture
def path_graph():
    """Simple path graph: 0-1-2-3."""
    nodes = 4
    edges = [(0, 1), (1, 2), (2, 3)]
    return nodes, edges


@pytest.fixture
def hamiltonian_constraint(triangle_graph):
    """Hamiltonian constraint for triangle graph."""
    nodes, edges = triangle_graph
    return HamiltonianConstraint(nodes, edges)


@pytest.fixture
def vertex_cover_constraint(triangle_graph):
    """Vertex cover constraint for triangle graph."""
    nodes, edges = triangle_graph
    return VertexCoverConstraint(nodes, edges, cover_size=2)


@pytest.fixture
def clique_constraint(triangle_graph):
    """Clique constraint for triangle graph."""
    nodes, edges = triangle_graph
    return CliqueConstraint(nodes, edges, clique_size=3)


@pytest.fixture
def exact_cover_constraint():
    """Exact cover constraint for simple problem."""
    universe = 6
    subsets = [
        [0, 1, 2],
        [3, 4, 5],
        [0, 3],
        [1, 4],
        [2, 5]
    ]
    return ExactCoverConstraint(universe, subsets)


@pytest.fixture(autouse=True)
def reset_random_seed():
    """Reset random seed before each test for reproducibility."""
    np.random.seed(42)
    yield
    np.random.seed()  # Reset to random state after test