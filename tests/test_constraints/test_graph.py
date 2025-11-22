"""Tests for graph-based constraint implementations."""

import numpy as np
import pytest

from srs.constraints.graph import (
    HamiltonianConstraint,
    VertexCoverConstraint,
    CliqueConstraint,
    ExactCoverConstraint
)


class TestHamiltonianConstraint:
    """Test HamiltonianConstraint class."""
    
    def test_create_constraint(self):
        """Test creating a Hamiltonian path constraint."""
        # Triangle graph
        adjacency = np.array([
            [0, 1, 1],
            [1, 0, 1],
            [1, 1, 0]
        ], dtype=np.int32)
        constraint = HamiltonianConstraint(adjacency)
        
        assert constraint.num_vertices == 3
        assert constraint.get_type() == "hamiltonian_path"
    
    def test_invalid_adjacency_matrix(self):
        """Test that non-square matrix raises error."""
        adjacency = np.array([[0, 1], [1, 0, 1]])
        with pytest.raises(ValueError, match="must be square"):
            HamiltonianConstraint(adjacency)
    
    def test_evaluate_valid_path_triangle(self):
        """Test valid Hamiltonian path on triangle graph."""
        adjacency = np.array([
            [0, 1, 1],
            [1, 0, 1],
            [1, 1, 0]
        ], dtype=np.int32)
        constraint = HamiltonianConstraint(adjacency)
        
        # Path: 0 -> 1 -> 2
        path = np.array([0, 1, 2], dtype=np.int32)
        assert constraint.evaluate(path) is True
    
    def test_evaluate_invalid_path_disconnected(self):
        """Test invalid path with disconnected vertices."""
        adjacency = np.array([
            [0, 1, 0],
            [1, 0, 0],
            [0, 0, 0]
        ], dtype=np.int32)
        constraint = HamiltonianConstraint(adjacency)
        
        # Path: 0 -> 1 -> 2 (but 1 and 2 not connected)
        path = np.array([0, 1, 2], dtype=np.int32)
        assert constraint.evaluate(path) is False
    
    def test_evaluate_invalid_path_repeated_vertex(self):
        """Test invalid path with repeated vertex."""
        adjacency = np.array([
            [0, 1, 1],
            [1, 0, 1],
            [1, 1, 0]
        ], dtype=np.int32)
        constraint = HamiltonianConstraint(adjacency)
        
        # Path with repeated vertex
        path = np.array([0, 1, 0], dtype=np.int32)
        assert constraint.evaluate(path) is False
    
    def test_evaluate_wrong_length(self):
        """Test path with wrong length."""
        adjacency = np.array([
            [0, 1, 1],
            [1, 0, 1],
            [1, 1, 0]
        ], dtype=np.int32)
        constraint = HamiltonianConstraint(adjacency)
        
        # Path too short
        path = np.array([0, 1], dtype=np.int32)
        assert constraint.evaluate(path) is False
    
    def test_get_variables(self):
        """Test getting variable indices."""
        adjacency = np.array([[0, 1], [1, 0]], dtype=np.int32)
        constraint = HamiltonianConstraint(adjacency)
        variables = constraint.get_variables()
        assert variables == [0, 1]


class TestVertexCoverConstraint:
    """Test VertexCoverConstraint class."""
    
    def test_create_constraint(self):
        """Test creating a vertex cover constraint."""
        adjacency = np.array([
            [0, 1, 1],
            [1, 0, 1],
            [1, 1, 0]
        ], dtype=np.int32)
        constraint = VertexCoverConstraint(adjacency, cover_size=2)
        
        assert constraint.num_vertices == 3
        assert constraint.cover_size == 2
        assert constraint.get_type() == "vertex_cover"
    
    def test_evaluate_valid_cover(self):
        """Test valid vertex cover."""
        # Triangle: need at least 2 vertices to cover all edges
        adjacency = np.array([
            [0, 1, 1],
            [1, 0, 1],
            [1, 1, 0]
        ], dtype=np.int32)
        constraint = VertexCoverConstraint(adjacency, cover_size=2)
        
        # Select vertices 0 and 1
        assignment = np.array([1, 1, 0], dtype=np.int32)
        assert constraint.evaluate(assignment) is True
    
    def test_evaluate_invalid_cover_uncovered_edge(self):
        """Test invalid cover with uncovered edge."""
        adjacency = np.array([
            [0, 1, 1],
            [1, 0, 1],
            [1, 1, 0]
        ], dtype=np.int32)
        constraint = VertexCoverConstraint(adjacency, cover_size=2)
        
        # Select only vertex 0 - doesn't cover edge (1,2)
        assignment = np.array([1, 0, 0], dtype=np.int32)
        assert constraint.evaluate(assignment) is False
    
    def test_evaluate_invalid_cover_wrong_size(self):
        """Test invalid cover with wrong size."""
        adjacency = np.array([
            [0, 1, 0],
            [1, 0, 0],
            [0, 0, 0]
        ], dtype=np.int32)
        constraint = VertexCoverConstraint(adjacency, cover_size=1)
        
        # Select 2 vertices instead of 1
        assignment = np.array([1, 1, 0], dtype=np.int32)
        assert constraint.evaluate(assignment) is False
    
    def test_evaluate_empty_graph(self):
        """Test vertex cover on graph with no edges."""
        adjacency = np.array([
            [0, 0],
            [0, 0]
        ], dtype=np.int32)
        constraint = VertexCoverConstraint(adjacency, cover_size=0)
        
        # No vertices needed
        assignment = np.array([0, 0], dtype=np.int32)
        assert constraint.evaluate(assignment) is True


class TestCliqueConstraint:
    """Test CliqueConstraint class."""
    
    def test_create_constraint(self):
        """Test creating a clique constraint."""
        adjacency = np.array([
            [0, 1, 1],
            [1, 0, 1],
            [1, 1, 0]
        ], dtype=np.int32)
        constraint = CliqueConstraint(adjacency, clique_size=3)
        
        assert constraint.num_vertices == 3
        assert constraint.clique_size == 3
        assert constraint.get_type() == "clique"
    
    def test_evaluate_valid_clique_complete(self):
        """Test valid clique (complete graph)."""
        # K3 (triangle)
        adjacency = np.array([
            [0, 1, 1],
            [1, 0, 1],
            [1, 1, 0]
        ], dtype=np.int32)
        constraint = CliqueConstraint(adjacency, clique_size=3)
        
        # All vertices form a clique
        assignment = np.array([1, 1, 1], dtype=np.int32)
        assert constraint.evaluate(assignment) is True
    
    def test_evaluate_valid_clique_subset(self):
        """Test valid clique from subset of vertices."""
        # Graph with clique {0, 1, 2}
        adjacency = np.array([
            [0, 1, 1, 0],
            [1, 0, 1, 0],
            [1, 1, 0, 0],
            [0, 0, 0, 0]
        ], dtype=np.int32)
        constraint = CliqueConstraint(adjacency, clique_size=3)
        
        # Vertices 0, 1, 2 form a clique
        assignment = np.array([1, 1, 1, 0], dtype=np.int32)
        assert constraint.evaluate(assignment) is True
    
    def test_evaluate_invalid_clique_missing_edge(self):
        """Test invalid clique with missing edge."""
        adjacency = np.array([
            [0, 1, 0],
            [1, 0, 1],
            [0, 1, 0]
        ], dtype=np.int32)
        constraint = CliqueConstraint(adjacency, clique_size=3)
        
        # Not all pairs connected
        assignment = np.array([1, 1, 1], dtype=np.int32)
        assert constraint.evaluate(assignment) is False
    
    def test_evaluate_invalid_clique_wrong_size(self):
        """Test invalid clique with wrong size."""
        adjacency = np.array([
            [0, 1, 1],
            [1, 0, 1],
            [1, 1, 0]
        ], dtype=np.int32)
        constraint = CliqueConstraint(adjacency, clique_size=3)
        
        # Only 2 vertices selected
        assignment = np.array([1, 1, 0], dtype=np.int32)
        assert constraint.evaluate(assignment) is False


class TestExactCoverConstraint:
    """Test ExactCoverConstraint class."""
    
    def test_create_constraint(self):
        """Test creating an exact cover constraint."""
        universe = {1, 2, 3, 4}
        subsets = [
            {1, 2},
            {2, 3},
            {3, 4},
            {1, 4}
        ]
        constraint = ExactCoverConstraint(universe, subsets)
        
        assert constraint.universe_size == 4
        assert constraint.num_subsets == 4
        assert constraint.get_type() == "exact_cover"
    
    def test_evaluate_valid_cover(self):
        """Test valid exact cover."""
        universe = {1, 2, 3, 4}
        subsets = [
            {1, 2},
            {3, 4},
            {1, 3},
            {2, 4}
        ]
        constraint = ExactCoverConstraint(universe, subsets)
        
        # Select subsets 0 and 1: {1,2} ∪ {3,4} = {1,2,3,4}
        assignment = np.array([1, 1, 0, 0], dtype=np.int32)
        assert constraint.evaluate(assignment) is True
    
    def test_evaluate_invalid_cover_incomplete(self):
        """Test invalid cover that doesn't cover all elements."""
        universe = {1, 2, 3, 4}
        subsets = [
            {1, 2},
            {2, 3},
            {3, 4}
        ]
        constraint = ExactCoverConstraint(universe, subsets)
        
        # Only covers {1,2,3}, missing 4
        assignment = np.array([1, 1, 0], dtype=np.int32)
        assert constraint.evaluate(assignment) is False
    
    def test_evaluate_invalid_cover_overlap(self):
        """Test invalid cover with overlapping elements."""
        universe = {1, 2, 3}
        subsets = [
            {1, 2},
            {2, 3}
        ]
        constraint = ExactCoverConstraint(universe, subsets)
        
        # Both subsets selected, element 2 covered twice
        assignment = np.array([1, 1], dtype=np.int32)
        assert constraint.evaluate(assignment) is False
    
    def test_evaluate_empty_selection(self):
        """Test empty selection."""
        universe = {1, 2}
        subsets = [{1}, {2}]
        constraint = ExactCoverConstraint(universe, subsets)
        
        # No subsets selected
        assignment = np.array([0, 0], dtype=np.int32)
        assert constraint.evaluate(assignment) is False
    
    def test_evaluate_classic_example(self):
        """Test classic exact cover example."""
        # Universe = {1, 2, 3, 4, 5, 6, 7}
        # Subsets: A={1,4,7}, B={1,4}, C={4,5,7}, D={3,5,6}, E={2,3,6,7}, F={2,7}
        universe = {1, 2, 3, 4, 5, 6, 7}
        subsets = [
            {1, 4, 7},  # A
            {1, 4},     # B
            {4, 5, 7},  # C
            {3, 5, 6},  # D
            {2, 3, 6, 7},  # E
            {2, 7}      # F
        ]
        constraint = ExactCoverConstraint(universe, subsets)
        
        # Solution: B={1,4}, D={3,5,6}, F={2,7}
        assignment = np.array([0, 1, 0, 1, 0, 1], dtype=np.int32)
        assert constraint.evaluate(assignment) is True


class TestGraphConstraintsIntegration:
    """Integration tests for graph constraints."""
    
    def test_hamiltonian_on_complete_graph(self):
        """Test Hamiltonian path on complete graph K4."""
        # K4 - every vertex connected to every other
        adjacency = np.array([
            [0, 1, 1, 1],
            [1, 0, 1, 1],
            [1, 1, 0, 1],
            [1, 1, 1, 0]
        ], dtype=np.int32)
        constraint = HamiltonianConstraint(adjacency)
        
        # Any permutation is a valid path
        path = np.array([0, 1, 2, 3], dtype=np.int32)
        assert constraint.evaluate(path) is True
    
    def test_vertex_cover_on_star_graph(self):
        """Test vertex cover on star graph."""
        # Star graph: center connected to 3 leaves
        adjacency = np.array([
            [0, 1, 1, 1],
            [1, 0, 0, 0],
            [1, 0, 0, 0],
            [1, 0, 0, 0]
        ], dtype=np.int32)
        constraint = VertexCoverConstraint(adjacency, cover_size=1)
        
        # Just the center vertex covers all edges
        assignment = np.array([1, 0, 0, 0], dtype=np.int32)
        assert constraint.evaluate(assignment) is True
    
    def test_clique_on_bipartite_graph(self):
        """Test that no large clique exists in bipartite graph."""
        # K(2,2) bipartite graph
        adjacency = np.array([
            [0, 0, 1, 1],
            [0, 0, 1, 1],
            [1, 1, 0, 0],
            [1, 1, 0, 0]
        ], dtype=np.int32)
        constraint = CliqueConstraint(adjacency, clique_size=3)
        
        # No 3-clique exists in bipartite graph
        assignment = np.array([1, 1, 1, 0], dtype=np.int32)
        assert constraint.evaluate(assignment) is False