"""Graph-based constraint implementations for NP-complete problems."""

from typing import List, Set, Tuple

import numpy as np
from numpy.typing import NDArray

from srs.constraints.base import Constraint


class HamiltonianConstraint(Constraint):
    """Constraint for Hamiltonian Path/Cycle problems.
    
    A Hamiltonian path visits each vertex exactly once.
    The assignment represents a permutation of vertices.
    
    Attributes:
        nodes: Number of nodes in the graph
        edges: List of edges as (from, to) tuples
        adj_matrix: Adjacency matrix for efficient edge lookup
        path_length: Expected path length (usually nodes for path, nodes+1 for cycle)
        weight: Constraint importance weight
    """
    
    def __init__(
        self,
        nodes: int,
        edges: List[Tuple[int, int]],
        path_length: int | None = None,
        weight: float = 1.0
    ):
        """Initialize Hamiltonian constraint.
        
        Args:
            nodes: Number of nodes in graph
            edges: List of edges (undirected)
            path_length: Expected path length (default: nodes)
            weight: Constraint importance weight
        """
        super().__init__(weight)
        self.nodes = nodes
        self.edges = edges
        self.path_length = path_length or nodes
        
        # Build adjacency matrix for O(1) edge lookup
        self.adj_matrix = np.zeros((nodes, nodes), dtype=bool)
        for u, v in edges:
            self.adj_matrix[u, v] = True
            self.adj_matrix[v, u] = True  # Undirected
    
    def evaluate(self, assignment: NDArray[np.int32]) -> bool:
        """Check if assignment represents a valid Hamiltonian path.
        
        Args:
            assignment: Vertex ordering (permutation)
            
        Returns:
            True if path is valid
        """
        if len(assignment) != self.nodes:
            return False
        
        # Check if all vertices visited exactly once
        if len(set(assignment)) != self.nodes:
            return False
        
        # Check if all consecutive vertices are connected
        for i in range(len(assignment) - 1):
            u, v = assignment[i], assignment[i + 1]
            if not (0 <= u < self.nodes and 0 <= v < self.nodes):
                return False
            if not self.adj_matrix[u, v]:
                return False
        
        return True
    
    def get_variables(self) -> List[int]:
        """Get all variable indices.
        
        Returns:
            List [0, 1, ..., nodes-1]
        """
        return list(range(self.nodes))
    
    def get_type(self) -> str:
        """Get constraint type.
        
        Returns:
            'hamiltonian'
        """
        return "hamiltonian"
    
    def __str__(self) -> str:
        """String representation."""
        return f"Hamiltonian(nodes={self.nodes}, edges={len(self.edges)})"


class VertexCoverConstraint(Constraint):
    """Constraint for Vertex Cover problem.
    
    Find a set of vertices such that every edge has at least one
    endpoint in the set, with size <= cover_size.
    
    Attributes:
        nodes: Number of nodes in graph
        edges: List of edges as (u, v) tuples
        cover_size: Maximum allowed cover size
        weight: Constraint importance weight
    """
    
    def __init__(
        self,
        nodes: int,
        edges: List[Tuple[int, int]],
        cover_size: int,
        weight: float = 1.0
    ):
        """Initialize vertex cover constraint.
        
        Args:
            nodes: Number of nodes
            edges: List of edges
            cover_size: Maximum cover size
            weight: Constraint importance weight
        """
        super().__init__(weight)
        self.nodes = nodes
        self.edges = edges
        self.cover_size = cover_size
        
        if cover_size > nodes:
            raise ValueError(f"Cover size {cover_size} exceeds number of nodes {nodes}")
    
    def evaluate(self, assignment: NDArray[np.int32]) -> bool:
        """Check if assignment is a valid vertex cover.
        
        assignment[i] = 1 means vertex i is in the cover.
        
        Args:
            assignment: Binary vector indicating cover vertices
            
        Returns:
            True if valid cover within size limit
        """
        if len(assignment) != self.nodes:
            return False
        
        # Check cover size constraint
        cover_count = np.sum(assignment)
        if cover_count > self.cover_size:
            return False
        
        # Check if all edges are covered
        for u, v in self.edges:
            if u >= len(assignment) or v >= len(assignment):
                continue
            # Edge is covered if at least one endpoint is in cover
            if assignment[u] == 0 and assignment[v] == 0:
                return False
        
        return True
    
    def get_variables(self) -> List[int]:
        """Get all variable indices.
        
        Returns:
            List [0, 1, ..., nodes-1]
        """
        return list(range(self.nodes))
    
    def get_type(self) -> str:
        """Get constraint type.
        
        Returns:
            'vertex_cover'
        """
        return "vertex_cover"
    
    def count_uncovered_edges(self, assignment: NDArray[np.int32]) -> int:
        """Count number of uncovered edges.
        
        Args:
            assignment: Binary cover vector
            
        Returns:
            Number of edges with no endpoint in cover
        """
        uncovered = 0
        for u, v in self.edges:
            if u < len(assignment) and v < len(assignment):
                if assignment[u] == 0 and assignment[v] == 0:
                    uncovered += 1
        return uncovered
    
    def __str__(self) -> str:
        """String representation."""
        return f"VertexCover(nodes={self.nodes}, edges={len(self.edges)}, k={self.cover_size})"


class CliqueConstraint(Constraint):
    """Constraint for Maximum Clique problem.
    
    Find a clique (complete subgraph) of exactly clique_size vertices.
    
    Attributes:
        nodes: Number of nodes in graph
        edges: List of edges as (u, v) tuples
        adj_matrix: Adjacency matrix
        clique_size: Required clique size
        weight: Constraint importance weight
    """
    
    def __init__(
        self,
        nodes: int,
        edges: List[Tuple[int, int]],
        clique_size: int,
        weight: float = 1.0
    ):
        """Initialize clique constraint.
        
        Args:
            nodes: Number of nodes
            edges: List of edges
            clique_size: Required clique size
            weight: Constraint importance weight
        """
        super().__init__(weight)
        self.nodes = nodes
        self.edges = edges
        self.clique_size = clique_size
        
        # Build adjacency matrix
        self.adj_matrix = np.zeros((nodes, nodes), dtype=bool)
        for u, v in edges:
            self.adj_matrix[u, v] = True
            self.adj_matrix[v, u] = True
    
    def evaluate(self, assignment: NDArray[np.int32]) -> bool:
        """Check if assignment represents a valid clique.
        
        assignment[i] = 1 means vertex i is in the clique.
        
        Args:
            assignment: Binary vector indicating clique vertices
            
        Returns:
            True if forms a complete subgraph of required size
        """
        if len(assignment) != self.nodes:
            return False
        
        # Get selected vertices
        selected = np.where(assignment == 1)[0]
        
        # Check size requirement
        if len(selected) != self.clique_size:
            return False
        
        # Check if all pairs are connected (complete subgraph)
        for i in range(len(selected)):
            for j in range(i + 1, len(selected)):
                u, v = selected[i], selected[j]
                if not self.adj_matrix[u, v]:
                    return False
        
        return True
    
    def get_variables(self) -> List[int]:
        """Get all variable indices.
        
        Returns:
            List [0, 1, ..., nodes-1]
        """
        return list(range(self.nodes))
    
    def get_type(self) -> str:
        """Get constraint type.
        
        Returns:
            'clique'
        """
        return "clique"
    
    def count_missing_edges(self, assignment: NDArray[np.int32]) -> int:
        """Count missing edges in selected subgraph.
        
        Args:
            assignment: Binary selection vector
            
        Returns:
            Number of missing edges for complete subgraph
        """
        selected = np.where(assignment == 1)[0]
        missing = 0
        
        for i in range(len(selected)):
            for j in range(i + 1, len(selected)):
                u, v = selected[i], selected[j]
                if not self.adj_matrix[u, v]:
                    missing += 1
        
        return missing
    
    def __str__(self) -> str:
        """String representation."""
        return f"Clique(nodes={self.nodes}, edges={len(self.edges)}, k={self.clique_size})"


class ExactCoverConstraint(Constraint):
    """Constraint for Exact Cover (X3C) problem.
    
    Given a universe and subsets, find a selection of subsets that
    covers each universe element exactly once.
    
    Attributes:
        universe: Number of elements in universe
        subsets: List of subsets (each subset is a list of element indices)
        weight: Constraint importance weight
    """
    
    def __init__(
        self,
        universe: int,
        subsets: List[List[int]],
        weight: float = 1.0
    ):
        """Initialize exact cover constraint.
        
        Args:
            universe: Number of universe elements
            subsets: List of subsets
            weight: Constraint importance weight
        """
        super().__init__(weight)
        self.universe = universe
        self.subsets = [set(s) for s in subsets]  # Convert to sets for efficiency
        
        if not subsets:
            raise ValueError("Subsets list cannot be empty")
    
    def evaluate(self, assignment: NDArray[np.int32]) -> bool:
        """Check if selected subsets form an exact cover.
        
        assignment[i] = 1 means subset i is selected.
        
        Args:
            assignment: Binary vector indicating selected subsets
            
        Returns:
            True if forms exact cover (each element covered once)
        """
        if len(assignment) != len(self.subsets):
            return False
        
        # Track coverage count for each element
        coverage = np.zeros(self.universe, dtype=int)
        
        # Count coverage from selected subsets
        for i, selected in enumerate(assignment):
            if selected == 1:
                for elem in self.subsets[i]:
                    if 0 <= elem < self.universe:
                        coverage[elem] += 1
        
        # Check if all elements covered exactly once
        return np.all(coverage == 1)
    
    def get_variables(self) -> List[int]:
        """Get all variable indices.
        
        Returns:
            List [0, 1, ..., len(subsets)-1]
        """
        return list(range(len(self.subsets)))
    
    def get_type(self) -> str:
        """Get constraint type.
        
        Returns:
            'exact_cover'
        """
        return "exact_cover"
    
    def count_coverage_violations(self, assignment: NDArray[np.int32]) -> int:
        """Count elements not covered exactly once.
        
        Args:
            assignment: Binary selection vector
            
        Returns:
            Number of under/over-covered elements
        """
        coverage = np.zeros(self.universe, dtype=int)
        
        for i, selected in enumerate(assignment):
            if selected == 1:
                for elem in self.subsets[i]:
                    if 0 <= elem < self.universe:
                        coverage[elem] += 1
        
        # Count violations (elements not covered exactly once)
        return np.sum(coverage != 1)
    
    def __str__(self) -> str:
        """String representation."""
        return f"ExactCover(universe={self.universe}, subsets={len(self.subsets)})"