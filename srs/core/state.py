"""Problem space and symbolic state management."""

from typing import Dict, List, Tuple

import numpy as np
from numpy.typing import NDArray


class ProblemSpace:
    """Defines the search space for an NP-complete problem.
    
    Attributes:
        dimensions: Number of dimensions in the space
        variables: Number of problem variables
        bounds: Variable bounds as (min, max) pairs
        problem_type: Type of problem (sat, subset_sum, hamiltonian, etc.)
        metadata: Additional problem-specific information
    """
    
    def __init__(
        self,
        dimensions: int,
        variables: int,
        problem_type: str,
        bounds: List[Tuple[float, float]] | None = None,
        metadata: Dict | None = None
    ):
        """Initialize problem space.
        
        Args:
            dimensions: Space dimensions
            variables: Number of variables
            problem_type: Problem type identifier
            bounds: Optional variable bounds (default: [0, 1] for each)
            metadata: Optional metadata dictionary
        """
        self.dimensions = dimensions
        self.variables = variables
        self.problem_type = problem_type
        
        # Set default bounds if not provided
        if bounds is None:
            if problem_type in {"sat", "3sat", "ksat"}:
                # Boolean variables: [0, 1]
                self.bounds = [(0.0, 1.0) for _ in range(dimensions)]
            else:
                # Default continuous bounds
                self.bounds = [(0.0, float(variables)) for _ in range(dimensions)]
        else:
            if len(bounds) != dimensions:
                raise ValueError(f"Bounds length {len(bounds)} != dimensions {dimensions}")
            self.bounds = bounds
        
        self.metadata = metadata or {}
    
    def is_position_valid(self, position: NDArray[np.float64]) -> bool:
        """Check if a position is within bounds.
        
        Args:
            position: Position vector to check
            
        Returns:
            True if all components within bounds
        """
        if len(position) != self.dimensions:
            return False
        
        for i, (low, high) in enumerate(self.bounds):
            if not (low <= position[i] <= high):
                return False
        return True
    
    def clip_position(self, position: NDArray[np.float64]) -> NDArray[np.float64]:
        """Clip position to bounds.
        
        Args:
            position: Position to clip
            
        Returns:
            Clipped position
        """
        clipped = position.copy()
        for i, (low, high) in enumerate(self.bounds):
            clipped[i] = np.clip(clipped[i], low, high)
        return clipped
    
    def random_position(self) -> NDArray[np.float64]:
        """Generate random position within bounds.
        
        Returns:
            Random position vector
        """
        position = np.zeros(self.dimensions, dtype=np.float64)
        for i, (low, high) in enumerate(self.bounds):
            position[i] = low + np.random.random() * (high - low)
        return position
    
    def position_to_assignment(self, position: NDArray[np.float64]) -> NDArray[np.int32]:
        """Convert continuous position to discrete assignment.
        
        Args:
            position: Continuous position vector
            
        Returns:
            Discrete variable assignment
        """
        assignment = np.zeros(self.variables, dtype=np.int32)
        
        if self.problem_type in {"sat", "3sat", "ksat"}:
            # Boolean: threshold at 0.5
            for i in range(min(len(position), self.variables)):
                assignment[i] = 1 if position[i] > 0.5 else 0
        else:
            # Integer: round to nearest
            for i in range(min(len(position), self.variables)):
                assignment[i] = int(np.round(position[i]))
                # Ensure within variable bounds
                if assignment[i] < 0:
                    assignment[i] = 0
                elif assignment[i] >= self.variables:
                    assignment[i] = self.variables - 1
        
        return assignment
    
    def assignment_to_position(self, assignment: NDArray[np.int32]) -> NDArray[np.float64]:
        """Convert discrete assignment to continuous position.
        
        Args:
            assignment: Discrete variable assignment
            
        Returns:
            Continuous position vector
        """
        position = np.zeros(self.dimensions, dtype=np.float64)
        
        for i in range(min(len(assignment), self.dimensions)):
            position[i] = float(assignment[i])
        
        return position
    
    def __repr__(self) -> str:
        """String representation."""
        return (
            f"ProblemSpace("
            f"dimensions={self.dimensions}, "
            f"variables={self.variables}, "
            f"type='{self.problem_type}')"
        )