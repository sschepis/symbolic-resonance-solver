"""Subset Sum constraint implementation."""

from typing import List

import numpy as np
from numpy.typing import NDArray

from srs.constraints.base import Constraint


class SubsetSumConstraint(Constraint):
    """Constraint for the Subset Sum problem.
    
    Given a set of numbers and a target sum, find a subset that sums
    to exactly the target value.
    
    Attributes:
        numbers: List of numbers to choose from
        target: Target sum to achieve
        weight: Constraint importance weight
        tolerance: Tolerance for floating-point comparison
    """
    
    def __init__(
        self,
        numbers: List[float],
        target: float,
        weight: float = 1.0,
        tolerance: float = 1e-6
    ):
        """Initialize subset sum constraint.
        
        Args:
            numbers: Numbers to choose from
            target: Target sum value
            weight: Constraint importance weight
            tolerance: Floating-point comparison tolerance
        """
        super().__init__(weight)
        self.numbers = np.array(numbers, dtype=np.float64)
        self.target = float(target)
        self.tolerance = tolerance
        
        if len(self.numbers) == 0:
            raise ValueError("Numbers array cannot be empty")
    
    def evaluate(self, assignment: NDArray[np.int32]) -> bool:
        """Check if subset sums to target.
        
        Assignment[i] = 1 means include numbers[i] in subset.
        
        Args:
            assignment: Binary vector indicating included numbers
            
        Returns:
            True if selected numbers sum to target within tolerance
        """
        if len(assignment) != len(self.numbers):
            return False
        
        # Calculate sum of selected numbers
        selected_sum = np.sum(self.numbers[assignment == 1])
        
        # Check if sum matches target within tolerance
        return abs(selected_sum - self.target) < self.tolerance
    
    def get_variables(self) -> List[int]:
        """Get all variable indices.
        
        Returns:
            List of indices [0, 1, ..., len(numbers)-1]
        """
        return list(range(len(self.numbers)))
    
    def get_type(self) -> str:
        """Get constraint type.
        
        Returns:
            'subset_sum'
        """
        return "subset_sum"
    
    def get_current_sum(self, assignment: NDArray[np.int32]) -> float:
        """Calculate current sum for given assignment.
        
        Args:
            assignment: Binary selection vector
            
        Returns:
            Sum of selected numbers
        """
        if len(assignment) != len(self.numbers):
            return 0.0
        return float(np.sum(self.numbers[assignment == 1]))
    
    def get_distance_to_target(self, assignment: NDArray[np.int32]) -> float:
        """Calculate distance from current sum to target.
        
        Args:
            assignment: Binary selection vector
            
        Returns:
            Absolute difference between current sum and target
        """
        current_sum = self.get_current_sum(assignment)
        return abs(current_sum - self.target)
    
    def __str__(self) -> str:
        """String representation."""
        return f"SubsetSum(numbers={len(self.numbers)}, target={self.target})"
    
    def __repr__(self) -> str:
        """Detailed representation."""
        return (
            f"SubsetSumConstraint("
            f"numbers={self.numbers.tolist()}, "
            f"target={self.target}, "
            f"weight={self.weight})"
        )