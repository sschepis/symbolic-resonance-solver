"""Base constraint interface for all problem types."""

from abc import ABC, abstractmethod
from typing import List

import numpy as np
from numpy.typing import NDArray


class Constraint(ABC):
    """Abstract base class for all constraints in NP-complete problems.
    
    All constraint implementations must provide:
    1. evaluate() - Check if an assignment satisfies the constraint
    2. get_variables() - Return indices of variables involved
    3. get_weight() - Return importance weight of this constraint
    4. get_type() - Return constraint type identifier
    """
    
    def __init__(self, weight: float = 1.0):
        """Initialize constraint with optional weight.
        
        Args:
            weight: Importance weight for this constraint (default: 1.0)
        """
        self.weight = weight
    
    @abstractmethod
    def evaluate(self, assignment: NDArray[np.int32]) -> bool:
        """Evaluate whether the assignment satisfies this constraint.
        
        Args:
            assignment: Variable assignment vector
            
        Returns:
            True if constraint is satisfied, False otherwise
        """
        pass
    
    @abstractmethod
    def get_variables(self) -> List[int]:
        """Get indices of variables that affect this constraint.
        
        Returns:
            List of variable indices
        """
        pass
    
    def get_weight(self) -> float:
        """Get the weight/importance of this constraint.
        
        Returns:
            Constraint weight
        """
        return self.weight
    
    @abstractmethod
    def get_type(self) -> str:
        """Get the type identifier for this constraint.
        
        Returns:
            Type string (e.g., 'sat_clause', 'subset_sum')
        """
        pass
    
    def __repr__(self) -> str:
        """String representation of constraint."""
        return f"{self.__class__.__name__}(weight={self.weight})"