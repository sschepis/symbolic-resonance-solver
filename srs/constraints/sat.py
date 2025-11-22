"""SAT (Boolean Satisfiability) constraint implementations."""

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
from numpy.typing import NDArray

from srs.constraints.base import Constraint


@dataclass
class Literal:
    """A literal in a SAT clause.
    
    Attributes:
        variable: Variable index
        negated: True if literal is negated (¬x), False otherwise (x)
    """
    variable: int
    negated: bool
    
    def evaluate(self, assignment: NDArray[np.int32]) -> bool:
        """Evaluate literal with given assignment.
        
        Args:
            assignment: Variable assignments
            
        Returns:
            True if literal is satisfied
        """
        if self.variable >= len(assignment):
            return False
        
        value = bool(assignment[self.variable])
        return value if not self.negated else not value
    
    def __str__(self) -> str:
        """String representation."""
        neg = "¬" if self.negated else ""
        return f"{neg}x{self.variable}"
    
    def __repr__(self) -> str:
        """Detailed representation."""
        return f"Literal(variable={self.variable}, negated={self.negated})"


class SATClause(Constraint):
    """A clause in a SAT formula (disjunction of literals).
    
    A clause is satisfied if at least one of its literals is true.
    For example: (x₀ ∨ ¬x₁ ∨ x₂)
    
    Attributes:
        literals: List of literals in the clause
        weight: Importance weight of this clause
    """
    
    def __init__(self, literals: List[Tuple[int, bool]], weight: float = 1.0):
        """Initialize SAT clause.
        
        Args:
            literals: List of (variable_index, negated) tuples
            weight: Clause importance weight
        """
        super().__init__(weight)
        self.literals = [Literal(var, neg) for var, neg in literals]
        
        if not self.literals:
            raise ValueError("SAT clause must have at least one literal")
    
    def evaluate(self, assignment: NDArray[np.int32]) -> bool:
        """Check if clause is satisfied by assignment.
        
        A clause is satisfied if ANY literal is true (OR operation).
        
        Args:
            assignment: Variable assignments
            
        Returns:
            True if at least one literal is satisfied
        """
        return any(lit.evaluate(assignment) for lit in self.literals)
    
    def get_variables(self) -> List[int]:
        """Get all variable indices in this clause.
        
        Returns:
            List of variable indices
        """
        return [lit.variable for lit in self.literals]
    
    def get_type(self) -> str:
        """Get constraint type.
        
        Returns:
            'sat_clause'
        """
        return "sat_clause"
    
    def __len__(self) -> int:
        """Get number of literals in clause."""
        return len(self.literals)
    
    def __str__(self) -> str:
        """String representation of clause."""
        lit_strs = " ∨ ".join(str(lit) for lit in self.literals)
        return f"({lit_strs})"
    
    def __repr__(self) -> str:
        """Detailed representation."""
        return f"SATClause({len(self)} literals, weight={self.weight})"


class KSATClause(SATClause):
    """A k-SAT clause with exactly k literals.
    
    This is a specialized version of SATClause that enforces
    exactly k literals per clause (e.g., 3-SAT has k=3).
    """
    
    def __init__(self, literals: List[Tuple[int, bool]], k: int = 3, weight: float = 1.0):
        """Initialize k-SAT clause.
        
        Args:
            literals: List of (variable_index, negated) tuples
            k: Number of literals required (default: 3 for 3-SAT)
            weight: Clause importance weight
            
        Raises:
            ValueError: If number of literals != k
        """
        super().__init__(literals, weight)
        
        if len(self.literals) != k:
            raise ValueError(f"k-SAT clause requires exactly {k} literals, got {len(self.literals)}")
        
        self.k = k
    
    def get_type(self) -> str:
        """Get constraint type.
        
        Returns:
            'ksat_clause'
        """
        return f"{self.k}sat_clause"
    
    def __repr__(self) -> str:
        """Detailed representation."""
        return f"{self.k}SATClause(weight={self.weight})"