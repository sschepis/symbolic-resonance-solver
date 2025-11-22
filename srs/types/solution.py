"""Solution types for the SRS solver."""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class Solution(BaseModel):
    """Solution returned by the SRS solver.
    
    Attributes:
        assignment: Variable assignment (0/1 for SAT, integers for other problems)
        feasible: Whether the solution satisfies all constraints
        objective: Objective function value (0 for satisfaction problems)
        satisfied: Number of satisfied constraints
        total: Total number of constraints
        energy: Final particle energy
        entropy: Final symbolic entropy
        confidence: Solution confidence score [0, 1]
        found_at: Iteration where solution was found
        compute_time: Total computation time in seconds
        telemetry: Optional telemetry data from solving process
        metadata: Additional problem-specific metadata
    """
    
    assignment: List[int] = Field(
        description="Variable assignment vector"
    )
    
    feasible: bool = Field(
        description="Whether solution satisfies all constraints"
    )
    
    objective: float = Field(
        description="Objective function value"
    )
    
    satisfied: int = Field(
        ge=0,
        description="Number of satisfied constraints"
    )
    
    total: int = Field(
        ge=0,
        description="Total number of constraints"
    )
    
    energy: float = Field(
        default=0.0,
        description="Final particle energy"
    )
    
    entropy: float = Field(
        default=0.0,
        description="Final symbolic entropy"
    )
    
    confidence: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Solution confidence score"
    )
    
    found_at: int = Field(
        default=0,
        ge=0,
        description="Iteration where solution was found"
    )
    
    compute_time: float = Field(
        default=0.0,
        ge=0.0,
        description="Computation time in seconds"
    )
    
    telemetry: Optional[List["TelemetryPoint"]] = Field(
        default=None,
        description="Telemetry data from solving process"
    )
    
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional problem-specific metadata"
    )
    
    class Config:
        """Pydantic configuration."""
        validate_assignment = True
        
    @property
    def satisfaction_rate(self) -> float:
        """Calculate the satisfaction rate."""
        if self.total == 0:
            return 0.0
        return self.satisfied / self.total
    
    @property
    def is_complete(self) -> bool:
        """Check if solution is complete (all constraints satisfied)."""
        return self.satisfied == self.total
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert solution to dictionary."""
        return self.model_dump(exclude_none=True)
    
    def __str__(self) -> str:
        """String representation of solution."""
        status = "FEASIBLE" if self.feasible else "INFEASIBLE"
        return (
            f"Solution({status}, "
            f"satisfied={self.satisfied}/{self.total}, "
            f"confidence={self.confidence:.3f}, "
            f"time={self.compute_time:.3f}s)"
        )


# Forward reference resolution
from srs.types.telemetry import TelemetryPoint
Solution.model_rebuild()