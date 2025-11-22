"""Telemetry types for tracking solver progress."""

from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, Field


class TelemetryPoint(BaseModel):
    """Single telemetry data point during solver evolution.
    
    Attributes:
        step: Iteration number
        symbolic_entropy: Average symbolic entropy across particles
        lyapunov_metric: System stability metric (higher = more stable)
        satisfaction_rate: Fraction of constraints satisfied [0, 1]
        resonance_strength: Average resonance coupling strength
        dominance: Maximum constraint satisfaction rate of any particle
        timestamp: Time when this point was recorded
    """
    
    step: int = Field(
        ge=0,
        description="Iteration number"
    )
    
    symbolic_entropy: float = Field(
        ge=0.0,
        description="Average symbolic entropy"
    )
    
    lyapunov_metric: float = Field(
        ge=0.0,
        le=1.0,
        description="System stability metric"
    )
    
    satisfaction_rate: float = Field(
        ge=0.0,
        le=1.0,
        description="Fraction of constraints satisfied"
    )
    
    resonance_strength: float = Field(
        ge=0.0,
        le=1.0,
        description="Average resonance coupling"
    )
    
    dominance: float = Field(
        ge=0.0,
        le=1.0,
        description="Best particle satisfaction rate"
    )
    
    timestamp: datetime = Field(
        default_factory=datetime.now,
        description="Recording timestamp"
    )
    
    class Config:
        """Pydantic configuration."""
        validate_assignment = True
        
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return self.model_dump()
    
    def __str__(self) -> str:
        """String representation."""
        return (
            f"Step {self.step}: "
            f"entropy={self.symbolic_entropy:.4f}, "
            f"satisfaction={self.satisfaction_rate:.3f}, "
            f"lyapunov={self.lyapunov_metric:.3f}"
        )


class Telemetry(BaseModel):
    """Telemetry data collection wrapper providing convenient access to history arrays.
    
    This wraps a list of TelemetryPoint objects and provides array-like access
    to common metrics over time.
    """
    
    points: List[TelemetryPoint] = Field(
        default_factory=list,
        description="List of telemetry points"
    )
    
    class Config:
        """Pydantic configuration."""
        validate_assignment = True
    
    @property
    def entropy_history(self) -> List[float]:
        """Get history of symbolic entropy values."""
        return [p.symbolic_entropy for p in self.points]
    
    @property
    def convergence_history(self) -> List[float]:
        """Get history of convergence (inverse of satisfaction rate).
        
        This represents the objective function over time (lower is better).
        """
        return [1.0 - p.satisfaction_rate for p in self.points]
    
    @property
    def violation_history(self) -> List[float]:
        """Get history of constraint violation rates."""
        return [1.0 - p.satisfaction_rate for p in self.points]
    
    @property
    def satisfaction_history(self) -> List[float]:
        """Get history of satisfaction rates."""
        return [p.satisfaction_rate for p in self.points]
    
    @property
    def lyapunov_history(self) -> List[float]:
        """Get history of Lyapunov metric values."""
        return [p.lyapunov_metric for p in self.points]
    
    @property
    def resonance_history(self) -> List[float]:
        """Get history of resonance strength values."""
        return [p.resonance_strength for p in self.points]
    
    def __len__(self) -> int:
        """Get number of telemetry points."""
        return len(self.points)
    
    def __getitem__(self, index: int) -> TelemetryPoint:
        """Get telemetry point by index."""
        return self.points[index]
    
    def __iter__(self):
        """Iterate over telemetry points."""
        return iter(self.points)