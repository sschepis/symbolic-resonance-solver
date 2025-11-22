"""Telemetry types for tracking solver progress."""

from datetime import datetime
from typing import Optional

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