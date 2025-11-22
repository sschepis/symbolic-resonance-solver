"""Configuration types for the SRS solver."""

from typing import Optional

from pydantic import BaseModel, Field, field_validator


class SRSConfig(BaseModel):
    """Configuration for the Symbolic Resonance Solver.
    
    Attributes:
        particle_count: Number of entropy particles in the solver
        max_iterations: Maximum number of evolution iterations
        plateau_threshold: Threshold for detecting entropy plateau convergence
        entropy_lambda: Weight for prime-aware entropy penalty
        resonance_strength: Strength of inter-particle resonance coupling
        inertia_weight: PSO inertia coefficient for particle velocity
        cognitive_factor: PSO cognitive coefficient (attraction to quantum position)
        social_factor: PSO social coefficient (attraction to best solution)
        quantum_factor: Quantum evolution strength parameter
        timeout_seconds: Maximum time allowed for solving (seconds)
    """
    
    particle_count: int = Field(
        default=50,
        ge=1,
        le=1000,
        description="Number of entropy particles"
    )
    
    max_iterations: int = Field(
        default=5000,
        ge=1,
        le=1000000,
        description="Maximum evolution iterations"
    )
    
    plateau_threshold: float = Field(
        default=1e-6,
        gt=0.0,
        lt=1.0,
        description="Entropy plateau detection threshold"
    )
    
    entropy_lambda: float = Field(
        default=0.02,
        ge=0.0,
        le=1.0,
        description="Prime entropy penalty weight"
    )
    
    resonance_strength: float = Field(
        default=0.8,
        ge=0.0,
        le=1.0,
        description="Inter-particle resonance strength"
    )
    
    inertia_weight: float = Field(
        default=0.9,
        ge=0.0,
        le=1.0,
        description="PSO inertia coefficient"
    )
    
    cognitive_factor: float = Field(
        default=2.0,
        ge=0.0,
        le=5.0,
        description="PSO cognitive factor"
    )
    
    social_factor: float = Field(
        default=2.0,
        ge=0.0,
        le=5.0,
        description="PSO social factor"
    )
    
    quantum_factor: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Quantum evolution strength"
    )
    
    timeout_seconds: int = Field(
        default=300,
        ge=1,
        le=3600,
        description="Solver timeout in seconds"
    )
    
    class Config:
        """Pydantic configuration."""
        validate_assignment = True
        extra = "forbid"
        
    @field_validator("particle_count")
    @classmethod
    def validate_particle_count(cls, v: int) -> int:
        """Validate particle count is reasonable."""
        if v < 10:
            raise ValueError("particle_count should be at least 10 for reliable convergence")
        return v
    
    @field_validator("max_iterations")
    @classmethod
    def validate_max_iterations(cls, v: int) -> int:
        """Validate max iterations is reasonable."""
        if v < 100:
            raise ValueError("max_iterations should be at least 100 for meaningful search")
        return v


class OptimizerConfig(BaseModel):
    """Configuration for specific optimizer behaviors.
    
    Attributes:
        enable_numba: Enable Numba JIT compilation for performance
        enable_parallel: Enable parallel particle evolution
        cache_quantum_states: Cache quantum state computations
        adaptive_parameters: Enable adaptive parameter tuning
    """
    
    enable_numba: bool = Field(
        default=True,
        description="Enable Numba JIT compilation"
    )
    
    enable_parallel: bool = Field(
        default=False,
        description="Enable parallel particle updates"
    )
    
    cache_quantum_states: bool = Field(
        default=True,
        description="Cache quantum state computations"
    )
    
    adaptive_parameters: bool = Field(
        default=True,
        description="Enable adaptive parameter tuning"
    )
    
    class Config:
        """Pydantic configuration."""
        validate_assignment = True
        extra = "forbid"