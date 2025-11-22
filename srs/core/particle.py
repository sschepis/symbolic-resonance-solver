"""Entropy particle dynamics for symbolic resonance solver."""

from typing import List

import numpy as np
from numpy.typing import NDArray

from srs.core.hilbert import QuantumState
from srs.core.state import ProblemSpace


class EntropyParticle:
    """Represents a symbolic entropy particle in the problem space.
    
    Particles evolve through the problem space using PSO-style dynamics
    combined with quantum state evolution, enabling resonance-based
    convergence to solutions.
    
    Attributes:
        particle_id: Unique particle identifier
        position: Current position in continuous space
        velocity: Velocity vector
        mass: Particle mass (affects inertia)
        energy: Current energy level
        entropy: Local entropy value
        assignment: Current discrete variable assignment
        constraints_affected: Indices of constraints affecting this particle
        satisfied_count: Number of satisfied constraints
        resonance: Resonance strength with solution space
        quantum_state: Associated quantum state in Hilbert space
        best_position: Personal best position found
        best_energy: Best energy achieved
    """
    
    def __init__(
        self,
        particle_id: int,
        problem_space: ProblemSpace,
        quantum_state: QuantumState,
        position: NDArray[np.float64] | None = None,
        velocity: NDArray[np.float64] | None = None,
        mass: float = 1.0
    ):
        """Initialize entropy particle.
        
        Args:
            particle_id: Unique ID for this particle
            problem_space: Problem space definition
            quantum_state: Associated quantum state
            position: Initial position (random if None)
            velocity: Initial velocity (small random if None)
            mass: Particle mass
        """
        self.particle_id = particle_id
        self.problem_space = problem_space
        self.quantum_state = quantum_state
        self.mass = mass
        
        # Initialize position
        if position is None:
            self.position = problem_space.random_position()
        else:
            self.position = np.array(position, dtype=np.float64)
        
        # Initialize velocity
        if velocity is None:
            self.velocity = 0.1 * (np.random.random(problem_space.dimensions) - 0.5)
        else:
            self.velocity = np.array(velocity, dtype=np.float64)
        
        # Initialize assignment from position
        self.assignment = problem_space.position_to_assignment(self.position)
        
        # Particle state
        self.energy = 0.0
        self.entropy = quantum_state.entropy
        self.constraints_affected: List[int] = []
        self.satisfied_count = 0
        self.resonance = 0.0
        
        # Best position tracking (for PSO)
        self.best_position = self.position.copy()
        self.best_energy = float('inf')
        self.best_satisfied = 0
    
    def update_position(
        self,
        delta_t: float = 1.0,
        bounds_check: bool = True
    ) -> None:
        """Update particle position using velocity.
        
        Args:
            delta_t: Time step
            bounds_check: Whether to enforce position bounds
        """
        # Update position: x(t+1) = x(t) + v(t) * dt
        self.position += self.velocity * delta_t
        
        if bounds_check:
            # Enforce bounds and set velocity to zero at boundaries
            for i, (low, high) in enumerate(self.problem_space.bounds):
                if self.position[i] < low:
                    self.position[i] = low
                    self.velocity[i] = 0.0
                elif self.position[i] > high:
                    self.position[i] = high
                    self.velocity[i] = 0.0
        
        # Update discrete assignment from new position
        self.assignment = self.problem_space.position_to_assignment(self.position)
    
    def update_velocity(
        self,
        quantum_position: NDArray[np.float64],
        global_best_position: NDArray[np.float64] | None,
        inertia: float = 0.9,
        cognitive: float = 2.0,
        social: float = 2.0,
        max_velocity: float = 0.1
    ) -> None:
        """Update velocity using PSO dynamics with quantum guidance.
        
        Implements: v = w*v + c1*r1*(p_best - x) + c2*r2*(g_best - x) + c3*r3*(q - x)
        
        Args:
            quantum_position: Position suggested by quantum state
            global_best_position: Global best position (can be None)
            inertia: Inertia weight (w)
            cognitive: Cognitive coefficient (c1) - attraction to personal best
            social: Social coefficient (c2) - attraction to global best
            max_velocity: Maximum velocity magnitude
        """
        # Inertia term
        v_inertia = inertia * self.velocity
        
        # Cognitive term (personal best attraction)
        r1 = np.random.random(len(self.velocity))
        v_cognitive = cognitive * r1 * (self.best_position - self.position)
        
        # Social term (global best attraction)
        v_social = np.zeros_like(self.velocity)
        if global_best_position is not None:
            r2 = np.random.random(len(self.velocity))
            v_social = social * r2 * (global_best_position - self.position)
        
        # Quantum term (quantum position attraction)
        r3 = np.random.random(len(self.velocity))
        v_quantum = cognitive * r3 * (quantum_position - self.position)
        
        # Update velocity
        self.velocity = v_inertia + v_cognitive + v_social + v_quantum
        
        # Clamp velocity to max
        velocity_magnitude = np.linalg.norm(self.velocity)
        if velocity_magnitude > max_velocity:
            self.velocity = (self.velocity / velocity_magnitude) * max_velocity
    
    def calculate_energy(self, constraint_violations: int, total_constraints: int) -> float:
        """Calculate particle energy based on constraint satisfaction.
        
        Args:
            constraint_violations: Number of violated constraints
            total_constraints: Total number of constraints
            
        Returns:
            Energy value (lower is better)
        """
        # Energy from constraint violations
        violation_energy = float(constraint_violations)
        
        # Kinetic energy component (small contribution)
        kinetic_energy = 0.1 * np.sum(self.position ** 2)
        
        self.energy = violation_energy + kinetic_energy
        return self.energy
    
    def update_best(self) -> bool:
        """Update personal best if current state is better.
        
        Returns:
            True if best was updated
        """
        # Check if current state is better than personal best
        if self.satisfied_count > self.best_satisfied or \
           (self.satisfied_count == self.best_satisfied and self.energy < self.best_energy):
            self.best_position = self.position.copy()
            self.best_energy = self.energy
            self.best_satisfied = self.satisfied_count
            return True
        return False
    
    def calculate_hamming_distance(self, other: "EntropyParticle") -> int:
        """Calculate Hamming distance to another particle.
        
        Args:
            other: Other particle to compare with
            
        Returns:
            Hamming distance (number of differing assignments)
        """
        min_len = min(len(self.assignment), len(other.assignment))
        return int(np.sum(self.assignment[:min_len] != other.assignment[:min_len]))
    
    def get_satisfaction_rate(self, total_constraints: int) -> float:
        """Get fraction of satisfied constraints.
        
        Args:
            total_constraints: Total number of constraints
            
        Returns:
            Satisfaction rate in [0, 1]
        """
        if total_constraints == 0:
            return 0.0
        return self.satisfied_count / total_constraints
    
    def __repr__(self) -> str:
        """String representation."""
        return (
            f"EntropyParticle("
            f"id={self.particle_id}, "
            f"satisfied={self.satisfied_count}, "
            f"energy={self.energy:.3f}, "
            f"entropy={self.entropy:.3f})"
        )
    
    def __str__(self) -> str:
        """Human-readable string."""
        return (
            f"Particle {self.particle_id}: "
            f"{self.satisfied_count} satisfied, "
            f"E={self.energy:.3f}, "
            f"S={self.entropy:.3f}"
        )