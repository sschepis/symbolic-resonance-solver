"""Main Symbolic Resonance Solver engine."""

from typing import List, Optional, Tuple
import numpy as np
import time

from srs.types.config import SRSConfig
from srs.types.solution import Solution
from srs.types.telemetry import TelemetryPoint
from srs.constraints.base import Constraint
from srs.core.hilbert import HilbertSpace
from srs.core.particle import EntropyParticle
from srs.core.state import ProblemSpace


class SRSSolver:
    """
    Symbolic Resonance Solver for NP-complete problems.
    
    Uses quantum-inspired particle swarm optimization in prime-basis
    Hilbert space to find constraint-satisfying solutions.
    
    Example:
        >>> from srs import SRSSolver, SRSConfig
        >>> from srs.constraints import SATClause, Literal
        >>> 
        >>> # Create a simple SAT problem
        >>> clauses = [
        ...     SATClause([Literal(0), Literal(1)]),
        ...     SATClause([Literal(1, negated=True), Literal(2)])
        ... ]
        >>> 
        >>> # Solve it
        >>> config = SRSConfig(particle_count=50, max_iterations=1000)
        >>> solver = SRSSolver(config)
        >>> solution = solver.solve(clauses, num_variables=3)
        >>> 
        >>> print(f"Success: {solution.success}")
        >>> print(f"Assignment: {solution.assignment}")
    """
    
    def __init__(self, config: Optional[SRSConfig] = None):
        """Initialize solver with configuration.
        
        Args:
            config: Solver configuration. If None, uses defaults.
        """
        self.config = config or SRSConfig()
        
    def solve(
        self,
        constraints: List[Constraint],
        num_variables: int,
        telemetry: bool = False
    ) -> Solution:
        """Solve a constraint satisfaction problem.
        
        Args:
            constraints: List of constraint objects to satisfy
            num_variables: Number of variables in the problem
            telemetry: Whether to collect telemetry data
            
        Returns:
            Solution object with results and optional telemetry
            
        Raises:
            ValueError: If constraints list is empty or num_variables < 1
        """
        if not constraints:
            raise ValueError("Constraints list cannot be empty")
        if num_variables < 1:
            raise ValueError("Number of variables must be at least 1")
        
        start_time = time.time()
        
        # Initialize quantum components
        hilbert_space = HilbertSpace(dimension=num_variables)
        problem_space = ProblemSpace(
            dimensions=num_variables,
            variables=num_variables,
            problem_type="sat"  # Default to SAT, could be inferred from constraints
        )
        
        # Initialize particle swarm
        particles = self._initialize_swarm(problem_space)
        
        # Track global best
        global_best_position = None
        global_best_fitness = float('inf')
        global_best_assignment = None
        
        # Telemetry tracking
        telemetry_points: List[TelemetryPoint] = []
        
        # Evolution loop
        converged = False
        iterations = 0
        
        for iteration in range(self.config.max_iterations):
            iterations = iteration + 1
            
            # Evaluate all particles
            for particle in particles:
                # Convert position to assignment
                assignment = problem_space.position_to_assignment(particle.position)
                
                # Calculate fitness (number of unsatisfied constraints)
                fitness = self._calculate_fitness(constraints, assignment)
                
                # Calculate quantum state entropy
                quantum_state = hilbert_space.position_to_state(particle.position)
                entropy = hilbert_space.entropy(quantum_state)
                
                # Update particle state
                particle.entropy = entropy
                particle.satisfied_count = len(constraints) - int(fitness)
                particle.energy = fitness
                
                # Update personal best
                particle.update_best()
                
                # Update global best
                if fitness < global_best_fitness:
                    global_best_fitness = fitness
                    global_best_position = particle.position.copy()
                    global_best_assignment = assignment.copy()
            
            # Collect telemetry
            if telemetry:
                avg_entropy = np.mean([p.entropy for p in particles])
                satisfaction_rate = 1.0 - (global_best_fitness / len(constraints))
                
                # Calculate additional telemetry metrics
                # Lyapunov metric: measure of convergence (inverse of position variance)
                position_variance = np.var([p.position for p in particles])
                lyapunov_metric = 1.0 / (1.0 + position_variance)
                
                # Resonance strength: average quantum coupling
                resonance_strength = self.config.quantum_factor
                
                # Dominance: best satisfaction rate
                dominance = satisfaction_rate
                
                telemetry_points.append(TelemetryPoint(
                    step=iteration,
                    symbolic_entropy=float(avg_entropy),
                    lyapunov_metric=float(lyapunov_metric),
                    satisfaction_rate=float(satisfaction_rate),
                    resonance_strength=float(resonance_strength),
                    dominance=float(dominance)
                ))
            
            # Check for convergence
            if global_best_fitness == 0:
                # Perfect solution found
                converged = True
                break
            
            # Check for convergence based on entropy
            if self._check_convergence(particles):
                converged = True
                break
            
            # Update particle velocities and positions
            for particle in particles:
                # Get quantum position (evolved state projected back)
                quantum_state = hilbert_space.position_to_state(particle.position)
                evolved_state = hilbert_space.evolve_state(
                    quantum_state,
                    dt=0.1,
                    resonance_factor=self.config.quantum_factor
                )
                quantum_position = hilbert_space.state_to_position(
                    evolved_state,
                    problem_space.dimensions
                )
                
                # Update velocity with quantum coupling
                particle.update_velocity(
                    quantum_position,
                    global_best_position,
                    inertia=self.config.inertia_weight,
                    cognitive=self.config.cognitive_factor,
                    social=self.config.social_factor
                )
                
                # Update position (with bounds checking)
                particle.update_position(bounds_check=True)
        
        # Calculate final satisfaction rate
        final_assignment = global_best_assignment
        if final_assignment is None:
            final_assignment = np.zeros(num_variables, dtype=np.int32)
            
        satisfied_count = sum(
            1 for c in constraints if c.evaluate(final_assignment)
        )
        
        execution_time = time.time() - start_time
        
        # Get average final entropy from particles
        avg_entropy = np.mean([p.entropy for p in particles]) if particles else 0.0
        
        return Solution(
            assignment=final_assignment.tolist(),
            feasible=(satisfied_count == len(constraints)),
            objective=global_best_fitness,
            satisfied=satisfied_count,
            total=len(constraints),
            energy=global_best_fitness,
            entropy=avg_entropy,
            confidence=satisfied_count / len(constraints),
            found_at=iterations if converged else 0,
            compute_time=execution_time,
            telemetry=telemetry_points if telemetry else None,
            metadata={"converged": converged, "iterations": iterations}
        )
    
    def _initialize_swarm(
        self,
        problem_space: ProblemSpace
    ) -> List[EntropyParticle]:
        """Initialize particle swarm with random positions.
        
        Args:
            problem_space: Problem space defining bounds
            
        Returns:
            List of initialized particles
        """
        particles = []
        
        # Create Hilbert space for quantum states
        from srs.core.hilbert import HilbertSpace
        hilbert = HilbertSpace(dimension=problem_space.dimensions)
        
        for i in range(self.config.particle_count):
            # Create random quantum state for this particle
            quantum_state = hilbert.random_state()
            
            # Create particle with quantum state
            particle = EntropyParticle(
                particle_id=i,
                problem_space=problem_space,
                quantum_state=quantum_state
            )
            particles.append(particle)
        
        return particles
    
    def _calculate_fitness(
        self,
        constraints: List[Constraint],
        assignment: np.ndarray
    ) -> float:
        """Calculate fitness (lower is better).
        
        Fitness is the weighted sum of unsatisfied constraints.
        
        Args:
            constraints: List of constraints
            assignment: Variable assignment to evaluate
            
        Returns:
            Fitness value (0 = all constraints satisfied)
        """
        fitness = 0.0
        
        for constraint in constraints:
            if not constraint.evaluate(assignment):
                fitness += constraint.get_weight()
        
        return fitness
    
    def _check_convergence(self, particles: List[EntropyParticle]) -> bool:
        """Check if swarm has converged.
        
        Convergence is detected when:
        1. Average entropy is below threshold
        2. Particle positions are clustered
        
        Args:
            particles: List of particles
            
        Returns:
            True if converged, False otherwise
        """
        if len(particles) < 2:
            return False
        
        # Check average entropy
        avg_entropy = np.mean([p.entropy for p in particles if p.entropy != float('inf')])
        if avg_entropy < self.config.plateau_threshold:
            return True
        
        # Check position clustering
        positions = np.array([p.position for p in particles])
        std = np.std(positions, axis=0)
        if np.mean(std) < self.config.plateau_threshold:
            return True
        
        return False


class Problem:
    """
    Convenience class for defining optimization problems.
    
    This class helps organize constraints and metadata about a problem.
    
    Example:
        >>> from srs.core.engine import Problem
        >>> from srs.constraints import SATClause, Literal
        >>> 
        >>> problem = Problem(
        ...     name="Simple SAT",
        ...     num_variables=3,
        ...     constraints=[
        ...         SATClause([Literal(0), Literal(1)]),
        ...         SATClause([Literal(1, negated=True), Literal(2)])
        ...     ]
        ... )
        >>> 
        >>> solver = SRSSolver()
        >>> solution = problem.solve(solver)
    """
    
    def __init__(
        self,
        name: str,
        num_variables: int,
        constraints: List[Constraint],
        metadata: Optional[dict] = None
    ):
        """Initialize problem.
        
        Args:
            name: Problem name/description
            num_variables: Number of variables
            constraints: List of constraints
            metadata: Optional metadata dictionary
        """
        self.name = name
        self.num_variables = num_variables
        self.constraints = constraints
        self.metadata = metadata or {}
    
    def solve(
        self,
        solver: Optional[SRSSolver] = None,
        telemetry: bool = False
    ) -> Solution:
        """Solve this problem.
        
        Args:
            solver: Solver instance. If None, creates default solver.
            telemetry: Whether to collect telemetry
            
        Returns:
            Solution object
        """
        if solver is None:
            solver = SRSSolver()
        
        return solver.solve(
            self.constraints,
            self.num_variables,
            telemetry=telemetry
        )
    
    def __repr__(self) -> str:
        return (
            f"Problem(name='{self.name}', "
            f"num_variables={self.num_variables}, "
            f"num_constraints={len(self.constraints)})"
        )