"""SAT problem benchmarks for SRS solver."""

import random
import time
from typing import Optional

import numpy as np

from benchmarks.base import Benchmark, BenchmarkResult
from srs import SRSSolver, SRSConfig
from srs.constraints import SATClause


class SATBenchmark(Benchmark):
    """Benchmark for SAT problems."""
    
    def __init__(self):
        """Initialize SAT benchmark."""
        super().__init__("SAT")
    
    def generate_problem(self, size: int, seed: Optional[int] = None):
        """Generate a random SAT problem.
        
        Args:
            size: Number of variables
            seed: Random seed
            
        Returns:
            List of SAT clauses
        """
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        
        # Generate 3-SAT problem with clause-to-variable ratio of 4.3
        # (This ratio is known to be in the hard satisfiability region)
        num_clauses = int(size * 4.3)
        
        clauses = []
        for _ in range(num_clauses):
            # Pick 3 random variables
            vars = random.sample(range(size), 3)
            
            # Randomly negate each literal
            literals = [(v, random.choice([True, False])) for v in vars]
            
            clauses.append(SATClause(literals))
        
        return clauses
    
    def solve_problem(self, problem, **kwargs):
        """Solve SAT problem and return benchmark result.
        
        Args:
            problem: List of SAT clauses
            **kwargs: Solver parameters
            
        Returns:
            BenchmarkResult
        """
        # Extract num_variables from problem
        all_vars = set()
        for clause in problem:
            all_vars.update(v for v, _ in [(lit.variable, lit.negated) for lit in clause.literals])
        num_variables = max(all_vars) + 1
        
        # Create solver configuration with only valid parameters
        config = SRSConfig(
            particle_count=kwargs.get('particle_count', 50),
            max_iterations=kwargs.get('max_iterations', 1000),
            quantum_factor=kwargs.get('quantum_factor', 0.5)
        )
        
        # Solve problem
        solver = SRSSolver(config)
        start_time = time.time()
        solution = solver.solve(problem, num_variables=num_variables)
        execution_time = time.time() - start_time
        
        # Get average entropy from solution
        entropy = solution.entropy
        
        # Determine convergence iteration
        convergence_iteration = solution.metadata.get('iterations', 0) if solution.metadata.get('converged', False) else None
        
        return BenchmarkResult(
            problem_type="SAT",
            problem_size=num_variables,
            trial=0,  # Will be set by run_trials
            success=solution.feasible,
            satisfaction_rate=solution.satisfaction_rate,
            iterations=solution.metadata.get('iterations', 0),
            execution_time=execution_time,
            entropy_final=entropy,
            convergence_iteration=convergence_iteration
        )


if __name__ == '__main__':
    """Quick test of SAT benchmark."""
    benchmark = SATBenchmark()
    
    print("Testing SAT Benchmark...")
    print("=" * 60)
    
    # Run small test
    results = benchmark.run_trials(
        sizes=[5, 10],
        num_trials=3,
        seed=42,
        particle_count=20,
        max_iterations=100
    )
    
    benchmark.print_summary()