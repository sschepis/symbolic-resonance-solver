"""Integration tests for the complete SRS solver."""

import numpy as np
import pytest

from srs import SRSSolver, SRSConfig
from srs.constraints import SATClause, Literal


class TestSolverIntegration:
    """Test complete solver integration."""
    
    def test_simple_sat_problem(self):
        """Test solving a simple SAT problem."""
        # Create a simple satisfiable 2-SAT problem
        # (x0 OR x1) AND (NOT x1 OR x2)
        clauses = [
            SATClause([(0, False), (1, False)]),
            SATClause([(1, True), (2, False)])
        ]
        
        # Configure solver with quick settings for testing
        config = SRSConfig(
            particle_count=20,
            max_iterations=100
        )
        
        solver = SRSSolver(config)
        solution = solver.solve(clauses, num_variables=3)
        
        # Check that solution was found
        assert solution is not None
        assert solution.assignment is not None
        assert len(solution.assignment) == 3
        assert solution.metadata.get("iterations", 0) > 0
        assert solution.compute_time > 0
        
        # Solution should satisfy at least some constraints
        assert solution.satisfaction_rate > 0
    
    def test_solver_with_telemetry(self):
        """Test solver with telemetry enabled."""
        clauses = [
            SATClause([(0, False), (1, False)])
        ]
        
        config = SRSConfig(particle_count=10, max_iterations=100)
        solver = SRSSolver(config)
        solution = solver.solve(clauses, num_variables=2, telemetry=True)
        
        # Check telemetry was collected
        assert solution.telemetry is not None
        assert len(solution.telemetry) > 0
        
        # Check telemetry structure
        first_point = solution.telemetry[0]
        assert first_point.step >= 0
        assert first_point.symbolic_entropy >= 0
        assert 0 <= first_point.lyapunov_metric <= 1
        assert 0 <= first_point.satisfaction_rate <= 1
        assert first_point.timestamp is not None
    
    def test_empty_constraints_raises(self):
        """Test that empty constraints list raises error."""
        solver = SRSSolver()
        
        with pytest.raises(ValueError, match="cannot be empty"):
            solver.solve([], num_variables=1)
    
    def test_invalid_num_variables_raises(self):
        """Test that invalid num_variables raises error."""
        clauses = [SATClause([(0, False)])]
        solver = SRSSolver()
        
        with pytest.raises(ValueError, match="at least 1"):
            solver.solve(clauses, num_variables=0)
    
    def test_default_config(self):
        """Test solver with default configuration."""
        clauses = [SATClause([(0, False)])]
        solver = SRSSolver()  # No config specified
        
        assert solver.config is not None
        assert solver.config.particle_count > 0
        assert solver.config.max_iterations > 0
    
    def test_convergence_detection(self):
        """Test that solver detects convergence."""
        # Simple single-clause problem should converge quickly
        clauses = [SATClause([(0, False)])]
        
        config = SRSConfig(
            particle_count=10,
            max_iterations=1000,
            plateau_threshold=0.01
        )
        
        solver = SRSSolver(config)
        solution = solver.solve(clauses, num_variables=1)
        
        # Should stop early due to convergence
        assert solution.metadata['iterations'] < config.max_iterations


class TestProblemClass:
    """Test the Problem convenience class."""
    
    def test_create_problem(self):
        """Test creating a Problem instance."""
        from srs.core.engine import Problem
        
        clauses = [
            SATClause([(0, False), (1, False)])
        ]
        
        problem = Problem(
            name="Test SAT",
            num_variables=2,
            constraints=clauses
        )
        
        assert problem.name == "Test SAT"
        assert problem.num_variables == 2
        assert len(problem.constraints) == 1
    
    def test_problem_solve(self):
        """Test solving via Problem.solve()."""
        from srs.core.engine import Problem
        
        clauses = [SATClause([(0, False)])]
        
        problem = Problem(
            name="Simple",
            num_variables=1,
            constraints=clauses
        )
        
        solution = problem.solve()
        assert solution is not None
        assert solution.assignment is not None
    
    def test_problem_with_metadata(self):
        """Test Problem with metadata."""
        from srs.core.engine import Problem
        
        problem = Problem(
            name="Test",
            num_variables=1,
            constraints=[SATClause([(0, False)])],
            metadata={"difficulty": "easy", "source": "test"}
        )
        
        assert problem.metadata["difficulty"] == "easy"
        assert problem.metadata["source"] == "test"
    
    def test_problem_repr(self):
        """Test Problem string representation."""
        from srs.core.engine import Problem
        
        problem = Problem(
            name="Test Problem",
            num_variables=5,
            constraints=[
                SATClause([(0, False)]),
                SATClause([(1, False)])
            ]
        )
        
        repr_str = repr(problem)
        assert "Test Problem" in repr_str
        assert "5" in repr_str
        assert "2" in repr_str  # 2 constraints