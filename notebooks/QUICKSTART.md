# Quick Start Guide

The notebook has an API mismatch. Use this corrected code:

## Correct Usage

```python
from srs import SRSSolver, Problem, SATClause

# Create clauses
clauses = [
    SATClause([(0, True), (1, False), (2, True)]),
    SATClause([(1, True), (3, True), (4, False)])
]

# Create problem
problem = Problem(
    num_variables=5,
    constraints=clauses,
    name="Simple 3-SAT"
)

# Solve using Problem's solve method
solver = SRSSolver()
solution = problem.solve(solver)  # ✅ CORRECT

# OR solve directly with solver
solver = SRSSolver()
solution = solver.solve(clauses, num_variables=5)  # ✅ ALSO CORRECT

print(f"Solution found: {solution.feasible}")
```

## API Reference

### Problem.solve()
```python
problem.solve(solver=None, telemetry=False)
```

### SRSSolver.solve()
```python
solver.solve(constraints, num_variables, telemetry=False)
```

Both work - choose the one that fits your workflow!