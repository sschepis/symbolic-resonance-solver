# Jupyter Notebooks

Interactive demonstrations of the Symbolic Resonance Solver.

## Available Notebooks

### `solver_demo.ipynb` - Complete Interactive Demo

A comprehensive demonstration covering:

1. **Basic Usage** - Simple SAT problem solving
2. **Convergence Visualization** - 4-panel analysis of solver behavior
3. **Scalability Testing** - Performance across problem sizes (5-15 variables)
4. **Configuration Tuning** - Comparing different solver settings
5. **Entropy Dynamics** - Advanced quantum-inspired algorithm analysis

## Prerequisites

```bash
pip install symbolic-resonance-solver matplotlib seaborn jupyter
```

## Running the Notebooks

```bash
# Navigate to notebooks directory
cd python/notebooks

# Start Jupyter
jupyter notebook

# Or use JupyterLab
jupyter lab
```

## What You'll Learn

- How to create and solve NP-complete problems
- Understanding solver convergence patterns
- Interpreting telemetry and metrics
- Tuning solver parameters for optimal performance
- Visualizing quantum-inspired entropy dynamics

## Visualizations Included

- ✅ Objective value convergence plots
- ✅ Entropy evolution tracking
- ✅ Constraint satisfaction over time
- ✅ Convergence rate analysis
- ✅ Scaling performance charts
- ✅ Configuration comparison plots
- ✅ Entropy-objective phase space
- ✅ Entropy distribution histograms

## Example Output

The notebooks include real-time visualizations showing:
- Solution quality (typically 96-100% satisfaction)
- Fast convergence (usually < 0.2s for 10 variables)
- Clear entropy reduction patterns
- Predictable scaling behavior

## License

These notebooks are part of the symbolic-resonance-solver package.
See the LICENSE file for usage terms.