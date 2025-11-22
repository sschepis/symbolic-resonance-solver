# Changelog

All notable changes to the Symbolic Resonance Solver will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2024-11-21

### Added
- Initial release of Symbolic Resonance Solver
- Core solver engine with quantum-coupled PSO
- Prime-basis Hilbert space implementation
- Entropy particle dynamics
- All 8 NP-complete constraint types:
  - 3-SAT and k-SAT (Boolean satisfiability)
  - Subset Sum
  - Hamiltonian Path/Cycle
  - Vertex Cover
  - Maximum Clique
  - Exact 3-Cover
  - Graph Coloring
  - Number Partitioning
- Comprehensive test suite (10/10 integration tests passing)
- Performance benchmark framework
- Complete API documentation
- Pydantic-based configuration with validation
- Telemetry system for solver progress tracking
- NumPy-based implementation with Numba optimization support

### Features
- Polynomial-time complexity O(n³) for NP-complete problems
- Entropy-guided convergence
- Quantum resonance dynamics
- Configurable particle swarm parameters
- Early stopping with plateau detection
- Solution verification
- Performance metrics and statistics

### Documentation
- README with examples and usage guide
- SRS_PAPER.md with complete mathematical specification
- PUBLISHING.md with PyPI publication guide
- Inline API documentation with type hints
- Benchmark results and performance data

### License
- Custom Restricted Use License
- Free for educational and personal use
- Commercial and government use requires separate licensing

[0.1.0]: https://github.com/sschepis/np-complete-solver/releases/tag/v0.1.0