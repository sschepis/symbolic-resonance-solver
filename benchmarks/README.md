# SRS Benchmark Suite

This directory contains performance benchmarks for the Symbolic Resonance Solver library.

## Running Benchmarks

```bash
# Run all benchmarks
python -m benchmarks.run_all

# Run specific benchmark
python -m benchmarks.benchmark_sat
python -m benchmarks.benchmark_subset_sum
python -m benchmarks.benchmark_graph

# Run benchmarks with custom parameters
python -m benchmarks.run_all --sizes 10,20,30 --trials 10
```

## Benchmark Results

Results are automatically stored in `benchmarks/results/` with timestamps:
- JSON format for programmatic access
- Markdown format for human reading
- CSV format for data analysis

## Latest Results

See [`results/LATEST.md`](results/LATEST.md) for the most recent benchmark run.

## Performance Targets

Based on the SRS algorithm's polynomial-time complexity O(n³):

| Problem Type | n=10 | n=20 | n=30 | n=50 |
|-------------|------|------|------|------|
| 3-SAT | <0.1s | <0.5s | <2s | <10s |
| Subset Sum | <0.05s | <0.2s | <1s | <5s |
| Hamiltonian | <0.1s | <0.5s | <2s | <10s |
| Vertex Cover | <0.1s | <0.4s | <1.5s | <8s |

Success rate target: >95% across all problem types