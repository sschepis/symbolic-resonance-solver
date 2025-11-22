"""Run all benchmarks and generate reports."""

import argparse
from pathlib import Path
import sys

from benchmarks.base import BenchmarkSuite
from benchmarks.benchmark_sat import SATBenchmark


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Run SRS solver benchmarks'
    )
    
    parser.add_argument(
        '--sizes',
        type=str,
        default='10,20,30,50',
        help='Comma-separated list of problem sizes (default: 10,20,30,50)'
    )
    
    parser.add_argument(
        '--trials',
        type=int,
        default=10,
        help='Number of trials per size (default: 10)'
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility (default: 42)'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='benchmarks/results',
        help='Output directory for results (default: benchmarks/results)'
    )
    
    parser.add_argument(
        '--problems',
        type=str,
        default='all',
        help='Comma-separated list of problems to benchmark '
             '(sat,subset_sum,graph or "all")'
    )
    
    parser.add_argument(
        '--particles',
        type=int,
        default=50,
        help='Number of particles for solver (default: 50)'
    )
    
    parser.add_argument(
        '--max-iterations',
        type=int,
        default=1000,
        help='Maximum iterations for solver (default: 1000)'
    )
    
    return parser.parse_args()


def main():
    """Main benchmark runner."""
    args = parse_args()
    
    # Parse sizes
    sizes = [int(s.strip()) for s in args.sizes.split(',')]
    
    # Parse problems to run
    if args.problems.lower() == 'all':
        problem_types = ['sat', 'subset_sum', 'graph']
    else:
        problem_types = [p.strip().lower() for p in args.problems.split(',')]
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Solver configuration
    solver_config = {
        'particle_count': args.particles,
        'max_iterations': args.max_iterations,
        'quantum_factor': 0.5,
        'inertia': 0.9,
        'cognitive': 2.0,
        'social': 2.0,
        'convergence_threshold': 1e-6,
        'early_stop': True
    }
    
    # Create benchmark suite
    suite = BenchmarkSuite()
    
    # Add benchmarks based on selected problems
    if 'sat' in problem_types:
        suite.add_benchmark(SATBenchmark())
    else:
        print("Note: Only SAT benchmarks are currently implemented")
        print("Defaulting to SAT benchmark...")
        suite.add_benchmark(SATBenchmark())
    
    # Run all benchmarks
    print(f"\nRunning benchmarks:")
    print(f"  Problem sizes: {sizes}")
    print(f"  Trials per size: {args.trials}")
    print(f"  Random seed: {args.seed}")
    print(f"  Output directory: {output_dir}")
    print(f"  Solver config: {solver_config}")
    print()
    
    suite.run_all(
        sizes=sizes,
        num_trials=args.trials,
        seed=args.seed,
        **solver_config
    )
    
    # Save results
    suite.save_all_results(output_dir)
    
    print(f"\n✓ All benchmarks complete!")
    print(f"  Results saved to: {output_dir}")


if __name__ == '__main__':
    main()