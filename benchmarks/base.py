"""Base benchmark framework for performance testing."""

import json
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Dict, Any, Optional
import numpy as np


@dataclass
class BenchmarkResult:
    """Result from a single benchmark run."""
    
    problem_type: str
    problem_size: int
    trial: int
    success: bool
    satisfaction_rate: float
    iterations: int
    execution_time: float
    entropy_final: float
    convergence_iteration: Optional[int] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class BenchmarkStatistics:
    """Statistical summary of benchmark results."""
    
    problem_type: str
    problem_size: int
    num_trials: int
    success_rate: float
    avg_satisfaction_rate: float
    avg_iterations: float
    avg_execution_time: float
    std_execution_time: float
    min_execution_time: float
    max_execution_time: float
    avg_entropy: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


class Benchmark(ABC):
    """Abstract base class for benchmarks."""
    
    def __init__(self, problem_type: str):
        """Initialize benchmark.
        
        Args:
            problem_type: Type of problem being benchmarked
        """
        self.problem_type = problem_type
        self.results: List[BenchmarkResult] = []
    
    @abstractmethod
    def generate_problem(self, size: int, seed: Optional[int] = None) -> Any:
        """Generate a problem instance.
        
        Args:
            size: Problem size parameter
            seed: Random seed for reproducibility
            
        Returns:
            Problem instance
        """
        pass
    
    @abstractmethod
    def solve_problem(self, problem: Any, **kwargs) -> BenchmarkResult:
        """Solve a problem and return benchmark result.
        
        Args:
            problem: Problem instance to solve
            **kwargs: Additional solver parameters
            
        Returns:
            BenchmarkResult with timing and accuracy info
        """
        pass
    
    def run_trials(
        self,
        sizes: List[int],
        num_trials: int = 10,
        seed: Optional[int] = None,
        **solver_kwargs
    ) -> List[BenchmarkResult]:
        """Run multiple trials across different problem sizes.
        
        Args:
            sizes: List of problem sizes to test
            num_trials: Number of trials per size
            seed: Base random seed
            **solver_kwargs: Additional solver parameters
            
        Returns:
            List of all benchmark results
        """
        results = []
        
        for size in sizes:
            print(f"Benchmarking {self.problem_type} size {size}...")
            
            for trial in range(num_trials):
                # Use different seed for each trial
                trial_seed = None if seed is None else seed + trial
                
                # Generate and solve problem
                problem = self.generate_problem(size, trial_seed)
                result = self.solve_problem(problem, **solver_kwargs)
                
                # Store result
                result.trial = trial
                results.append(result)
                
                print(f"  Trial {trial + 1}/{num_trials}: "
                      f"{'✓' if result.success else '✗'} "
                      f"({result.satisfaction_rate:.1%} satisfied, "
                      f"{result.execution_time:.3f}s)")
        
        self.results.extend(results)
        return results
    
    def compute_statistics(self) -> List[BenchmarkStatistics]:
        """Compute statistics from benchmark results.
        
        Returns:
            List of statistics per problem size
        """
        if not self.results:
            return []
        
        # Group results by problem size
        size_groups: Dict[int, List[BenchmarkResult]] = {}
        for result in self.results:
            size = result.problem_size
            if size not in size_groups:
                size_groups[size] = []
            size_groups[size].append(result)
        
        # Compute statistics for each size
        statistics = []
        for size, results in sorted(size_groups.items()):
            times = [r.execution_time for r in results]
            
            stats = BenchmarkStatistics(
                problem_type=self.problem_type,
                problem_size=size,
                num_trials=len(results),
                success_rate=sum(r.success for r in results) / len(results),
                avg_satisfaction_rate=np.mean([r.satisfaction_rate for r in results]),
                avg_iterations=np.mean([r.iterations for r in results]),
                avg_execution_time=np.mean(times),
                std_execution_time=np.std(times),
                min_execution_time=np.min(times),
                max_execution_time=np.max(times),
                avg_entropy=np.mean([r.entropy_final for r in results])
            )
            statistics.append(stats)
        
        return statistics
    
    def save_results(self, output_dir: Path) -> None:
        """Save benchmark results to files.
        
        Args:
            output_dir: Directory to save results
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save raw results as JSON
        results_file = output_dir / f"{self.problem_type}_results.json"
        with open(results_file, 'w') as f:
            json.dump(
                [r.to_dict() for r in self.results],
                f,
                indent=2
            )
        
        # Save statistics
        stats = self.compute_statistics()
        stats_file = output_dir / f"{self.problem_type}_statistics.json"
        with open(stats_file, 'w') as f:
            json.dump(
                [s.to_dict() for s in stats],
                f,
                indent=2
            )
        
        # Save as CSV for easy analysis
        csv_file = output_dir / f"{self.problem_type}_results.csv"
        with open(csv_file, 'w') as f:
            # Write header
            f.write("problem_size,trial,success,satisfaction_rate,"
                   "iterations,execution_time,entropy_final\n")
            
            # Write data
            for r in self.results:
                f.write(f"{r.problem_size},{r.trial},{r.success},"
                       f"{r.satisfaction_rate},{r.iterations},"
                       f"{r.execution_time},{r.entropy_final}\n")
        
        print(f"Results saved to {output_dir}")
    
    def print_summary(self) -> None:
        """Print summary of benchmark results."""
        stats = self.compute_statistics()
        
        print(f"\n{'='*80}")
        print(f"Benchmark Summary: {self.problem_type}")
        print(f"{'='*80}")
        print(f"{'Size':<10} {'Trials':<8} {'Success':<10} {'Avg Time (s)':<15} "
              f"{'Std Time':<12} {'Avg Sat %':<12}")
        print(f"{'-'*80}")
        
        for s in stats:
            print(f"{s.problem_size:<10} {s.num_trials:<8} "
                  f"{s.success_rate*100:>6.1f}%    "
                  f"{s.avg_execution_time:>8.4f}       "
                  f"{s.std_execution_time:>8.4f}    "
                  f"{s.avg_satisfaction_rate*100:>6.1f}%")
        
        print(f"{'='*80}\n")


class BenchmarkSuite:
    """Suite of benchmarks to run together."""
    
    def __init__(self):
        """Initialize benchmark suite."""
        self.benchmarks: List[Benchmark] = []
    
    def add_benchmark(self, benchmark: Benchmark) -> None:
        """Add a benchmark to the suite.
        
        Args:
            benchmark: Benchmark to add
        """
        self.benchmarks.append(benchmark)
    
    def run_all(
        self,
        sizes: List[int],
        num_trials: int = 10,
        seed: Optional[int] = None,
        **solver_kwargs
    ) -> Dict[str, List[BenchmarkResult]]:
        """Run all benchmarks in the suite.
        
        Args:
            sizes: List of problem sizes to test
            num_trials: Number of trials per size
            seed: Base random seed
            **solver_kwargs: Additional solver parameters
            
        Returns:
            Dictionary mapping problem type to results
        """
        all_results = {}
        
        for benchmark in self.benchmarks:
            print(f"\n{'='*80}")
            print(f"Running {benchmark.problem_type} benchmark")
            print(f"{'='*80}\n")
            
            results = benchmark.run_trials(sizes, num_trials, seed, **solver_kwargs)
            all_results[benchmark.problem_type] = results
            
            benchmark.print_summary()
        
        return all_results
    
    def save_all_results(self, output_dir: Path) -> None:
        """Save results from all benchmarks.
        
        Args:
            output_dir: Directory to save results
        """
        for benchmark in self.benchmarks:
            benchmark.save_results(output_dir)
        
        # Create a summary report
        self.create_summary_report(output_dir)
    
    def create_summary_report(self, output_dir: Path) -> None:
        """Create a summary report of all benchmarks.
        
        Args:
            output_dir: Directory to save report
        """
        report_file = output_dir / "BENCHMARK_SUMMARY.md"
        
        with open(report_file, 'w') as f:
            f.write("# SRS Benchmark Results\n\n")
            f.write(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            for benchmark in self.benchmarks:
                stats = benchmark.compute_statistics()
                
                f.write(f"## {benchmark.problem_type}\n\n")
                f.write("| Size | Trials | Success Rate | Avg Time (s) | "
                       "Std Time | Avg Satisfaction |\n")
                f.write("|------|--------|--------------|--------------|"
                       "----------|------------------|\n")
                
                for s in stats:
                    f.write(f"| {s.problem_size} | {s.num_trials} | "
                           f"{s.success_rate*100:.1f}% | "
                           f"{s.avg_execution_time:.4f} | "
                           f"{s.std_execution_time:.4f} | "
                           f"{s.avg_satisfaction_rate*100:.1f}% |\n")
                
                f.write("\n")
        
        print(f"Summary report saved to {report_file}")