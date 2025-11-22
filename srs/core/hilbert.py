"""Quantum state and Hilbert space operations for symbolic resonance.

This module implements the prime-basis Hilbert space used by the SRS algorithm.
Quantum states are represented as superpositions over a prime-number basis,
enabling symbolic entropy calculations and resonance-driven convergence.
"""

from typing import Optional

import numpy as np
from numpy.typing import NDArray
from scipy import linalg

try:
    from numba import njit
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    # Fallback decorator that does nothing
    def njit(*args, **kwargs):
        def decorator(func):
            return func
        if len(args) == 1 and callable(args[0]):
            return args[0]
        return decorator


class QuantumState:
    """Represents a quantum state in prime-basis Hilbert space.
    
    The state is a normalized superposition over prime basis states,
    with associated entropy and phase information.
    
    Attributes:
        amplitudes: Complex amplitudes in prime basis
        entropy: von Neumann entropy of the state
        dimension: Hilbert space dimension
    """
    
    def __init__(self, amplitudes: NDArray[np.complex128], normalize: bool = True):
        """Initialize quantum state.
        
        Args:
            amplitudes: Complex amplitude vector
            normalize: Whether to normalize amplitudes (default: True)
        """
        self.amplitudes = np.array(amplitudes, dtype=np.complex128)
        
        if normalize:
            self._normalize()
        
        self.dimension = len(self.amplitudes)
        self.entropy = self._calculate_entropy()
    
    def _normalize(self) -> None:
        """Normalize the quantum state to unit norm."""
        norm = np.sqrt(np.sum(np.abs(self.amplitudes) ** 2))
        if norm > 1e-10:
            self.amplitudes /= norm
    
    def _calculate_entropy(self) -> float:
        """Calculate von Neumann entropy of the state.
        
        Returns:
            Entropy value S = -Tr(ρ log ρ) where ρ = |ψ⟩⟨ψ|
        """
        # Get probability distribution from amplitudes
        probabilities = np.abs(self.amplitudes) ** 2
        
        # Filter out zero probabilities to avoid log(0)
        nonzero_probs = probabilities[probabilities > 1e-10]
        
        if len(nonzero_probs) == 0:
            return 0.0
        
        # Calculate Shannon entropy (approximation of von Neumann for pure states)
        entropy = -np.sum(nonzero_probs * np.log(nonzero_probs))
        return float(entropy)
    
    def update_entropy(self) -> None:
        """Recalculate and update entropy value."""
        self.entropy = self._calculate_entropy()
    
    def norm(self) -> float:
        """Calculate norm of the state vector.
        
        Returns:
            Norm ||ψ||
        """
        return float(np.sqrt(np.sum(np.abs(self.amplitudes) ** 2)))
    
    def normalize(self) -> "QuantumState":
        """Return normalized version of this state.
        
        Returns:
            New normalized QuantumState
        """
        return QuantumState(self.amplitudes.copy(), normalize=True)
    
    def probabilities(self) -> NDArray[np.float64]:
        """Calculate probability distribution |ψ_i|².
        
        Returns:
            Probability array
        """
        return np.abs(self.amplitudes) ** 2
    
    def measure(self) -> int:
        """Perform measurement, collapsing to basis state.
        
        Returns:
            Index of measured basis state
        """
        probs = self.probabilities()
        return int(np.random.choice(len(probs), p=probs / np.sum(probs)))
    
    def copy(self) -> "QuantumState":
        """Create a deep copy of this state.
        
        Returns:
            New QuantumState with copied amplitudes
        """
        return QuantumState(self.amplitudes.copy(), normalize=False)
    
    def __repr__(self) -> str:
        """String representation."""
        return f"QuantumState(dim={self.dimension}, entropy={self.entropy:.4f})"


class HilbertSpace:
    """Prime-basis Hilbert space for symbolic resonance.
    
    Provides operations for state evolution, inner products, and
    quantum-inspired transformations in the symbolic space.
    
    Attributes:
        dimension: Dimension of the Hilbert space
        primes: Prime number basis (first N primes)
    """
    
    def __init__(self, dimension: int = 100):
        """Initialize Hilbert space.
        
        Args:
            dimension: Dimension of the space (number of prime basis states)
        """
        if dimension < 1:
            raise ValueError(f"Dimension must be positive, got {dimension}")
        
        self.dimension = dimension
        self.primes = self._generate_primes(dimension)
    
    @staticmethod
    def _generate_primes(n: int) -> NDArray[np.int32]:
        """Generate first n prime numbers using Sieve of Eratosthenes.
        
        Args:
            n: Number of primes to generate
            
        Returns:
            Array of first n primes
        """
        if n <= 0:
            return np.array([], dtype=np.int32)
        
        # Estimate upper bound for nth prime (Rosser's theorem)
        if n < 6:
            limit = 15
        else:
            limit = int(n * (np.log(n) + np.log(np.log(n)))) + 100
        
        # Sieve of Eratosthenes
        sieve = np.ones(limit, dtype=bool)
        sieve[0] = sieve[1] = False
        
        for i in range(2, int(np.sqrt(limit)) + 1):
            if sieve[i]:
                sieve[i*i::i] = False
        
        primes = np.where(sieve)[0]
        return primes[:n].astype(np.int32)
    
    def create_state(
        self,
        amplitudes: Optional[NDArray[np.complex128]] = None,
        random_init: bool = False
    ) -> QuantumState:
        """Create a quantum state in this Hilbert space.
        
        Args:
            amplitudes: Optional amplitude vector
            random_init: If True and amplitudes=None, initialize randomly
            
        Returns:
            New QuantumState
        """
        if amplitudes is not None:
            if len(amplitudes) != self.dimension:
                raise ValueError(
                    f"Amplitudes dimension {len(amplitudes)} != space dimension {self.dimension}"
                )
            return QuantumState(amplitudes)
        
        if random_init:
            # Random complex amplitudes
            real_part = np.random.randn(self.dimension)
            imag_part = np.random.randn(self.dimension)
            amplitudes = real_part + 1j * imag_part
            return QuantumState(amplitudes)
        
        # Default: uniform superposition
        amplitudes = np.ones(self.dimension, dtype=np.complex128)
        return QuantumState(amplitudes)
    
    def evolve_state(
        self,
        state: QuantumState,
        dt: float = 0.01,
        resonance_factor: float = 0.5
    ) -> QuantumState:
        """Evolve quantum state with resonance dynamics.
        
        Applies unitary evolution with symbolic resonance mixing.
        
        Args:
            state: State to evolve
            dt: Time step for evolution
            resonance_factor: Strength of resonance coupling
            
        Returns:
            Evolved quantum state
        """
        # Create rotation in phase space with prime-dependent phases
        phases = np.exp(-1j * dt * resonance_factor * self.primes[:self.dimension] / self.primes[0])
        
        # Apply phase rotation
        new_amplitudes = state.amplitudes * phases
        
        # Add small random perturbation for exploration
        noise = 0.01 * dt * (np.random.randn(self.dimension) + 1j * np.random.randn(self.dimension))
        new_amplitudes += noise
        
        return QuantumState(new_amplitudes)
    
    def compute_inner_product(
        self,
        state1: QuantumState,
        state2: QuantumState
    ) -> complex:
        """Compute inner product ⟨ψ₁|ψ₂⟩.
        
        Args:
            state1: First state
            state2: Second state
            
        Returns:
            Complex inner product
        """
        if state1.dimension != state2.dimension:
            raise ValueError("States must have same dimension for inner product")
        
        return np.vdot(state1.amplitudes, state2.amplitudes)
    
    def compute_fidelity(
        self,
        state1: QuantumState,
        state2: QuantumState
    ) -> float:
        """Compute fidelity F = |⟨ψ₁|ψ₂⟩|² between states.
        
        Args:
            state1: First state
            state2: Second state
            
        Returns:
            Fidelity value in [0, 1]
        """
        inner_product = self.compute_inner_product(state1, state2)
        return float(np.abs(inner_product) ** 2)
    
    def measure_observable(
        self,
        state: QuantumState,
        observable: NDArray[np.float64]
    ) -> float:
        """Measure expectation value of an observable.
        
        Args:
            state: Quantum state
            observable: Hermitian observable matrix
            
        Returns:
            Expectation value ⟨ψ|O|ψ⟩
        """
        if observable.shape != (self.dimension, self.dimension):
            raise ValueError(f"Observable shape {observable.shape} != ({self.dimension}, {self.dimension})")
        
        # Compute ⟨ψ|O|ψ⟩
        result = np.vdot(state.amplitudes, observable @ state.amplitudes)
        return float(np.real(result))
    
    def apply_projector(
        self,
        state: QuantumState,
        projector_indices: NDArray[np.int32],
        strength: float = 1.0
    ) -> QuantumState:
        """Apply a constraint projector to the state.
        
        Projector keeps amplitudes at specified indices, suppresses others.
        
        Args:
            state: State to project
            projector_indices: Indices to keep
            strength: Projection strength in [0, 1]
            
        Returns:
            Projected state
        """
        new_amplitudes = state.amplitudes.copy()
        
        # Create projection mask
        mask = np.zeros(self.dimension, dtype=np.float64)
        valid_indices = projector_indices[projector_indices < self.dimension]
        mask[valid_indices] = 1.0
        
        # Apply soft projection: (1-α)I + αP
        projection = (1 - strength) + strength * mask
        new_amplitudes *= projection
        
        return QuantumState(new_amplitudes)
    
    def calculate_entanglement_entropy(
        self,
        state: QuantumState,
        subsystem_size: int
    ) -> float:
        """Calculate entanglement entropy for a subsystem.
        
        Args:
            state: Full system state
            subsystem_size: Size of subsystem A
            
        Returns:
            Entanglement entropy S_A
        """
        if subsystem_size >= self.dimension or subsystem_size <= 0:
            return 0.0
        
        # Reshape into subsystem structure
        # This is a simplified version - full implementation would require tensor reshaping
        subsystem_probs = np.abs(state.amplitudes[:subsystem_size]) ** 2
        subsystem_probs /= np.sum(subsystem_probs) + 1e-10
        
        # Calculate entropy
        nonzero = subsystem_probs[subsystem_probs > 1e-10]
        if len(nonzero) == 0:
            return 0.0
        
        return float(-np.sum(nonzero * np.log(nonzero)))
    
    def position_to_state(self, position: NDArray[np.float64]) -> QuantumState:
        """Convert position vector to quantum state.
        
        Maps real-valued position coordinates to complex amplitudes
        in the prime basis.
        
        Args:
            position: Position vector (real-valued)
            
        Returns:
            Quantum state representation
        """
        # Map position to complex amplitudes using prime phases
        amplitudes = np.zeros(self.dimension, dtype=np.complex128)
        
        for i in range(min(len(position), self.dimension)):
            # Use position to modulate amplitude and phase
            magnitude = np.abs(position[i])
            phase = 2 * np.pi * position[i] * self.primes[i] / self.primes[0]
            amplitudes[i] = magnitude * np.exp(1j * phase)
        
        return QuantumState(amplitudes)
    
    def state_to_position(
        self,
        state: QuantumState,
        dimension: int
    ) -> NDArray[np.float64]:
        """Convert quantum state to position vector.
        
        Projects quantum amplitudes back to real-valued coordinates.
        
        Args:
            state: Quantum state
            dimension: Dimension of position vector
            
        Returns:
            Position vector (real-valued)
        """
        position = np.zeros(dimension, dtype=np.float64)
        
        for i in range(min(dimension, state.dimension)):
            # Extract magnitude and phase
            amp = state.amplitudes[i]
            magnitude = np.abs(amp)
            phase = np.angle(amp)
            
            # Map back to position coordinate
            # Use phase modulo to get value in reasonable range
            position[i] = magnitude * np.cos(phase)
        
        return position
    
    def random_state(self) -> QuantumState:
        """Generate random normalized quantum state.
        
        Returns:
            Random quantum state
        """
        return self.create_state(random_init=True)
    
    def inner_product(
        self,
        state1: QuantumState,
        state2: QuantumState
    ) -> complex:
        """Alias for compute_inner_product for convenience.
        
        Args:
            state1: First state
            state2: Second state
            
        Returns:
            Complex inner product
        """
        return self.compute_inner_product(state1, state2)
    
    def fidelity(
        self,
        state1: QuantumState,
        state2: QuantumState
    ) -> float:
        """Alias for compute_fidelity for convenience.
        
        Args:
            state1: First state
            state2: Second state
            
        Returns:
            Fidelity value in [0, 1]
        """
        return self.compute_fidelity(state1, state2)
    
    def entropy(self, state: QuantumState) -> float:
        """Calculate entropy of a quantum state.
        
        Args:
            state: Quantum state
            
        Returns:
            von Neumann entropy
        """
        return state.entropy
    
    def __repr__(self) -> str:
        """String representation."""
        return f"HilbertSpace(dimension={self.dimension})"


# Numba-optimized functions for performance-critical operations

if NUMBA_AVAILABLE:
    @njit
    def _fast_entropy_calc(probabilities: NDArray[np.float64]) -> float:
        """Fast entropy calculation with Numba JIT.
        
        Args:
            probabilities: Probability distribution
            
        Returns:
            Shannon entropy
        """
        entropy = 0.0
        for p in probabilities:
            if p > 1e-10:
                entropy -= p * np.log(p)
        return entropy
    
    @njit
    def _fast_normalize(amplitudes: NDArray[np.complex128]) -> NDArray[np.complex128]:
        """Fast normalization with Numba JIT.
        
        Args:
            amplitudes: Complex amplitude vector
            
        Returns:
            Normalized amplitudes
        """
        norm_sq = 0.0
        for amp in amplitudes:
            norm_sq += amp.real ** 2 + amp.imag ** 2
        
        if norm_sq > 1e-10:
            norm = np.sqrt(norm_sq)
            return amplitudes / norm
        return amplitudes
else:
    # Fallback implementations
    def _fast_entropy_calc(probabilities: NDArray[np.float64]) -> float:
        nonzero = probabilities[probabilities > 1e-10]
        if len(nonzero) == 0:
            return 0.0
        return float(-np.sum(nonzero * np.log(nonzero)))
    
    def _fast_normalize(amplitudes: NDArray[np.complex128]) -> NDArray[np.complex128]:
        norm = np.sqrt(np.sum(np.abs(amplitudes) ** 2))
        if norm > 1e-10:
            return amplitudes / norm
        return amplitudes