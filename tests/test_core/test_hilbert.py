"""Tests for Hilbert space implementation."""

import numpy as np
import pytest

from srs.core.hilbert import HilbertSpace, QuantumState


class TestHilbertSpace:
    """Test HilbertSpace class."""
    
    def test_create_hilbert_space(self):
        """Test creating a Hilbert space."""
        space = HilbertSpace(dimension=10)
        
        assert space.dimension == 10
        assert len(space.primes) == 10
        assert space.primes[0] == 2
        assert space.primes[-1] == 29  # 10th prime
    
    def test_create_with_custom_max_prime(self):
        """Test creating Hilbert space with custom max prime."""
        space = HilbertSpace(dimension=5, max_prime=20)
        
        assert space.dimension == 5
        assert len(space.primes) == 5
        assert all(p <= 20 for p in space.primes)
    
    def test_generate_primes(self):
        """Test prime number generation."""
        space = HilbertSpace(dimension=6)
        primes = space.primes
        
        # First 6 primes: 2, 3, 5, 7, 11, 13
        expected = [2, 3, 5, 7, 11, 13]
        assert list(primes) == expected
    
    def test_random_state(self):
        """Test generating random quantum state."""
        space = HilbertSpace(dimension=5)
        state = space.random_state()
        
        assert isinstance(state, QuantumState)
        assert len(state.amplitudes) == 5
        # Should be normalized
        assert abs(state.norm() - 1.0) < 1e-6
    
    def test_random_state_normalization(self):
        """Test that random states are properly normalized."""
        space = HilbertSpace(dimension=10)
        
        for _ in range(10):
            state = space.random_state()
            assert abs(state.norm() - 1.0) < 1e-6
    
    def test_evolve_state(self):
        """Test state evolution."""
        space = HilbertSpace(dimension=5)
        state = space.random_state()
        
        evolved = space.evolve_state(state, dt=0.1, resonance_factor=1.0)
        
        assert isinstance(evolved, QuantumState)
        assert len(evolved.amplitudes) == 5
        # Evolution preserves norm
        assert abs(evolved.norm() - 1.0) < 1e-6
    
    def test_evolve_state_multiple_steps(self):
        """Test multiple evolution steps."""
        space = HilbertSpace(dimension=5)
        state = space.random_state()
        
        current = state
        for _ in range(10):
            current = space.evolve_state(current, dt=0.01, resonance_factor=0.5)
            assert abs(current.norm() - 1.0) < 1e-6
    
    def test_inner_product(self):
        """Test inner product calculation."""
        space = HilbertSpace(dimension=5)
        state1 = space.random_state()
        state2 = space.random_state()
        
        inner = space.inner_product(state1, state2)
        
        assert isinstance(inner, complex)
        # Inner product with itself should be 1
        self_inner = space.inner_product(state1, state1)
        assert abs(self_inner - 1.0) < 1e-6
    
    def test_fidelity(self):
        """Test fidelity calculation."""
        space = HilbertSpace(dimension=5)
        state1 = space.random_state()
        state2 = space.random_state()
        
        fidelity = space.fidelity(state1, state2)
        
        assert 0 <= fidelity <= 1
        # Fidelity with itself should be 1
        assert abs(space.fidelity(state1, state1) - 1.0) < 1e-6
    
    def test_entropy(self):
        """Test entropy calculation."""
        space = HilbertSpace(dimension=5)
        state = space.random_state()
        
        entropy = space.entropy(state)
        
        assert entropy >= 0
        # Maximum entropy for pure state is log(dimension)
        assert entropy <= np.log(5) + 0.1  # small tolerance
    
    def test_entropy_pure_state(self):
        """Test entropy of pure state (should be near zero)."""
        space = HilbertSpace(dimension=5)
        # Create pure state |0⟩
        amplitudes = np.zeros(5, dtype=np.complex128)
        amplitudes[0] = 1.0
        state = QuantumState(amplitudes)
        
        entropy = space.entropy(state)
        
        # Pure state should have zero entropy
        assert abs(entropy) < 1e-6
    
    def test_entropy_maximally_mixed(self):
        """Test entropy of maximally mixed state."""
        space = HilbertSpace(dimension=5)
        # Create maximally mixed state
        amplitudes = np.ones(5, dtype=np.complex128) / np.sqrt(5)
        state = QuantumState(amplitudes)
        
        entropy = space.entropy(state)
        
        # Should be close to log(dimension)
        expected = np.log(5)
        assert abs(entropy - expected) < 0.1


class TestQuantumState:
    """Test QuantumState class."""
    
    def test_create_state(self):
        """Test creating a quantum state."""
        amplitudes = np.array([1.0, 0.0, 0.0], dtype=np.complex128)
        state = QuantumState(amplitudes)
        
        assert len(state.amplitudes) == 3
        assert state.amplitudes[0] == 1.0
    
    def test_norm(self):
        """Test norm calculation."""
        amplitudes = np.array([1.0, 0.0, 0.0], dtype=np.complex128)
        state = QuantumState(amplitudes)
        
        assert abs(state.norm() - 1.0) < 1e-10
    
    def test_norm_unnormalized(self):
        """Test norm of unnormalized state."""
        amplitudes = np.array([1.0, 1.0, 1.0], dtype=np.complex128)
        state = QuantumState(amplitudes)
        
        assert abs(state.norm() - np.sqrt(3)) < 1e-10
    
    def test_normalize(self):
        """Test state normalization."""
        amplitudes = np.array([2.0, 2.0, 2.0], dtype=np.complex128)
        state = QuantumState(amplitudes)
        
        normalized = state.normalize()
        
        assert abs(normalized.norm() - 1.0) < 1e-10
        # Check values
        expected_val = 2.0 / np.sqrt(12)
        assert abs(normalized.amplitudes[0] - expected_val) < 1e-10
    
    def test_probabilities(self):
        """Test probability calculation."""
        amplitudes = np.array([1.0/np.sqrt(2), 1.0/np.sqrt(2), 0.0], dtype=np.complex128)
        state = QuantumState(amplitudes)
        
        probs = state.probabilities()
        
        assert len(probs) == 3
        assert abs(probs[0] - 0.5) < 1e-10
        assert abs(probs[1] - 0.5) < 1e-10
        assert abs(probs[2]) < 1e-10
        assert abs(np.sum(probs) - 1.0) < 1e-10
    
    def test_entropy_calculation(self):
        """Test entropy calculation from state."""
        # Maximally mixed state
        amplitudes = np.ones(4, dtype=np.complex128) / 2.0
        state = QuantumState(amplitudes)
        
        entropy = state.entropy()
        
        # Should be log(4) = 2*log(2)
        expected = np.log(4)
        assert abs(entropy - expected) < 1e-10
    
    def test_measure(self):
        """Test measurement."""
        np.random.seed(42)
        amplitudes = np.array([1.0, 0.0, 0.0], dtype=np.complex128)
        state = QuantumState(amplitudes)
        
        # Should always measure 0 for pure state |0⟩
        measurement = state.measure()
        assert measurement == 0
    
    def test_measure_superposition(self):
        """Test measurement of superposition state."""
        np.random.seed(42)
        # Equal superposition
        amplitudes = np.ones(10, dtype=np.complex128) / np.sqrt(10)
        state = QuantumState(amplitudes)
        
        # Measure multiple times
        measurements = [state.measure() for _ in range(100)]
        
        # Should get various outcomes
        unique_outcomes = set(measurements)
        assert len(unique_outcomes) > 1
        # All outcomes should be valid indices
        assert all(0 <= m < 10 for m in measurements)
    
    def test_copy(self):
        """Test state copying."""
        amplitudes = np.array([1.0, 2.0, 3.0], dtype=np.complex128)
        state = QuantumState(amplitudes)
        
        copy = state.copy()
        
        assert np.allclose(copy.amplitudes, state.amplitudes)
        # Ensure it's a deep copy
        copy.amplitudes[0] = 99.0
        assert state.amplitudes[0] != 99.0


class TestHilbertSpaceIntegration:
    """Integration tests for Hilbert space operations."""
    
    def test_evolution_preserves_normalization(self):
        """Test that evolution always preserves normalization."""
        space = HilbertSpace(dimension=10)
        state = space.random_state()
        
        for _ in range(100):
            state = space.evolve_state(state, dt=0.01, resonance_factor=0.5)
            assert abs(state.norm() - 1.0) < 1e-6
    
    def test_inner_product_properties(self):
        """Test mathematical properties of inner product."""
        space = HilbertSpace(dimension=5)
        state1 = space.random_state()
        state2 = space.random_state()
        
        # ⟨ψ|ψ⟩ = 1
        assert abs(space.inner_product(state1, state1) - 1.0) < 1e-6
        
        # ⟨ψ|φ⟩ = ⟨φ|ψ⟩*
        ip12 = space.inner_product(state1, state2)
        ip21 = space.inner_product(state2, state1)
        assert abs(ip12 - np.conj(ip21)) < 1e-6
    
    def test_fidelity_properties(self):
        """Test mathematical properties of fidelity."""
        space = HilbertSpace(dimension=5)
        state1 = space.random_state()
        state2 = space.random_state()
        
        # 0 ≤ F ≤ 1
        fidelity = space.fidelity(state1, state2)
        assert 0 <= fidelity <= 1
        
        # F(ψ,ψ) = 1
        assert abs(space.fidelity(state1, state1) - 1.0) < 1e-6
        
        # F(ψ,φ) = F(φ,ψ)
        assert abs(space.fidelity(state1, state2) - space.fidelity(state2, state1)) < 1e-6
    
    def test_entropy_bounds(self):
        """Test entropy is bounded correctly."""
        space = HilbertSpace(dimension=8)
        
        for _ in range(10):
            state = space.random_state()
            entropy = space.entropy(state)
            
            # 0 ≤ S ≤ log(d)
            assert 0 <= entropy <= np.log(8) + 0.1
    
    def test_large_dimension(self):
        """Test with large dimension."""
        space = HilbertSpace(dimension=100)
        
        assert len(space.primes) == 100
        state = space.random_state()
        assert len(state.amplitudes) == 100
        assert abs(state.norm() - 1.0) < 1e-6