"""Tests for Subset Sum constraint implementation."""

import numpy as np
import pytest

from srs.constraints.subset_sum import SubsetSumConstraint


class TestSubsetSumConstraint:
    """Test SubsetSumConstraint class."""
    
    def test_create_constraint(self):
        """Test creating a subset sum constraint."""
        numbers = [3.0, 34.0, 4.0, 12.0, 5.0, 2.0]
        target = 9.0
        constraint = SubsetSumConstraint(numbers, target)
        
        assert len(constraint.numbers) == 6
        assert constraint.target == 9.0
        assert constraint.get_type() == "subset_sum"
    
    def test_empty_numbers_raises(self):
        """Test that empty numbers array raises error."""
        with pytest.raises(ValueError, match="cannot be empty"):
            SubsetSumConstraint([], 10.0)
    
    def test_evaluate_exact_match(self):
        """Test evaluation with exact sum match."""
        constraint = SubsetSumConstraint([3.0, 4.0, 2.0], 9.0)
        # Select all: 3 + 4 + 2 = 9
        assignment = np.array([1, 1, 1], dtype=np.int32)
        assert constraint.evaluate(assignment) is True
    
    def test_evaluate_subset_match(self):
        """Test evaluation with subset matching target."""
        constraint = SubsetSumConstraint([3.0, 34.0, 4.0, 12.0, 5.0, 2.0], 9.0)
        # Select {3, 4, 2} = 9
        assignment = np.array([1, 0, 1, 0, 0, 1], dtype=np.int32)
        assert constraint.evaluate(assignment) is True
    
    def test_evaluate_no_match(self):
        """Test evaluation with sum not matching target."""
        constraint = SubsetSumConstraint([3.0, 4.0, 2.0], 9.0)
        # Select {3, 4} = 7, not 9
        assignment = np.array([1, 1, 0], dtype=np.int32)
        assert constraint.evaluate(assignment) is False
    
    def test_evaluate_empty_subset(self):
        """Test evaluation with no numbers selected."""
        constraint = SubsetSumConstraint([3.0, 4.0, 2.0], 0.0)
        # Select nothing = 0
        assignment = np.array([0, 0, 0], dtype=np.int32)
        assert constraint.evaluate(assignment) is True
    
    def test_evaluate_all_selected(self):
        """Test evaluation with all numbers selected."""
        constraint = SubsetSumConstraint([1.0, 2.0, 3.0, 4.0], 10.0)
        # Select all: 1+2+3+4 = 10
        assignment = np.array([1, 1, 1, 1], dtype=np.int32)
        assert constraint.evaluate(assignment) is True
    
    def test_tolerance(self):
        """Test floating-point tolerance in evaluation."""
        constraint = SubsetSumConstraint([1.0, 2.0, 3.0], 6.0, tolerance=1e-6)
        # Sum slightly off due to floating point
        assignment = np.array([1, 1, 1], dtype=np.int32)
        assert constraint.evaluate(assignment) is True
    
    def test_custom_tolerance(self):
        """Test custom tolerance setting."""
        constraint = SubsetSumConstraint([1.0, 2.0, 3.0], 6.0, tolerance=0.1)
        # Within 0.1 of target
        constraint.numbers = np.array([1.0, 2.0, 2.95])
        assignment = np.array([1, 1, 1], dtype=np.int32)
        assert constraint.evaluate(assignment) is True
    
    def test_get_variables(self):
        """Test getting variable indices."""
        constraint = SubsetSumConstraint([1.0, 2.0, 3.0, 4.0], 10.0)
        variables = constraint.get_variables()
        assert variables == [0, 1, 2, 3]
    
    def test_get_current_sum(self):
        """Test calculating current sum."""
        constraint = SubsetSumConstraint([3.0, 34.0, 4.0, 12.0, 5.0, 2.0], 9.0)
        assignment = np.array([1, 0, 1, 0, 0, 1], dtype=np.int32)
        current_sum = constraint.get_current_sum(assignment)
        assert abs(current_sum - 9.0) < 1e-6
    
    def test_get_distance_to_target(self):
        """Test calculating distance to target."""
        constraint = SubsetSumConstraint([3.0, 4.0, 2.0], 9.0)
        
        # Select {3, 4} = 7, distance = 2
        assignment = np.array([1, 1, 0], dtype=np.int32)
        distance = constraint.get_distance_to_target(assignment)
        assert abs(distance - 2.0) < 1e-6
    
    def test_wrong_assignment_length(self):
        """Test with wrong assignment length."""
        constraint = SubsetSumConstraint([1.0, 2.0, 3.0], 6.0)
        # Wrong length assignment
        assignment = np.array([1, 1], dtype=np.int32)
        assert constraint.evaluate(assignment) is False
    
    def test_negative_numbers(self):
        """Test with negative numbers."""
        constraint = SubsetSumConstraint([-3.0, 5.0, -1.0, 4.0], 5.0)
        # Select {5} or {-3, 5, -1, 4}
        assignment = np.array([0, 1, 0, 0], dtype=np.int32)
        assert constraint.evaluate(assignment) is True
    
    def test_weight(self):
        """Test constraint weight."""
        constraint = SubsetSumConstraint([1.0, 2.0], 3.0, weight=2.5)
        assert constraint.get_weight() == 2.5
    
    def test_string_representation(self):
        """Test string representation."""
        constraint = SubsetSumConstraint([1.0, 2.0, 3.0], 6.0)
        s = str(constraint)
        assert "SubsetSum" in s
        assert "3" in s
        assert "6" in s


class TestSubsetSumEdgeCases:
    """Test edge cases for subset sum."""
    
    def test_large_numbers(self):
        """Test with large numbers."""
        numbers = [1000000.0, 2000000.0, 3000000.0]
        target = 6000000.0
        constraint = SubsetSumConstraint(numbers, target)
        assignment = np.array([1, 1, 1], dtype=np.int32)
        assert constraint.evaluate(assignment) is True
    
    def test_zero_target(self):
        """Test with zero target."""
        constraint = SubsetSumConstraint([1.0, -1.0, 2.0, -2.0], 0.0)
        # Select {1, -1}
        assignment = np.array([1, 1, 0, 0], dtype=np.int32)
        assert constraint.evaluate(assignment) is True
    
    def test_single_number(self):
        """Test with single number."""
        constraint = SubsetSumConstraint([42.0], 42.0)
        assignment = np.array([1], dtype=np.int32)
        assert constraint.evaluate(assignment) is True
    
    def test_fractional_numbers(self):
        """Test with fractional numbers."""
        constraint = SubsetSumConstraint([1.5, 2.5, 3.5], 7.5)
        assignment = np.array([1, 1, 1], dtype=np.int32)
        assert constraint.evaluate(assignment) is True