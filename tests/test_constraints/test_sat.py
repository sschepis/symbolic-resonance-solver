"""Tests for SAT constraint implementations."""

import numpy as np
import pytest

from srs.constraints.sat import KSATClause, Literal, SATClause


class TestLiteral:
    """Test Literal class."""
    
    def test_create_positive_literal(self):
        """Test creating a positive literal."""
        lit = Literal(variable=0, negated=False)
        assert lit.variable == 0
        assert lit.negated is False
    
    def test_create_negative_literal(self):
        """Test creating a negated literal."""
        lit = Literal(variable=1, negated=True)
        assert lit.variable == 1
        assert lit.negated is True
    
    def test_evaluate_positive_true(self):
        """Test evaluating positive literal with true assignment."""
        lit = Literal(0, False)
        assignment = np.array([1], dtype=np.int32)
        assert lit.evaluate(assignment) is True
    
    def test_evaluate_positive_false(self):
        """Test evaluating positive literal with false assignment."""
        lit = Literal(0, False)
        assignment = np.array([0], dtype=np.int32)
        assert lit.evaluate(assignment) is False
    
    def test_evaluate_negative_true(self):
        """Test evaluating negated literal with true assignment."""
        lit = Literal(0, True)
        assignment = np.array([1], dtype=np.int32)
        assert lit.evaluate(assignment) is False
    
    def test_evaluate_negative_false(self):
        """Test evaluating negated literal with false assignment."""
        lit = Literal(0, True)
        assignment = np.array([0], dtype=np.int32)
        assert lit.evaluate(assignment) is True
    
    def test_out_of_bounds(self):
        """Test literal with out-of-bounds variable."""
        lit = Literal(5, False)
        assignment = np.array([1, 0], dtype=np.int32)
        assert lit.evaluate(assignment) is False
    
    def test_string_representation(self):
        """Test string representation."""
        lit_pos = Literal(3, False)
        lit_neg = Literal(5, True)
        assert "x3" in str(lit_pos)
        assert "¬x5" in str(lit_neg)


class TestSATClause:
    """Test SATClause class."""
    
    def test_create_clause(self):
        """Test creating a SAT clause."""
        clause = SATClause([(0, False), (1, True), (2, False)])
        assert len(clause) == 3
        assert clause.get_type() == "sat_clause"
    
    def test_empty_clause_raises(self):
        """Test that empty clause raises error."""
        with pytest.raises(ValueError, match="at least one literal"):
            SATClause([])
    
    def test_evaluate_satisfied_first_literal(self):
        """Test clause satisfied by first literal."""
        clause = SATClause([(0, False), (1, False), (2, False)])
        assignment = np.array([1, 0, 0], dtype=np.int32)
        assert clause.evaluate(assignment) is True
    
    def test_evaluate_satisfied_middle_literal(self):
        """Test clause satisfied by middle literal."""
        clause = SATClause([(0, False), (1, False), (2, False)])
        assignment = np.array([0, 1, 0], dtype=np.int32)
        assert clause.evaluate(assignment) is True
    
    def test_evaluate_satisfied_last_literal(self):
        """Test clause satisfied by last literal."""
        clause = SATClause([(0, False), (1, False), (2, False)])
        assignment = np.array([0, 0, 1], dtype=np.int32)
        assert clause.evaluate(assignment) is True
    
    def test_evaluate_unsatisfied(self):
        """Test unsatisfied clause."""
        clause = SATClause([(0, False), (1, False), (2, False)])
        assignment = np.array([0, 0, 0], dtype=np.int32)
        assert clause.evaluate(assignment) is False
    
    def test_evaluate_with_negation(self):
        """Test clause with negated literals."""
        clause = SATClause([(0, True), (1, False)])  # (¬x0 ∨ x1)
        
        # x0=0, x1=0 -> True ∨ False = True
        assert clause.evaluate(np.array([0, 0], dtype=np.int32)) is True
        
        # x0=1, x1=0 -> False ∨ False = False
        assert clause.evaluate(np.array([1, 0], dtype=np.int32)) is False
        
        # x0=1, x1=1 -> False ∨ True = True
        assert clause.evaluate(np.array([1, 1], dtype=np.int32)) is True
    
    def test_get_variables(self):
        """Test getting variable indices."""
        clause = SATClause([(0, False), (3, True), (7, False)])
        variables = clause.get_variables()
        assert variables == [0, 3, 7]
    
    def test_weight(self):
        """Test clause weight."""
        clause = SATClause([(0, False)], weight=2.5)
        assert clause.get_weight() == 2.5
    
    def test_string_representation(self):
        """Test string representation."""
        clause = SATClause([(0, False), (1, True)])
        s = str(clause)
        assert "x0" in s
        assert "¬x1" in s or "x1" in s


class TestKSATClause:
    """Test KSATClause class."""
    
    def test_create_3sat_clause(self):
        """Test creating a 3-SAT clause."""
        clause = KSATClause([(0, False), (1, True), (2, False)], k=3)
        assert len(clause) == 3
        assert clause.k == 3
        assert clause.get_type() == "3sat_clause"
    
    def test_wrong_literal_count_raises(self):
        """Test that wrong literal count raises error."""
        with pytest.raises(ValueError, match="exactly 3 literals"):
            KSATClause([(0, False), (1, True)], k=3)
    
    def test_2sat_clause(self):
        """Test 2-SAT clause."""
        clause = KSATClause([(0, False), (1, True)], k=2)
        assert clause.k == 2
        assert clause.get_type() == "2sat_clause"
    
    def test_4sat_clause(self):
        """Test 4-SAT clause."""
        literals = [(0, False), (1, True), (2, False), (3, True)]
        clause = KSATClause(literals, k=4)
        assert clause.k == 4
        assert len(clause) == 4
    
    def test_k_sat_evaluation(self):
        """Test that k-SAT clauses evaluate correctly."""
        clause = KSATClause([(0, False), (1, False), (2, False)], k=3)
        
        # Should work same as regular SAT
        assert clause.evaluate(np.array([1, 0, 0], dtype=np.int32)) is True
        assert clause.evaluate(np.array([0, 0, 0], dtype=np.int32)) is False


class TestSATIntegration:
    """Integration tests for SAT constraints."""
    
    def test_multiple_clauses(self):
        """Test multiple clauses together."""
        # (x0 ∨ x1) ∧ (¬x0 ∨ x2) ∧ (¬x1 ∨ ¬x2)
        clauses = [
            SATClause([(0, False), (1, False)]),
            SATClause([(0, True), (2, False)]),
            SATClause([(1, True), (2, True)])
        ]
        
        # Solution: x0=1, x1=0, x2=1
        solution = np.array([1, 0, 1], dtype=np.int32)
        assert all(clause.evaluate(solution) for clause in clauses)
        
        # Non-solution: x0=0, x1=0, x2=0
        non_solution = np.array([0, 0, 0], dtype=np.int32)
        assert not all(clause.evaluate(non_solution) for clause in clauses)
    
    def test_unsatisfiable_formula(self):
        """Test unsatisfiable formula."""
        # (x0) ∧ (¬x0)
        clauses = [
            SATClause([(0, False)]),
            SATClause([(0, True)])
        ]
        
        # No assignment can satisfy both
        for val in [0, 1]:
            assignment = np.array([val], dtype=np.int32)
            assert not all(clause.evaluate(assignment) for clause in clauses)