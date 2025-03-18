"""Tests for the ConstraintAdapter class."""

import pytest
import numpy as np
from typing import List, Dict, Any

from portopt.core.problem import PortfolioOptProblem
from portopt.solvers.constraint_adapter import ConstraintAdapter
from portopt.constraints.constraint_types import IndustryClassification, AssetClass, CurrencyExposure, CreditProfile


class TestConstraintAdapter:
    """Tests for the ConstraintAdapter class."""
    
    @pytest.fixture
    def sample_problem(self) -> PortfolioOptProblem:
        """Create a sample portfolio optimization problem with various constraints."""
        n_assets = 10
        n_periods = 100
        
        # Generate random returns
        np.random.seed(42)
        returns = np.random.normal(0.001, 0.02, (n_assets, n_periods))
        
        # Create classifications
        classifications = [
            IndustryClassification(
                sector=f"Sector{i % 3 + 1}",
                industry_group=f"Group{i % 5 + 1}",
                industry=f"Industry{i % 7 + 1}",
                sub_industry=f"SubIndustry{i}"
            ) for i in range(n_assets)
        ]
        
        # Create asset classes
        asset_classes = [
            AssetClass(
                primary=f"Class{i % 2 + 1}",
                sub_type=f"SubType{i % 3 + 1}",
                region=f"Region{i % 4 + 1}",
                style=f"Style{i % 2 + 1}" if i % 2 == 0 else None
            ) for i in range(n_assets)
        ]
        
        # Create problem with various constraints
        return PortfolioOptProblem(
            returns=returns,
            constraints={
                'min_weight': 0.01,
                'max_weight': 0.3,
                'sum_to_one': True,
                'sector_limits': {
                    'Sector1': 0.4,
                    'Sector2': 0.5,
                    'Sector3': 0.6
                },
                'asset_class_limits': {
                    'Class1': 0.6,
                    'Class2': 0.7
                }
            },
            classifications=classifications,
            asset_classes=asset_classes
        )
    
    def test_to_scipy_constraints(self, sample_problem):
        """Test conversion to SciPy constraints format."""
        constraints = ConstraintAdapter.to_scipy_constraints(sample_problem)
        
        # Check that constraints list is not empty
        assert len(constraints) > 0
        
        # Check that sum-to-one constraint is included
        sum_to_one_found = False
        for constraint in constraints:
            if constraint['type'] == 'eq':
                # Create a test weight vector
                test_weights = np.ones(sample_problem.n_assets) / sample_problem.n_assets
                if np.isclose(constraint['fun'](test_weights), 0.0):
                    sum_to_one_found = True
                    break
        
        assert sum_to_one_found, "Sum-to-one constraint not found"
        
        # Check that sector constraints are included
        sector_constraints_count = 0
        for constraint in constraints:
            if constraint['type'] == 'ineq' and 'mask' in constraint['fun'].__code__.co_varnames:
                sector_constraints_count += 1
        
        assert sector_constraints_count >= 3, "Sector constraints not properly converted"
    
    def test_to_penalty_functions(self, sample_problem):
        """Test conversion to penalty functions for heuristic solvers."""
        penalties = ConstraintAdapter.to_penalty_functions(sample_problem)
        
        # Check that penalties list is not empty
        assert len(penalties) > 0
        
        # Create a test weight vector
        test_weights = np.ones(sample_problem.n_assets) / sample_problem.n_assets
        
        # Check that all penalties can be evaluated
        for penalty_func, weight in penalties:
            penalty_value = penalty_func(test_weights)
            assert isinstance(penalty_value, (int, float)), "Penalty function should return a numeric value"
            assert isinstance(weight, (int, float)), "Penalty weight should be a numeric value"
    
    def test_to_hamiltonian_terms(self, sample_problem):
        """Test conversion to Hamiltonian terms for quantum solvers."""
        hamiltonian_terms = ConstraintAdapter.to_hamiltonian_terms(sample_problem)
        
        # Check that hamiltonian_terms is not empty
        assert len(hamiltonian_terms) > 0
        
        # Check structure of hamiltonian terms
        for term in hamiltonian_terms:
            assert isinstance(term, tuple), "Hamiltonian term should be a tuple"
            assert len(term) == 2, "Hamiltonian term should have two elements (operator, coefficient)"
            assert isinstance(term[1], (int, float)), "Hamiltonian coefficient should be a numeric value"
    
    def test_create_bounds(self, sample_problem):
        """Test creation of bounds for optimization variables."""
        bounds = ConstraintAdapter.create_bounds(sample_problem)
        
        # Check that bounds list has correct length
        assert len(bounds) == sample_problem.n_assets
        
        # Check that bounds are within expected range
        for lower, upper in bounds:
            assert lower >= 0.0, "Lower bound should be non-negative"
            assert upper <= 1.0, "Upper bound should not exceed 1.0"
            assert lower <= upper, "Lower bound should not exceed upper bound"
            
            # Check specific bounds from the problem
            assert lower >= sample_problem.constraints['min_weight']
            assert upper <= sample_problem.constraints['max_weight']
    
    def test_to_qubo(self, sample_problem):
        """Test conversion to QUBO format for quantum solvers."""
        # Create a smaller problem for QUBO conversion
        small_problem = PortfolioOptProblem(
            returns=sample_problem.returns[:4, :],
            constraints=sample_problem.constraints
        )
        
        qubo = ConstraintAdapter.to_qubo(small_problem)
        
        # Check that QUBO dictionary is not empty
        assert len(qubo) > 0
        
        # Check structure of QUBO dictionary
        for key, value in qubo.items():
            assert isinstance(key, tuple), "QUBO key should be a tuple"
            assert len(key) <= 2, "QUBO key should have at most two elements"
            assert isinstance(value, (int, float)), "QUBO value should be a numeric value"
    
    def test_to_quadratic_program(self, sample_problem):
        """Test conversion to QuadraticProgram for quantum solvers."""
        # Create a smaller problem for QuadraticProgram conversion
        small_problem = PortfolioOptProblem(
            returns=sample_problem.returns[:4, :],
            constraints=sample_problem.constraints
        )
        
        quad_prog = ConstraintAdapter.to_quadratic_program(small_problem)
        
        # Check that the QuadraticProgram has the correct number of variables
        assert quad_prog.get_num_vars() == small_problem.n_assets
        
        # Check that the QuadraticProgram has objective terms
        assert len(quad_prog.objective.quadratic.to_dict()) > 0
        
        # Check that the QuadraticProgram has constraints
        assert quad_prog.get_num_linear_constraints() > 0
