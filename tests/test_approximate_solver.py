"""Tests for the ApproximateSolver class."""

import pytest
import numpy as np

from portopt.core.problem import PortfolioOptProblem
from portopt.solvers.approximate import GeneticSolver


class TestGeneticSolver:
    """Tests for the GeneticSolver class."""
    
    @pytest.fixture
    def sample_problem(self) -> PortfolioOptProblem:
        """Create a sample portfolio optimization problem."""
        n_assets = 10
        n_periods = 100
        
        # Generate random returns
        np.random.seed(42)
        returns = np.random.normal(0.001, 0.02, (n_assets, n_periods))
        
        # Create problem with basic constraints
        return PortfolioOptProblem(
            returns=returns,
            constraints={
                'min_weight': 0.01,
                'max_weight': 0.3,
                'sum_to_one': True
            }
        )
    
    def test_genetic_solver_initialization(self):
        """Test GeneticSolver initialization with default and custom parameters."""
        # Default parameters
        solver = GeneticSolver()
        assert solver.population_size == 100
        assert solver.generations == 50
        assert solver.mutation_rate == 0.1
        assert solver.crossover_rate == 0.8
        
        # Custom parameters
        solver = GeneticSolver(
            population_size=200,
            generations=100,
            mutation_rate=0.2,
            crossover_rate=0.7
        )
        assert solver.population_size == 200
        assert solver.generations == 100
        assert solver.mutation_rate == 0.2
        assert solver.crossover_rate == 0.7
    
    def test_genetic_solver_solve(self, sample_problem):
        """Test that GeneticSolver can solve a basic problem."""
        solver = GeneticSolver(generations=10)  # Reduced generations for faster testing
        result = solver.solve(sample_problem)
        
        # Check that weights sum to 1
        assert np.isclose(np.sum(result.weights), 1.0)
        
        # Check that weights are within bounds
        assert np.all(result.weights >= 0.0)
        assert np.all(result.weights <= sample_problem.constraints['max_weight'])
        
        # Check that solution is feasible
        assert result.feasible
        
        # Check that solve time is recorded
        assert result.solve_time > 0
    
    def test_genetic_solver_with_cardinality_constraint(self):
        """Test GeneticSolver with cardinality constraint."""
        # Create problem with cardinality constraint
        n_assets = 20
        n_periods = 100
        
        np.random.seed(42)
        returns = np.random.normal(0.001, 0.02, (n_assets, n_periods))
        
        problem = PortfolioOptProblem(
            returns=returns,
            constraints={
                'min_weight': 0.05,
                'max_weight': 0.4,
                'sum_to_one': True,
                'cardinality': 5  # Only 5 assets can have non-zero weights
            }
        )
        
        solver = GeneticSolver(generations=10)  # Reduced generations for faster testing
        result = solver.solve(problem)
        
        # Check that weights sum to 1
        assert np.isclose(np.sum(result.weights), 1.0)
        
        # Check cardinality constraint
        non_zero_weights = np.sum(result.weights > problem.constraints['min_weight'])
        assert non_zero_weights <= problem.constraints['cardinality']
        
        # Check that solution is feasible
        assert result.feasible
    
    def test_genetic_solver_with_sector_constraints(self):
        """Test GeneticSolver with sector constraints."""
        # Create problem with sector constraints
        n_assets = 10
        n_periods = 100
        
        np.random.seed(42)
        returns = np.random.normal(0.001, 0.02, (n_assets, n_periods))
        
        # Create classifications (3 sectors)
        from dataclasses import dataclass
        
        @dataclass
        class Classification:
            asset_id: int
            sector: str
        
        classifications = [
            Classification(i, f"Sector{i % 3 + 1}") for i in range(n_assets)
        ]
        
        problem = PortfolioOptProblem(
            returns=returns,
            constraints={
                'min_weight': 0.01,
                'max_weight': 0.3,
                'sum_to_one': True,
                'sector_limits': {
                    'Sector1': 0.4,  # Max 40% in Sector1
                    'Sector2': 0.3,  # Max 30% in Sector2
                    'Sector3': 0.5   # Max 50% in Sector3
                }
            },
            classifications=classifications
        )
        
        solver = GeneticSolver(generations=10)  # Reduced generations for faster testing
        result = solver.solve(problem)
        
        # Check that weights sum to 1
        assert np.isclose(np.sum(result.weights), 1.0)
        
        # Check sector constraints
        for sector_id in range(1, 4):
            sector = f"Sector{sector_id}"
            sector_mask = np.array([c.sector == sector for c in classifications])
            sector_weight = np.sum(result.weights * sector_mask)
            assert sector_weight <= problem.constraints['sector_limits'][sector]
        
        # Check that solution is feasible
        assert result.feasible
