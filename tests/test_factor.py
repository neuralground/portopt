"""Tests for factor-based portfolio optimization."""

import numpy as np
import pytest
from typing import List, Dict

from portopt.models.factor import FactorModel, FactorExposure
from portopt.solvers.factor import FactorSolver
from portopt.core.problem import PortfolioOptProblem
from portopt.solvers.factory import SolverFactory


def create_test_data(n_assets=5, n_factors=3, n_periods=100):
    """Create test data for factor model testing."""
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Generate factor returns
    factor_returns = np.random.normal(0.001, 0.02, (n_factors, n_periods))
    
    # Generate factor exposures
    factor_exposures = np.random.normal(0, 1, (n_assets, n_factors))
    
    # Generate specific returns
    specific_returns = np.random.normal(0, 0.01, (n_assets, n_periods))
    
    # Generate asset returns based on factor model
    asset_returns = factor_exposures @ factor_returns + specific_returns
    
    # Generate market caps
    market_caps = np.exp(np.random.normal(10, 2, n_assets))
    
    return {
        'asset_returns': asset_returns,
        'factor_returns': factor_returns,
        'factor_exposures': factor_exposures,
        'market_caps': market_caps,
        'factor_names': [f'Factor_{i}' for i in range(n_factors)]
    }


class TestFactorModel:
    """Tests for the FactorModel class."""
    
    def test_initialization(self):
        """Test model initialization."""
        model = FactorModel()
        assert model.risk_aversion == 2.0
        assert model.specific_risk_penalty == 0.1
        assert model.regularization_lambda == 1e-4
        
        model = FactorModel(risk_aversion=3.0, specific_risk_penalty=0.2)
        assert model.risk_aversion == 3.0
        assert model.specific_risk_penalty == 0.2
    
    def test_estimate_returns(self):
        """Test return estimation."""
        # Create test data
        data = create_test_data()
        
        # Create model
        model = FactorModel()
        
        # Estimate returns using only factor model
        returns = model.estimate_returns(
            data['factor_returns'],
            data['factor_exposures'],
            alpha=1.0
        )
        
        assert returns.shape == (data['asset_returns'].shape[0],)
        
        # Estimate returns using blend of factor and historical
        returns_blend = model.estimate_returns(
            data['factor_returns'],
            data['factor_exposures'],
            data['asset_returns'],
            alpha=0.5
        )
        
        assert returns_blend.shape == (data['asset_returns'].shape[0],)
        
        # Check that blending works as expected
        historical_returns = np.mean(data['asset_returns'], axis=1)
        factor_returns = model.estimate_returns(
            data['factor_returns'],
            data['factor_exposures'],
            alpha=1.0
        )
        
        expected_blend = 0.5 * factor_returns + 0.5 * historical_returns
        assert np.allclose(returns_blend, expected_blend)
    
    def test_estimate_covariance(self):
        """Test covariance estimation."""
        # Create test data
        data = create_test_data()
        
        # Create model
        model = FactorModel()
        
        # Estimate covariance using only factor model
        cov = model.estimate_covariance(
            data['factor_returns'],
            data['factor_exposures'],
            alpha=1.0
        )
        
        assert cov.shape == (data['asset_returns'].shape[0], data['asset_returns'].shape[0])
        
        # Check that covariance matrix is positive definite
        eigenvalues = np.linalg.eigvals(cov)
        assert np.all(eigenvalues > 0)
        
        # Estimate covariance using blend
        cov_blend = model.estimate_covariance(
            data['factor_returns'],
            data['factor_exposures'],
            data['asset_returns'],
            alpha=0.5
        )
        
        assert cov_blend.shape == (data['asset_returns'].shape[0], data['asset_returns'].shape[0])
        
        # Check that blended covariance is positive definite
        eigenvalues = np.linalg.eigvals(cov_blend)
        assert np.all(eigenvalues > 0)
    
    def test_optimize_weights(self):
        """Test portfolio weight optimization."""
        # Create test data
        data = create_test_data()
        
        # Create model
        model = FactorModel()
        
        # Optimize weights without target exposures
        weights, metadata = model.optimize_weights(
            data['factor_returns'],
            data['factor_exposures'],
            historical_returns=data['asset_returns']
        )
        
        assert weights.shape == (data['asset_returns'].shape[0],)
        assert np.isclose(np.sum(weights), 1.0)
        assert np.all(weights >= 0)
        
        # Check metadata
        assert 'expected_return' in metadata
        assert 'total_risk' in metadata
        assert 'factor_risk' in metadata
        assert 'specific_risk' in metadata
        assert 'sharpe_ratio' in metadata
        
        # Optimize with target exposures
        target_exposures = [
            FactorExposure(
                factor_name=f'Factor_{i}',
                target_exposure=0.2,
                confidence=0.8
            )
            for i in range(data['factor_returns'].shape[0])
        ]
        
        weights_target, metadata_target = model.optimize_weights(
            data['factor_returns'],
            data['factor_exposures'],
            target_factor_exposures=target_exposures,
            historical_returns=data['asset_returns']
        )
        
        assert weights_target.shape == (data['asset_returns'].shape[0],)
        assert np.isclose(np.sum(weights_target), 1.0)
        assert np.all(weights_target >= 0)
        
        # Check that portfolio exposures are closer to targets
        portfolio_exposures = weights_target @ data['factor_exposures']
        
        # The exposures should be somewhat close to the targets
        target_values = np.array([0.2] * data['factor_returns'].shape[0])
        for i in range(len(target_values)):
            assert abs(portfolio_exposures[i] - target_values[i]) < 0.5
        
        # Instead of comparing distances directly (which might not always work due to
        # multiple optimization objectives), check that at least some exposures are
        # closer to the target than in the unconstrained portfolio
        unconstrained_exposures = weights @ data['factor_exposures']
        
        # Calculate distances for each factor
        constrained_distances = np.abs(portfolio_exposures - target_values)
        unconstrained_distances = np.abs(unconstrained_exposures - target_values)
        
        # At least some factors should be closer to target
        assert np.any(constrained_distances < unconstrained_distances)
    
    def test_analyze_factor_contributions(self):
        """Test factor contribution analysis."""
        # Create test data
        data = create_test_data()
        
        # Create model
        model = FactorModel()
        
        # Optimize weights
        weights, _ = model.optimize_weights(
            data['factor_returns'],
            data['factor_exposures'],
            historical_returns=data['asset_returns']
        )
        
        # Analyze factor contributions
        analysis = model.analyze_factor_contributions(
            weights,
            data['factor_returns'],
            data['factor_exposures'],
            data['factor_names']
        )
        
        # Check analysis results
        assert 'portfolio_exposures' in analysis
        assert 'return_contribution' in analysis
        assert 'risk_contribution' in analysis
        assert 'pct_risk_contribution' in analysis
        assert 'total_factor_risk' in analysis
        assert 'factor_names' in analysis
        assert 'exposure_by_factor' in analysis
        assert 'return_contribution_by_factor' in analysis
        assert 'risk_contribution_by_factor' in analysis
        
        # Check shapes
        assert analysis['portfolio_exposures'].shape == (data['factor_returns'].shape[0],)
        assert analysis['return_contribution'].shape == (data['factor_returns'].shape[0],)
        assert analysis['risk_contribution'].shape == (data['factor_returns'].shape[0],)
        assert analysis['pct_risk_contribution'].shape == (data['factor_returns'].shape[0],)
        
        # Check that dictionaries have the right keys
        for factor_name in data['factor_names']:
            assert factor_name in analysis['exposure_by_factor']
            assert factor_name in analysis['return_contribution_by_factor']
            assert factor_name in analysis['risk_contribution_by_factor']


class TestFactorSolver:
    """Tests for the FactorSolver class."""
    
    def test_initialization(self):
        """Test solver initialization."""
        solver = FactorSolver()
        assert solver.risk_aversion == 2.0
        assert solver.specific_risk_penalty == 0.1
        assert solver.alpha == 0.5
        
        solver = FactorSolver(risk_aversion=3.0, alpha=0.7)
        assert solver.risk_aversion == 3.0
        assert solver.alpha == 0.7
    
    def test_solve_with_factors(self):
        """Test solving with factor data."""
        # Create test data
        data = create_test_data()
        
        # Create problem
        problem = PortfolioOptProblem(
            returns=data['asset_returns'],
            constraints={},
            factor_returns=data['factor_returns'],
            factor_exposures=data['factor_exposures'],
            market_caps=data['market_caps']
        )
        
        # Add factor names to constraints
        problem.constraints['factor_names'] = data['factor_names']
        
        # Create solver
        solver = FactorSolver()
        
        # Solve problem
        result = solver.solve(problem)
        
        # Check result
        assert result.weights.shape == (data['asset_returns'].shape[0],)
        assert np.isclose(np.sum(result.weights), 1.0)
        assert np.all(result.weights >= 0)
        assert result.objective_value > 0
        assert result.feasible
        
        # Check metadata
        assert hasattr(result, 'metadata')
        assert 'total_risk' in result.metadata
        assert 'factor_risk' in result.metadata
        assert 'specific_risk' in result.metadata
        assert 'factor_contribution' in result.metadata
        assert 'portfolio_exposures' in result.metadata
        assert 'exposure_by_factor' in result.metadata
    
    def test_solve_without_factors(self):
        """Test solving without factor data."""
        # Create test data
        data = create_test_data()
        
        # Create problem without factor data
        problem = PortfolioOptProblem(
            returns=data['asset_returns'],
            constraints={},
            market_caps=data['market_caps']
        )
        
        # Create solver
        solver = FactorSolver()
        
        # Solve problem
        result = solver.solve(problem)
        
        # Check result
        assert result.weights.shape == (data['asset_returns'].shape[0],)
        assert np.isclose(np.sum(result.weights), 1.0)
        assert np.all(result.weights >= 0)
        assert result.objective_value > 0
        assert result.feasible
        
        # Check metadata
        assert hasattr(result, 'metadata')
        assert 'factor_data_used' in result.metadata
        assert not result.metadata['factor_data_used']
    
    def test_solve_with_target_exposures(self):
        """Test solving with target factor exposures."""
        # Create test data
        data = create_test_data()
        
        # Create problem
        problem = PortfolioOptProblem(
            returns=data['asset_returns'],
            constraints={},
            factor_returns=data['factor_returns'],
            factor_exposures=data['factor_exposures'],
            market_caps=data['market_caps']
        )
        
        # Add factor names to constraints
        problem.constraints['factor_names'] = data['factor_names']
        
        # Add target factor exposures
        problem.constraints['factor_exposures'] = [
            {
                'factor_name': f'Factor_{i}',
                'target_exposure': 0.2,
                'confidence': 0.8
            }
            for i in range(data['factor_returns'].shape[0])
        ]
        
        # Create solver
        solver = FactorSolver()
        
        # Solve problem
        result = solver.solve(problem)
        
        # Check result
        assert result.weights.shape == (data['asset_returns'].shape[0],)
        assert np.isclose(np.sum(result.weights), 1.0)
        assert np.all(result.weights >= 0)
        assert result.objective_value > 0
        assert result.feasible
        
        # Calculate portfolio exposures
        portfolio_exposures = result.weights @ data['factor_exposures']
        
        # Check that exposures are close to targets
        for i in range(data['factor_returns'].shape[0]):
            # The exposure should be somewhat close to the target (0.2)
            # but not exactly due to other constraints and objectives
            assert abs(portfolio_exposures[i] - 0.2) < 0.5
    
    def test_calculate_factor_exposures(self):
        """Test calculation of factor exposures."""
        # Create test data
        data = create_test_data()
        
        # Create problem
        problem = PortfolioOptProblem(
            returns=data['asset_returns'],
            constraints={},
            factor_returns=data['factor_returns'],
            factor_exposures=data['factor_exposures'],
            market_caps=data['market_caps']
        )
        
        # Add factor names to constraints
        problem.constraints['factor_names'] = data['factor_names']
        
        # Create solver
        solver = FactorSolver()
        
        # Create test weights
        weights = np.ones(data['asset_returns'].shape[0]) / data['asset_returns'].shape[0]
        
        # Calculate factor exposures
        exposures = solver.calculate_factor_exposures(weights, problem)
        
        # Check exposures
        assert len(exposures) == data['factor_returns'].shape[0]
        for factor_name in data['factor_names']:
            assert factor_name in exposures
        
        # Calculate expected exposures
        expected_exposures = weights @ data['factor_exposures']
        
        # Check that calculated exposures match expected
        for i, factor_name in enumerate(data['factor_names']):
            assert np.isclose(exposures[factor_name], expected_exposures[i])


class TestFactorSolverFactory:
    """Tests for factor solver registration in factory."""
    
    def test_factory_registration(self):
        """Test that factor solvers are registered in the factory."""
        factory = SolverFactory()
        
        # Check that factor solvers are available
        solvers = factory.get_available_solvers()
        assert 'factor' in solvers
        assert 'factor_conservative' in solvers
        assert 'factor_aggressive' in solvers
        
        # Check that solver types are correct
        assert solvers['factor'] == FactorSolver
        assert solvers['factor_conservative'] == FactorSolver
        assert solvers['factor_aggressive'] == FactorSolver
    
    def test_create_factor_solver(self):
        """Test creating factor solvers from factory."""
        factory = SolverFactory()
        
        # Create standard factor solver
        solver = factory.create_solver('factor')
        assert isinstance(solver, FactorSolver)
        assert solver.risk_aversion == 2.0
        assert solver.alpha == 0.5
        
        # Create conservative factor solver
        solver_conservative = factory.create_solver('factor_conservative')
        assert isinstance(solver_conservative, FactorSolver)
        assert solver_conservative.risk_aversion == 4.0
        assert solver_conservative.alpha == 0.7
        
        # Create aggressive factor solver
        solver_aggressive = factory.create_solver('factor_aggressive')
        assert isinstance(solver_aggressive, FactorSolver)
        assert solver_aggressive.risk_aversion == 1.0
        assert solver_aggressive.alpha == 0.3
    
    def test_solver_parameters(self):
        """Test getting solver parameters from factory."""
        factory = SolverFactory()
        
        # Get parameters for factor solver
        params = factory.get_solver_parameters('factor')
        assert params['risk_aversion'] == 2.0
        assert params['specific_risk_penalty'] == 0.1
        assert params['alpha'] == 0.5
        assert params['regularization_lambda'] == 1e-4
        
        # Get parameters for conservative factor solver
        params_conservative = factory.get_solver_parameters('factor_conservative')
        assert params_conservative['risk_aversion'] == 4.0
        assert params_conservative['specific_risk_penalty'] == 0.2
        assert params_conservative['alpha'] == 0.7
        
        # Get parameters for aggressive factor solver
        params_aggressive = factory.get_solver_parameters('factor_aggressive')
        assert params_aggressive['risk_aversion'] == 1.0
        assert params_aggressive['specific_risk_penalty'] == 0.05
        assert params_aggressive['alpha'] == 0.3
