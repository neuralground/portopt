"""Tests for the Black-Litterman model."""

import numpy as np
import pytest
from typing import List, Dict, Any

from portopt.core.problem import PortfolioOptProblem
from portopt.solvers.factory import SolverFactory
from portopt.solvers.black_litterman import BlackLittermanSolver
from portopt.models.black_litterman import BlackLittermanModel, InvestorView


def create_test_problem(n_assets: int = 5, seed: int = 42) -> PortfolioOptProblem:
    """Create a test portfolio optimization problem.
    
    Args:
        n_assets: Number of assets in the portfolio
        seed: Random seed for reproducibility
        
    Returns:
        PortfolioOptProblem instance
    """
    # Set random seed for reproducibility
    np.random.seed(seed)
    
    # Generate random returns for multiple periods (n_assets x n_periods)
    n_periods = 100  # Use 100 periods for realistic covariance estimation
    returns = np.random.uniform(0.05, 0.15, (n_assets, n_periods))
    
    # Generate random market caps (for equilibrium weights)
    market_caps = np.random.uniform(1e9, 1e11, n_assets)
    
    # Create asset names
    asset_names = [f"Asset_{i+1}" for i in range(n_assets)]
    
    # Create problem instance
    problem = PortfolioOptProblem(
        returns=returns,
        market_caps=market_caps,
        constraints={
            'asset_names': asset_names,
            'min_weight': 0.0,  # Allow zero weights
            'max_weight': 1.0,  # No single asset can be more than 100%
        }
    )
    
    return problem


def create_test_views(n_assets: int = 5) -> List[InvestorView]:
    """Create test investor views.
    
    Args:
        n_assets: Number of assets in the portfolio
        
    Returns:
        List of InvestorView instances
    """
    views = [
        # Absolute view: Asset_1 will return 20%
        InvestorView(
            assets=[0],
            weights=[1.0],
            value=0.20,
            confidence=0.8,
            is_relative=False
        ),
        # Relative view: Asset_2 will outperform Asset_3 by 5%
        InvestorView(
            assets=[1, 2],
            weights=[1.0, -1.0],
            value=0.05,
            confidence=0.6,
            is_relative=True
        )
    ]
    
    return views


class TestBlackLittermanModel:
    """Tests for the BlackLittermanModel class."""
    
    def test_initialization(self):
        """Test initialization of the BlackLittermanModel."""
        model = BlackLittermanModel()
        assert model.risk_aversion == 2.5
        assert model.tau == 0.05
        assert model.use_market_caps == True
        assert model.default_view_confidence == 0.5
        
        # Test custom parameters
        model = BlackLittermanModel(
            risk_aversion=3.0,
            tau=0.1,
            use_market_caps=False,
            default_view_confidence=0.7
        )
        assert model.risk_aversion == 3.0
        assert model.tau == 0.1
        assert model.use_market_caps == False
        assert model.default_view_confidence == 0.7
    
    def test_calculate_equilibrium_returns(self):
        """Test calculation of equilibrium returns."""
        model = BlackLittermanModel()
        
        # Create a simple covariance matrix
        cov_matrix = np.array([
            [0.04, 0.01, 0.02],
            [0.01, 0.09, 0.03],
            [0.02, 0.03, 0.16]
        ])
        
        # Test with equal weights
        equal_weights = np.array([1/3, 1/3, 1/3])
        eq_returns = model.calculate_equilibrium_returns(cov_matrix)
        
        # Verify shape
        assert eq_returns.shape == (3,)
        
        # Verify calculation (π = λΣw)
        expected_returns = model.risk_aversion * cov_matrix @ equal_weights
        np.testing.assert_array_almost_equal(eq_returns, expected_returns)
        
        # Test with custom weights
        custom_weights = np.array([0.5, 0.3, 0.2])
        eq_returns = model.calculate_equilibrium_returns(cov_matrix, custom_weights)
        
        # Verify calculation
        expected_returns = model.risk_aversion * cov_matrix @ custom_weights
        np.testing.assert_array_almost_equal(eq_returns, expected_returns)
    
    def test_create_view_matrices(self):
        """Test creation of view matrices."""
        model = BlackLittermanModel()
        
        # Create views
        views = [
            InvestorView(
                assets=[0],
                weights=[1.0],
                value=0.10,
                confidence=0.8,
                is_relative=False
            ),
            InvestorView(
                assets=[1, 2],
                weights=[1.0, -1.0],
                value=0.05,
                confidence=0.6,
                is_relative=True
            )
        ]
        
        # Create view matrices
        P, Q, Omega = model._create_view_matrices(views, n_assets=3)
        
        # Verify shapes
        assert P.shape == (2, 3)
        assert Q.shape == (2,)
        assert Omega.shape == (2, 2)
        
        # Verify P matrix (picking matrix)
        expected_P = np.array([
            [1.0, 0.0, 0.0],  # First view: Asset 0
            [0.0, 1.0, -1.0]  # Second view: Asset 1 - Asset 2
        ])
        np.testing.assert_array_almost_equal(P, expected_P)
        
        # Verify Q matrix (view values)
        expected_Q = np.array([0.10, 0.05])
        np.testing.assert_array_almost_equal(Q, expected_Q)
        
        # Verify Omega is diagonal
        assert np.count_nonzero(Omega - np.diag(np.diagonal(Omega))) == 0
        
        # Verify confidence affects Omega (higher confidence = lower uncertainty)
        assert Omega[0, 0] < Omega[1, 1]  # First view has higher confidence
        
        # Verify minimum value for Omega diagonal elements
        assert np.all(np.diag(Omega) >= 1e-6)
        
        # Test with extreme confidence (0.9999)
        views[0].confidence = 0.9999
        _, _, Omega = model._create_view_matrices(views, n_assets=3)
        
        # Verify Omega still has non-zero values
        assert np.all(np.diag(Omega) >= 1e-6)
    
    def test_blend_returns(self):
        """Test blending of market equilibrium with investor views."""
        model = BlackLittermanModel()
        
        # Create a simple covariance matrix
        cov_matrix = np.array([
            [0.04, 0.01, 0.02],
            [0.01, 0.09, 0.03],
            [0.02, 0.03, 0.16]
        ])
        
        # Create views
        views = [
            InvestorView(
                assets=[0],
                weights=[1.0],
                value=0.10,
                confidence=0.8,
                is_relative=False
            )
        ]
        
        # Blend returns
        posterior_returns, posterior_cov = model.blend_returns(cov_matrix, views)
        
        # Verify shapes
        assert posterior_returns.shape == (3,)
        assert posterior_cov.shape == (3, 3)
        
        # Verify posterior returns are different from prior
        prior_returns = model.calculate_equilibrium_returns(cov_matrix)
        assert not np.array_equal(posterior_returns, prior_returns)
        
        # Verify posterior returns are influenced by the view
        # The first asset's return should be pulled toward the view value
        assert abs(posterior_returns[0] - 0.10) < abs(prior_returns[0] - 0.10)
        
        # Test with extreme confidence (0.9999)
        views[0].confidence = 0.9999
        posterior_returns, posterior_cov = model.blend_returns(cov_matrix, views)
        
        # Verify function still works with high confidence
        assert posterior_returns.shape == (3,)
        assert posterior_cov.shape == (3, 3)
    
    def test_optimize_portfolio(self):
        """Test portfolio optimization using the Black-Litterman model."""
        model = BlackLittermanModel()
        
        # Create a simple covariance matrix
        cov_matrix = np.array([
            [0.04, 0.01, 0.02],
            [0.01, 0.09, 0.03],
            [0.02, 0.03, 0.16]
        ])
        
        # Create views
        views = [
            InvestorView(
                assets=[0],
                weights=[1.0],
                value=0.20,  # Very high return for first asset
                confidence=0.9,  # High confidence but not 1.0 to avoid singular matrix
                is_relative=False
            )
        ]
        
        # Optimize portfolio
        weights = model.optimize_portfolio(cov_matrix, views)
        
        # Verify shape
        assert weights.shape == (3,)
        
        # Verify weights sum to 1
        assert np.isclose(np.sum(weights), 1.0)
        
        # Verify weights are non-negative
        assert np.all(weights >= 0)
        
        # Verify first asset has higher weight due to the view
        assert weights[0] > weights[1] and weights[0] > weights[2]
        
        # Test with extreme confidence (0.9999)
        views[0].confidence = 0.9999
        weights = model.optimize_portfolio(cov_matrix, views)
        
        # Verify function still works with high confidence
        assert weights.shape == (3,)
        assert np.isclose(np.sum(weights), 1.0)


class TestBlackLittermanSolver:
    """Tests for the BlackLittermanSolver class."""
    
    def test_initialization(self):
        """Test initialization of the BlackLittermanSolver."""
        solver = BlackLittermanSolver()
        assert solver.risk_aversion == 2.5
        assert solver.tau == 0.05
        assert solver.use_market_caps == True
        assert solver.default_view_confidence == 0.5
        
        # Test custom parameters
        solver = BlackLittermanSolver(
            risk_aversion=3.0,
            tau=0.1,
            use_market_caps=False,
            default_view_confidence=0.7
        )
        assert solver.risk_aversion == 3.0
        assert solver.tau == 0.1
        assert solver.use_market_caps == False
        assert solver.default_view_confidence == 0.7
    
    def test_extract_views_from_problem(self):
        """Test extraction of views from problem constraints."""
        solver = BlackLittermanSolver()
        
        # Create a problem with views
        problem = create_test_problem()
        
        # Add views to problem
        views = create_test_views()
        view_dicts = []
        for view in views:
            view_dict = {
                'assets': view.assets,
                'weights': view.weights,
                'value': view.value,
                'confidence': view.confidence,
                'is_relative': view.is_relative
            }
            view_dicts.append(view_dict)
        
        problem.constraints['views'] = view_dicts
        
        # Extract views
        extracted_views = solver._extract_views_from_problem(problem)
        
        # Verify number of views
        assert len(extracted_views) == len(views)
        
        # Verify view properties
        for i, view in enumerate(extracted_views):
            assert view.assets == views[i].assets
            assert view.weights == views[i].weights
            assert view.value == views[i].value
            assert view.confidence == views[i].confidence
            assert view.is_relative == views[i].is_relative
    
    def test_get_market_weights(self):
        """Test getting market weights from problem."""
        solver = BlackLittermanSolver()
        
        # Create a problem with market caps
        problem = create_test_problem()
        
        # Get market weights
        market_weights = solver._get_market_weights(problem)
        
        # Verify shape
        assert market_weights.shape == (problem.n_assets,)
        
        # Verify weights sum to 1
        assert np.isclose(np.sum(market_weights), 1.0)
        
        # Verify weights are proportional to market caps
        expected_weights = problem.market_caps / np.sum(problem.market_caps)
        np.testing.assert_array_almost_equal(market_weights, expected_weights)
        
        # Test with use_market_caps=False
        solver.use_market_caps = False
        market_weights = solver._get_market_weights(problem)
        
        # Verify equal weights
        expected_weights = np.ones(problem.n_assets) / problem.n_assets
        np.testing.assert_array_almost_equal(market_weights, expected_weights)
    
    def test_solve(self):
        """Test solving a portfolio optimization problem."""
        solver = BlackLittermanSolver()
        
        # Create a problem
        problem = create_test_problem()
        
        # Add views to problem
        views = create_test_views()
        view_dicts = []
        for view in views:
            view_dict = {
                'assets': view.assets,
                'weights': view.weights,
                'value': view.value,
                'confidence': view.confidence,
                'is_relative': view.is_relative
            }
            view_dicts.append(view_dict)
        
        problem.constraints['views'] = view_dicts
        
        # Solve the problem
        result = solver.solve(problem)
        
        # Verify result properties
        assert hasattr(result, 'weights')
        assert hasattr(result, 'objective_value')
        assert hasattr(result, 'solve_time')
        assert hasattr(result, 'feasible')
        
        # Verify weights shape
        assert result.weights.shape == (problem.n_assets,)
        
        # Verify weights sum to 1
        assert np.isclose(np.sum(result.weights), 1.0)
        
        # Verify weights are non-negative
        assert np.all(result.weights >= 0)
        
        # Verify objective value is portfolio variance
        expected_variance = result.weights.T @ problem.cov_matrix @ result.weights
        assert np.isclose(result.objective_value, expected_variance)
        
        # Verify solution is feasible
        assert result.feasible == True
        
        # Test with high confidence views
        for view_dict in problem.constraints['views']:
            view_dict['confidence'] = 0.9999
            
        # Solve again with high confidence views
        result = solver.solve(problem)
        
        # Verify solution is still valid
        assert result.weights.shape == (problem.n_assets,)
        assert np.isclose(np.sum(result.weights), 1.0)
    
    def test_calculate_weight_uncertainty(self):
        """Test calculation of weight uncertainty."""
        solver = BlackLittermanSolver()
        
        # Create a problem
        problem = create_test_problem()
        
        # Add views to problem
        views = create_test_views()
        view_dicts = []
        for view in views:
            view_dict = {
                'assets': view.assets,
                'weights': view.weights,
                'value': view.value,
                'confidence': view.confidence,
                'is_relative': view.is_relative
            }
            view_dicts.append(view_dict)
        
        problem.constraints['views'] = view_dicts
        
        # Calculate weight uncertainty
        uncertainty = solver.calculate_weight_uncertainty(problem, n_samples=100)
        
        # Verify result properties
        assert 'mean_weights' in uncertainty
        assert 'std_weights' in uncertainty
        
        # Verify shapes
        assert uncertainty['mean_weights'].shape == (problem.n_assets,)
        assert uncertainty['std_weights'].shape == (problem.n_assets,)
        
        # Verify mean weights sum to approximately 1
        assert np.isclose(np.sum(uncertainty['mean_weights']), 1.0, atol=1e-2)
        
        # Verify standard deviations are non-negative
        assert np.all(uncertainty['std_weights'] >= 0)
        
        # Test with high confidence views
        for view_dict in problem.constraints['views']:
            view_dict['confidence'] = 0.9999
            
        # Calculate again with high confidence views
        uncertainty = solver.calculate_weight_uncertainty(problem, n_samples=100)
        
        # Verify result is still valid
        assert 'mean_weights' in uncertainty
        assert 'std_weights' in uncertainty


class TestSolverFactory:
    """Tests for the SolverFactory with Black-Litterman solver."""
    
    def test_factory_registration(self):
        """Test that the Black-Litterman solver is registered in the factory."""
        factory = SolverFactory()
        
        # Get available solvers
        solvers = factory.get_available_solvers()
        
        # Verify Black-Litterman solver is registered
        assert 'black_litterman' in solvers
        assert 'black_litterman_conservative' in solvers
    
    def test_create_solver(self):
        """Test creating a Black-Litterman solver from the factory."""
        factory = SolverFactory()
        
        # Create a Black-Litterman solver
        solver = factory.create_solver('black_litterman')
        
        # Verify solver type
        assert isinstance(solver, BlackLittermanSolver)
        
        # Verify default parameters
        assert solver.risk_aversion == 2.5
        assert solver.tau == 0.05
        assert solver.use_market_caps == True
        assert solver.default_view_confidence == 0.5
        
        # Create a conservative Black-Litterman solver
        solver = factory.create_solver('black_litterman_conservative')
        
        # Verify solver type
        assert isinstance(solver, BlackLittermanSolver)
        
        # Verify conservative parameters
        assert solver.risk_aversion == 5.0
        
        # Create a Black-Litterman solver with custom parameters
        solver = factory.create_solver('black_litterman', risk_aversion=3.0, tau=0.1)
        
        # Verify custom parameters
        assert solver.risk_aversion == 3.0
        assert solver.tau == 0.1
    
    def test_solver_integration(self):
        """Test integration of Black-Litterman solver with the framework."""
        factory = SolverFactory()
        
        # Create a problem
        problem = create_test_problem()
        
        # Add views to problem
        views = create_test_views()
        view_dicts = []
        for view in views:
            view_dict = {
                'assets': view.assets,
                'weights': view.weights,
                'value': view.value,
                'confidence': view.confidence,
                'is_relative': view.is_relative
            }
            view_dicts.append(view_dict)
        
        problem.constraints['views'] = view_dicts
        
        # Create and solve with Black-Litterman solver
        bl_solver = factory.create_solver('black_litterman')
        bl_result = bl_solver.solve(problem)
        
        # Create and solve with classical solver for comparison
        classical_solver = factory.create_solver('classical')
        classical_result = classical_solver.solve(problem)
        
        # Verify both solvers produce valid results
        assert np.isclose(np.sum(bl_result.weights), 1.0)
        assert np.isclose(np.sum(classical_result.weights), 1.0)
        
        # Verify both solutions are feasible
        assert bl_result.feasible == True
        assert classical_result.feasible == True
        
        # Verify solutions are different (Black-Litterman should be influenced by views)
        assert not np.array_equal(bl_result.weights, classical_result.weights)
