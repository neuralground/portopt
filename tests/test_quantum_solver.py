"""Tests for the quantum solvers."""

import pytest
import numpy as np

from portopt.core.problem import PortfolioOptProblem
from portopt.solvers.quantum import QAOASolver, VQESolver


class TestQuantumSolvers:
    """Tests for quantum solvers."""
    
    @pytest.fixture
    def sample_problem(self) -> PortfolioOptProblem:
        """Create a sample portfolio optimization problem."""
        n_assets = 5  # Small problem for quantum solvers
        n_periods = 50
        
        # Generate random returns
        np.random.seed(42)
        returns = np.random.normal(0.001, 0.02, (n_assets, n_periods))
        
        # Create problem with basic constraints
        return PortfolioOptProblem(
            returns=returns,
            constraints={
                'min_weight': 0.01,
                'max_weight': 0.5,
                'sum_to_one': True
            }
        )
    
    @pytest.fixture
    def large_problem(self) -> PortfolioOptProblem:
        """Create a larger portfolio optimization problem to test hybrid approach."""
        n_assets = 12  # Large enough to trigger hybrid approach
        n_periods = 50
        
        # Generate random returns
        np.random.seed(42)
        returns = np.random.normal(0.001, 0.02, (n_assets, n_periods))
        
        # Create problem with basic constraints
        return PortfolioOptProblem(
            returns=returns,
            constraints={
                'min_weight': 0.01,
                'max_weight': 0.5,
                'sum_to_one': True
            }
        )
    
    def test_qaoa_solver_init(self):
        """Test that the QAOA solver initializes correctly."""
        solver = QAOASolver()
        assert solver.depth == 1
        assert solver.shots == 1024
        assert solver.optimizer_name == 'COBYLA'
        assert solver.max_iterations == 100
        
        # Test with custom parameters
        solver = QAOASolver(
            depth=2,
            shots=2048,
            backend_name='aer_simulator',
            optimizer_name='COBYLA',
            max_iterations=200
        )
        assert solver.depth == 2
        assert solver.shots == 2048
        assert solver.optimizer_name == 'COBYLA'
        assert solver.max_iterations == 200
    
    def test_vqe_solver_init(self):
        """Test that the VQE solver initializes correctly."""
        solver = VQESolver()
        assert solver.ansatz_type == 'RealAmplitudes'
        assert solver.shots == 1024
        assert solver.optimizer_name == 'COBYLA'
        assert solver.max_iterations == 100
        
        # Test with custom parameters
        solver = VQESolver(
            ansatz_type='TwoLocal',
            depth=2,
            shots=2048,
            backend_name='aer_simulator',
            optimizer_name='COBYLA',
            max_iterations=200
        )
        assert solver.ansatz_type == 'TwoLocal'
        assert solver.shots == 2048
        assert solver.optimizer_name == 'COBYLA'
        assert solver.max_iterations == 200
    
    def test_qaoa_solver_implementation(self, sample_problem):
        """Test that the QAOA solver Qiskit implementation runs."""
        solver = QAOASolver()
        result = solver.solve(sample_problem)
        
        # Check that weights sum to 1
        assert np.isclose(np.sum(result.weights), 1.0)
        
        # Check that weights are within bounds
        assert np.all(result.weights >= 0.0)
        assert np.all(result.weights <= 1.0)
        
        # Check that solve time is recorded
        assert result.solve_time > 0
        
        # Check that the objective value is reasonable
        assert result.objective_value is not None
    
    def test_vqe_solver_implementation(self, sample_problem):
        """Test that the VQE solver Qiskit implementation runs."""
        solver = VQESolver()
        result = solver.solve(sample_problem)
        
        # Check that weights sum to 1
        assert np.isclose(np.sum(result.weights), 1.0)
        
        # Check that weights are within bounds
        assert np.all(result.weights >= 0.0)
        assert np.all(result.weights <= 1.0)
        
        # Check that solve time is recorded
        assert result.solve_time > 0
        
        # Check that the objective value is reasonable
        assert result.objective_value is not None
    
    def test_qaoa_hybrid_approach(self, large_problem):
        """Test the QAOA hybrid approach for larger problems."""
        solver = QAOASolver(max_assets_per_subproblem=5)
        result = solver.solve(large_problem)
        
        # Check that weights sum to 1
        assert np.isclose(np.sum(result.weights), 1.0)
        
        # Check that weights are within bounds
        assert np.all(result.weights >= 0.0)
        assert np.all(result.weights <= 1.0)
        
        # Check that solve time is recorded
        assert result.solve_time > 0
        
        # Check that the objective value is reasonable
        assert result.objective_value is not None
    
    def test_vqe_hybrid_approach(self, large_problem):
        """Test the VQE hybrid approach for larger problems."""
        solver = VQESolver(max_assets_per_subproblem=5)
        result = solver.solve(large_problem)
        
        # Check that weights sum to 1
        assert np.isclose(np.sum(result.weights), 1.0)
        
        # Check that weights are within bounds
        assert np.all(result.weights >= 0.0)
        assert np.all(result.weights <= 1.0)
        
        # Check that solve time is recorded
        assert result.solve_time > 0
        
        # Check that the objective value is reasonable
        assert result.objective_value is not None
    
    def test_vqe_different_ansatz(self, sample_problem):
        """Test VQE with different ansatz types."""
        # Test with TwoLocal ansatz
        solver = VQESolver(ansatz_type='TwoLocal')
        result = solver.solve(sample_problem)
        
        # Check that weights sum to 1
        assert np.isclose(np.sum(result.weights), 1.0)
        
        # Check that the objective value is reasonable
        assert result.objective_value is not None
    
    def test_qaoa_different_depth(self, sample_problem):
        """Test QAOA with different circuit depths."""
        # Test with depth=2
        solver = QAOASolver(depth=2)
        result = solver.solve(sample_problem)
        
        # Check that weights sum to 1
        assert np.isclose(np.sum(result.weights), 1.0)
        
        # Check that the objective value is reasonable
        assert result.objective_value is not None
    
    def test_quantum_solver_with_constraints(self):
        """Test quantum solvers with various constraints."""
        n_assets = 5
        n_periods = 50
        
        np.random.seed(42)
        returns = np.random.normal(0.001, 0.02, (n_assets, n_periods))
        
        # Create problem with cardinality constraint
        problem = PortfolioOptProblem(
            returns=returns,
            constraints={
                'min_weight': 0.05,
                'max_weight': 0.5,
                'sum_to_one': True,
                'cardinality': 3  # Only 3 assets can have non-zero weights
            }
        )
        
        # Test with QAOA
        solver = QAOASolver()
        result = solver.solve(problem)
        
        # Check cardinality constraint
        non_zero_weights = np.sum(result.weights > problem.constraints['min_weight'])
        assert non_zero_weights <= problem.constraints['cardinality']
        
        # Test with VQE
        solver = VQESolver()
        result = solver.solve(problem)
        
        # Check cardinality constraint
        non_zero_weights = np.sum(result.weights > problem.constraints['min_weight'])
        assert non_zero_weights <= problem.constraints['cardinality']
    
    def test_quantum_solver_preprocess(self, sample_problem):
        """Test the preprocessing step of quantum solvers."""
        solver = QAOASolver()
        processed_problem = solver.preprocess_problem(sample_problem)
        
        # Check that the processed problem has the expected constraints
        assert 'min_weight' in processed_problem.constraints
        assert 'max_weight' in processed_problem.constraints
        assert 'sum_to_one' in processed_problem.constraints
        
        # Check that the returns and covariance matrix are preserved
        assert processed_problem.returns.shape == sample_problem.returns.shape
        assert processed_problem.cov_matrix.shape == sample_problem.cov_matrix.shape
