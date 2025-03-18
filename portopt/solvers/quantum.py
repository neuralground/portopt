"""Quantum solvers for portfolio optimization.

This module implements quantum computing approaches to portfolio optimization
using Qiskit. It provides QAOA and VQE implementations that can be used to
solve portfolio optimization problems of varying sizes through full quantum
or hybrid quantum-classical approaches.

References:
    [1] Farhi, E., Goldstone, J., & Gutmann, S. (2014). A quantum approximate 
        optimization algorithm. arXiv preprint arXiv:1411.4028.
    [2] Peruzzo, A., McClean, J., et al. (2014). A variational eigenvalue solver
        on a photonic quantum processor. Nature communications, 5(1), 4213.
    [3] Herman, D., Googin, C., Liu, X., Galda, A., & Safro, I. (2022). 
        A survey of quantum computing for finance. ACM Computing Surveys, 55(9), 1-37.
"""

from typing import Dict, List, Optional, Tuple, Union, Any
import numpy as np
import time

from qiskit_aer import AerSimulator
# Use Qiskit Aer's Sampler and Estimator
from qiskit_aer.primitives import Sampler, Estimator
from qiskit_algorithms import QAOA, VQE, NumPyMinimumEigensolver
from qiskit_algorithms import SamplingVQE
from qiskit_algorithms.optimizers import COBYLA, SPSA, SLSQP
from qiskit.circuit.library import QAOAAnsatz
from qiskit.circuit.library import RealAmplitudes, TwoLocal
from qiskit_optimization.algorithms import MinimumEigenOptimizer
from qiskit_optimization.converters import QuadraticProgramToQubo
from qiskit_optimization.problems import QuadraticProgram

from portopt.core.problem import PortfolioOptProblem
from portopt.core.result import PortfolioOptResult
from portopt.solvers.base import BaseSolver
from portopt.solvers.constraint_adapter import ConstraintAdapter

class QuantumSolver(BaseSolver):
    """Base class for quantum solvers.
    
    This class provides common functionality for quantum solvers, including
    problem preprocessing and result formatting. Specific quantum algorithms
    should inherit from this class and implement the _run_quantum_algorithm method.
    
    The QuantumSolver converts portfolio optimization problems into a format suitable
    for quantum computing, typically Quadratic Unconstrained Binary Optimization (QUBO).
    It handles the setup of quantum backends, optimizers, and manages the execution
    of quantum algorithms.
    
    For larger problems (those with more assets than max_assets_per_subproblem),
    the solver automatically switches to a hybrid approach that divides the problem
    into smaller subproblems.
    
    Attributes:
        shots: Number of measurement shots for the quantum circuit execution
        backend_name: Name of the quantum backend to use (e.g., 'aer_simulator')
        optimizer_name: Name of the classical optimizer to use (e.g., 'COBYLA', 'SPSA')
        max_iterations: Maximum number of iterations for the classical optimizer
        max_assets_per_subproblem: Maximum number of assets per subproblem
            for hybrid approaches
        backend: Instantiated quantum backend object
        sampler: Quantum sampler primitive
        estimator: Quantum estimator primitive
        optimizer: Classical optimizer instance
        
    References:
        [1] Egger, D. J., et al. (2020). Quantum computing for Finance: state of the art
            and future prospects. IEEE Transactions on Quantum Engineering.
        [2] Brandhofer, N., et al. (2022). Quantum algorithms for portfolio optimization.
            Journal of Finance and Data Science, 8, 71-83.
    """
    
    def __init__(
        self,
        shots: int = 1024,
        backend_name: str = 'aer_simulator',
        optimizer_name: str = 'COBYLA',
        max_iterations: int = 100,
        max_assets_per_subproblem: int = 5,
        **kwargs
    ):
        """Initialize a quantum solver.
        
        Args:
            shots: Number of shots for the quantum algorithm
            backend_name: Name of the backend to use
            optimizer_name: Name of the optimizer to use
            max_iterations: Maximum number of iterations for the optimizer
            max_assets_per_subproblem: Maximum number of assets per subproblem
                for hybrid approaches
            **kwargs: Additional keyword arguments
        """
        super().__init__(**kwargs)
        self.shots = shots
        self.max_iterations = max_iterations
        self.max_assets_per_subproblem = max_assets_per_subproblem
        self.backend_name = backend_name
        self.optimizer_name = optimizer_name
        
        # Initialize backend
        if backend_name == 'aer_simulator':
            self.backend = AerSimulator()
        else:
            raise ValueError(f"Unsupported backend: {backend_name}")
        
        # Initialize sampler and estimator
        self.sampler = Sampler()
        self.estimator = Estimator()
        
        # Initialize optimizer
        if optimizer_name == 'COBYLA':
            self.optimizer = COBYLA(maxiter=max_iterations)
        elif optimizer_name == 'SPSA':
            self.optimizer = SPSA(maxiter=max_iterations)
        elif optimizer_name == 'SLSQP':
            self.optimizer = SLSQP(maxiter=max_iterations)
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_name}")
    
    def solve(self, problem: PortfolioOptProblem) -> PortfolioOptResult:
        """Solve the portfolio optimization problem using a quantum algorithm.
        
        Args:
            problem: The portfolio optimization problem
            
        Returns:
            PortfolioOptResult containing the optimized weights
        """
        start_time = time.time()
        
        # Preprocess the problem
        processed_problem = self.preprocess_problem(problem)
        
        # Run the quantum algorithm (to be implemented by subclasses)
        weights, iterations = self._run_quantum_algorithm(processed_problem)
        
        # Process weights to handle minimum weight threshold
        min_weight = processed_problem.constraints.get('min_weight', 0.0)
        weights = self.process_weights(weights, min_weight)
        
        # Check if solution is feasible
        constraint_results = self.check_constraints(weights, processed_problem)
        feasible = all(constraint_results.values())
        
        # Calculate objective value
        objective_value = self.calculate_objective(weights, processed_problem)
        
        # Create result
        solve_time = time.time() - start_time
        result = PortfolioOptResult(
            weights=weights,
            objective_value=objective_value,
            solve_time=solve_time,
            feasible=feasible,
            iterations_used=iterations
        )
        
        return result
    
    def preprocess_problem(self, problem: PortfolioOptProblem) -> PortfolioOptProblem:
        """Preprocess the problem for quantum optimization.
        
        This method applies any necessary transformations to the problem
        to make it suitable for quantum optimization.
        
        Args:
            problem: The original portfolio optimization problem
            
        Returns:
            The preprocessed problem
        """
        # Create a new instance of the problem to avoid modifying the original
        processed_problem = PortfolioOptProblem(
            returns=problem.returns.copy(),
            constraints=problem.constraints.copy(),
            target_return=problem.target_return,
            volumes=problem.volumes.copy() if problem.volumes is not None else None,
            spreads=problem.spreads.copy() if problem.spreads is not None else None,
            market_caps=problem.market_caps.copy() if problem.market_caps is not None else None,
            factor_returns=problem.factor_returns.copy() if problem.factor_returns is not None else None,
            factor_exposures=problem.factor_exposures.copy() if problem.factor_exposures is not None else None,
            classifications=problem.classifications,
            asset_classes=problem.asset_classes,
            currencies=problem.currencies,
            credit_profiles=problem.credit_profiles
        )
        
        # Apply any necessary transformations
        # (e.g., scaling, normalization, etc.)
        
        return processed_problem
    
    def _run_quantum_algorithm(self, problem: PortfolioOptProblem) -> Tuple[np.ndarray, int]:
        """Run the quantum algorithm to solve the problem.
        
        This method should be implemented by subclasses to run a specific
        quantum algorithm for portfolio optimization.
        
        Args:
            problem: The preprocessed portfolio optimization problem
            
        Returns:
            Tuple of (weights, iterations)
        """
        raise NotImplementedError("Subclasses must implement this method")
    
    def _create_quadratic_program(self, problem: PortfolioOptProblem) -> QuadraticProgram:
        """Create a QuadraticProgram from the portfolio optimization problem.
        
        Args:
            problem: The portfolio optimization problem
            
        Returns:
            QuadraticProgram representation of the problem
        """
        n_assets = problem.returns.shape[0]
        
        # Create a quadratic program
        qp = QuadraticProgram(name="Portfolio Optimization")
        
        # Add binary variables for each asset
        for i in range(n_assets):
            qp.binary_var(name=f"x{i}")
        
        # Set up the objective function (maximize expected return)
        expected_returns = np.mean(problem.returns, axis=1)
        linear = {f"x{i}": expected_returns[i] for i in range(n_assets)}
        
        # Calculate covariance matrix from returns
        covariance = np.cov(problem.returns)
        
        # Add risk term using the covariance matrix
        quadratic = {}
        for i in range(n_assets):
            for j in range(n_assets):
                if covariance[i, j] != 0:
                    quadratic[(f"x{i}", f"x{j}")] = -0.5 * covariance[i, j]
        
        # Set the objective to maximize return and minimize risk
        qp.maximize(linear=linear, quadratic=quadratic)
        
        # Add constraints
        constraints = problem.constraints
        
        # Sum to one constraint
        if constraints.get('sum_to_one', False):
            qp.linear_constraint(
                linear={f"x{i}": 1 for i in range(n_assets)},
                sense="==",
                rhs=1,
                name="sum_to_one"
            )
        
        # Min weight constraint
        min_weight = constraints.get('min_weight', 0.0)
        if min_weight > 0:
            for i in range(n_assets):
                qp.linear_constraint(
                    linear={f"x{i}": 1},
                    sense=">=",
                    rhs=min_weight,
                    name=f"min_weight_{i}"
                )
        
        # Max weight constraint
        max_weight = constraints.get('max_weight', 1.0)
        if max_weight < 1.0:
            for i in range(n_assets):
                qp.linear_constraint(
                    linear={f"x{i}": 1},
                    sense="<=",
                    rhs=max_weight,
                    name=f"max_weight_{i}"
                )
        
        return qp
    
    def process_weights(self, weights: np.ndarray, min_weight: float = 0.0) -> np.ndarray:
        """Process weights to ensure they satisfy constraints.
        
        Args:
            weights: Raw weights from the quantum algorithm
            min_weight: Minimum weight threshold
            
        Returns:
            Processed weights
        """
        # Apply minimum weight threshold
        if min_weight > 0:
            weights[weights < min_weight] = 0
            
            # Renormalize if sum is not zero
            if np.sum(weights) > 0:
                weights = weights / np.sum(weights)
        
        return weights


class QAOASolver(QuantumSolver):
    """Quantum Approximate Optimization Algorithm (QAOA) solver for portfolio optimization.
    
    This solver uses the QAOA algorithm to find the optimal portfolio allocation.
    QAOA is a hybrid quantum-classical algorithm designed for combinatorial
    optimization problems. It works by creating a parameterized quantum circuit
    with alternating problem and mixing Hamiltonians, optimizing the circuit
    parameters using classical optimization techniques, and sampling from the
    optimized circuit to obtain portfolio weights.
    
    For portfolio optimization, QAOA is particularly well-suited for problems
    with discrete constraints, such as cardinality constraints (limiting the
    number of assets in the portfolio).
    
    The solver's performance generally improves with increasing circuit depth,
    but deeper circuits take longer to run and are more susceptible to noise
    on real quantum hardware.
    
    Attributes:
        depth: Circuit depth for the QAOA algorithm (number of QAOA layers)
        shots: Number of measurement shots
        backend_name: Name of the quantum backend to use
        optimizer_name: Name of the classical optimizer to use
        max_iterations: Maximum number of iterations for the optimizer
        max_assets_per_subproblem: Maximum number of assets per subproblem
            for hybrid approaches
            
    References:
        [1] Farhi, E., Goldstone, J., & Gutmann, S. (2014). A quantum approximate 
            optimization algorithm. arXiv preprint arXiv:1411.4028.
        [2] Zhou, L., et al. (2020). Quantum Approximate Optimization Algorithm: 
            Performance, Mechanism, and Implementation on Near-Term Devices. 
            Physical Review X, 10(2), 021067.
    """
    
    def __init__(
        self,
        depth: int = 1,
        shots: int = 1024,
        backend_name: str = 'aer_simulator',
        optimizer_name: str = 'COBYLA',
        max_iterations: int = 100,
        max_assets_per_subproblem: int = 5,
        **kwargs
    ):
        """Initialize a QAOA solver.
        
        Args:
            depth: Circuit depth for the QAOA algorithm
            shots: Number of shots for the quantum algorithm
            backend_name: Name of the backend to use
            optimizer_name: Name of the optimizer to use
            max_iterations: Maximum number of iterations for the optimizer
            max_assets_per_subproblem: Maximum number of assets per subproblem
                for hybrid approaches
            **kwargs: Additional keyword arguments
        """
        super().__init__(
            shots=shots,
            backend_name=backend_name,
            optimizer_name=optimizer_name,
            max_iterations=max_iterations,
            max_assets_per_subproblem=max_assets_per_subproblem,
            **kwargs
        )
        self.depth = depth
    
    def _run_quantum_algorithm(self, problem: PortfolioOptProblem) -> Tuple[np.ndarray, int]:
        """Run the QAOA algorithm to solve the problem.
        
        Args:
            problem: The preprocessed portfolio optimization problem
            
        Returns:
            Tuple of (weights, iterations)
        """
        # For small problems, we can use full QAOA
        if problem.returns.shape[0] <= self.max_assets_per_subproblem:
            return self._run_full_qaoa(problem)
        else:
            # For larger problems, we need to use a hybrid approach or approximation
            return self._run_hybrid_qaoa(problem)
    
    def _run_full_qaoa(self, problem: PortfolioOptProblem) -> Tuple[np.ndarray, int]:
        """Run the full QAOA algorithm on the problem.
        
        Args:
            problem: The portfolio optimization problem
            
        Returns:
            Tuple of (weights, iterations)
        """
        # Create a quadratic program
        qp = self._create_quadratic_program(problem)
        
        # Convert to QUBO
        converter = QuadraticProgramToQubo()
        qubo = converter.convert(qp)
        
        # Create QAOA ansatz
        ansatz = QAOAAnsatz(
            qubo.to_ising()[0],
            reps=self.depth
        )
        
        # Create SamplingVQE with QAOA ansatz
        vqe = SamplingVQE(
            sampler=self.sampler,
            ansatz=ansatz,
            optimizer=self.optimizer
        )
        
        # Create optimizer
        optimizer = MinimumEigenOptimizer(vqe)
        
        # Solve the problem
        try:
            result = optimizer.solve(qubo)
            
            # Get the solution
            x = result.x
            
            # Convert binary solution to weights
            weights = np.array([x[i] for i in range(problem.returns.shape[0])])
            
            # Normalize weights to sum to 1
            if np.sum(weights) > 0:
                weights = weights / np.sum(weights)
            
            # Get iterations, default to 0 if None
            iterations = 0
            if hasattr(result, 'min_eigen_solver_result') and result.min_eigen_solver_result is not None:
                if hasattr(result.min_eigen_solver_result, 'optimizer_evals') and result.min_eigen_solver_result.optimizer_evals is not None:
                    iterations = result.min_eigen_solver_result.optimizer_evals
            
            return weights, iterations
        except Exception as e:
            raise ValueError(f"QAOA solver failed: {str(e)}")
    
    def _run_hybrid_qaoa(self, problem: PortfolioOptProblem) -> Tuple[np.ndarray, int]:
        """Run a hybrid QAOA approach for larger problems.
        
        For larger problems, we use a hybrid approach:
        1. Divide the problem into smaller subproblems
        2. Solve each subproblem with QAOA
        3. Combine the solutions
        
        Args:
            problem: The portfolio optimization problem
            
        Returns:
            Tuple of (weights, iterations)
        """
        # Get the number of assets and covariance matrix
        n_assets = problem.returns.shape[0]
        covariance = np.cov(problem.returns)
        
        # Determine the number of subproblems
        n_subproblems = (n_assets + self.max_assets_per_subproblem - 1) // self.max_assets_per_subproblem
        
        # Initialize weights
        weights = np.zeros(n_assets)
        total_iterations = 0
        
        # Solve each subproblem
        for i in range(n_subproblems):
            # Determine assets for this subproblem
            start_idx = i * self.max_assets_per_subproblem
            end_idx = min((i + 1) * self.max_assets_per_subproblem, n_assets)
            sub_assets = list(range(start_idx, end_idx))
            
            # Create subproblem
            sub_returns = problem.returns[sub_assets, :]
            sub_problem = PortfolioOptProblem(
                returns=sub_returns,
                constraints=problem.constraints.copy()
            )
            
            # Solve subproblem
            sub_weights, sub_iterations = self._run_full_qaoa(sub_problem)
            
            # Update weights and iterations
            weights[sub_assets] = sub_weights
            total_iterations += sub_iterations
        
        # Normalize weights
        if np.sum(weights) > 0:
            weights = weights / np.sum(weights)
        else:
            # If all weights are zero, use equal weights
            weights = np.ones(n_assets) / n_assets
        
        return weights, total_iterations


class VQESolver(QuantumSolver):
    """Variational Quantum Eigensolver (VQE) solver for portfolio optimization.
    
    This solver uses the VQE algorithm to find the optimal portfolio allocation.
    VQE is a hybrid quantum-classical algorithm that finds the ground state of
    a Hamiltonian. It works by creating a parameterized quantum circuit (ansatz),
    optimizing the circuit parameters to minimize the Hamiltonian's expectation
    value, and using the final state to determine the optimal portfolio weights.
    
    The solver supports different ansatz types:
    - 'RealAmplitudes': A hardware-efficient ansatz using only RY rotations and CX gates
    - 'TwoLocal': A more expressive ansatz with configurable entanglement patterns
    
    The depth parameter controls the expressivity of the ansatz, with deeper
    circuits potentially finding better solutions but requiring more quantum
    resources and being more susceptible to noise.
    
    Attributes:
        ansatz_type: Type of ansatz to use ('RealAmplitudes' or 'TwoLocal')
        depth: Circuit depth for the VQE algorithm
        shots: Number of measurement shots
        backend_name: Name of the quantum backend to use
        optimizer_name: Name of the classical optimizer to use
        max_iterations: Maximum number of iterations for the optimizer
        max_assets_per_subproblem: Maximum number of assets per subproblem
            for hybrid approaches
            
    References:
        [1] Peruzzo, A., McClean, J., et al. (2014). A variational eigenvalue solver
            on a photonic quantum processor. Nature communications, 5(1), 4213.
        [2] Kandala, A., et al. (2017). Hardware-efficient variational quantum
            eigensolver for small molecules and quantum magnets. Nature, 549(7671), 242-246.
        [3] LaRose, R., et al. (2019). Overview and Comparison of Gate Level
            Quantum Software Platforms. Quantum, 3, 130.
    """
    
    def __init__(
        self,
        ansatz_type: str = 'RealAmplitudes',
        depth: int = 1,
        shots: int = 1024,
        backend_name: str = 'aer_simulator',
        optimizer_name: str = 'COBYLA',
        max_iterations: int = 100,
        max_assets_per_subproblem: int = 5,
        **kwargs
    ):
        """Initialize a VQE solver.
        
        Args:
            ansatz_type: Type of ansatz to use (RealAmplitudes or TwoLocal)
            depth: Circuit depth for the VQE algorithm
            shots: Number of shots for the quantum algorithm
            backend_name: Name of the backend to use
            optimizer_name: Name of the optimizer to use
            max_iterations: Maximum number of iterations for the optimizer
            max_assets_per_subproblem: Maximum number of assets per subproblem
                for hybrid approaches
            **kwargs: Additional keyword arguments
        """
        super().__init__(
            shots=shots,
            backend_name=backend_name,
            optimizer_name=optimizer_name,
            max_iterations=max_iterations,
            max_assets_per_subproblem=max_assets_per_subproblem,
            **kwargs
        )
        self.ansatz_type = ansatz_type
        self.depth = depth
    
    def _run_quantum_algorithm(self, problem: PortfolioOptProblem) -> Tuple[np.ndarray, int]:
        """Run the VQE algorithm to solve the problem.
        
        Args:
            problem: The preprocessed portfolio optimization problem
            
        Returns:
            Tuple of (weights, iterations)
        """
        # For small problems, we can use full VQE
        if problem.returns.shape[0] <= self.max_assets_per_subproblem:
            return self._run_full_vqe(problem)
        else:
            # For larger problems, we need to use a hybrid approach or approximation
            return self._run_hybrid_vqe(problem)
    
    def _run_full_vqe(self, problem: PortfolioOptProblem) -> Tuple[np.ndarray, int]:
        """Run the full VQE algorithm on the problem.
        
        Args:
            problem: The portfolio optimization problem
            
        Returns:
            Tuple of (weights, iterations)
        """
        # Create a quadratic program
        qp = self._create_quadratic_program(problem)
        
        # Convert to QUBO
        converter = QuadraticProgramToQubo()
        qubo = converter.convert(qp)
        
        # Create ansatz
        if self.ansatz_type == 'RealAmplitudes':
            ansatz = RealAmplitudes(qubo.get_num_binary_vars(), reps=self.depth)
        elif self.ansatz_type == 'TwoLocal':
            ansatz = TwoLocal(qubo.get_num_binary_vars(), 'ry', 'cz', reps=self.depth)
        else:
            raise ValueError(f"Unsupported ansatz type: {self.ansatz_type}")
        
        # Create SamplingVQE instance
        vqe = SamplingVQE(
            sampler=self.sampler,
            ansatz=ansatz,
            optimizer=self.optimizer
        )
        
        # Create optimizer
        optimizer = MinimumEigenOptimizer(vqe)
        
        # Solve the problem
        try:
            result = optimizer.solve(qubo)
            
            # Get the solution
            x = result.x
            
            # Convert binary solution to weights
            weights = np.array([x[i] for i in range(problem.returns.shape[0])])
            
            # Normalize weights to sum to 1
            if np.sum(weights) > 0:
                weights = weights / np.sum(weights)
            
            # Get iterations, default to 0 if None
            iterations = 0
            if hasattr(result, 'min_eigen_solver_result') and result.min_eigen_solver_result is not None:
                if hasattr(result.min_eigen_solver_result, 'optimizer_evals') and result.min_eigen_solver_result.optimizer_evals is not None:
                    iterations = result.min_eigen_solver_result.optimizer_evals
            
            return weights, iterations
        except Exception as e:
            raise ValueError(f"VQE solver failed: {str(e)}")
    
    def _run_hybrid_vqe(self, problem: PortfolioOptProblem) -> Tuple[np.ndarray, int]:
        """Run a hybrid VQE approach for larger problems.
        
        For larger problems, we use a hybrid approach:
        1. Divide the problem into smaller subproblems
        2. Solve each subproblem with VQE
        3. Combine the solutions
        
        Args:
            problem: The portfolio optimization problem
            
        Returns:
            Tuple of (weights, iterations)
        """
        # Get the number of assets and covariance matrix
        n_assets = problem.returns.shape[0]
        covariance = np.cov(problem.returns)
        
        # Determine the number of subproblems
        n_subproblems = (n_assets + self.max_assets_per_subproblem - 1) // self.max_assets_per_subproblem
        
        # Initialize weights
        weights = np.zeros(n_assets)
        total_iterations = 0
        
        # Solve each subproblem
        for i in range(n_subproblems):
            # Determine assets for this subproblem
            start_idx = i * self.max_assets_per_subproblem
            end_idx = min((i + 1) * self.max_assets_per_subproblem, n_assets)
            sub_assets = list(range(start_idx, end_idx))
            
            # Create subproblem
            sub_returns = problem.returns[sub_assets, :]
            sub_problem = PortfolioOptProblem(
                returns=sub_returns,
                constraints=problem.constraints.copy()
            )
            
            # Solve subproblem
            sub_weights, sub_iterations = self._run_full_vqe(sub_problem)
            
            # Update weights and iterations
            weights[sub_assets] = sub_weights
            total_iterations += sub_iterations
        
        # Normalize weights
        if np.sum(weights) > 0:
            weights = weights / np.sum(weights)
        else:
            # If all weights are zero, use equal weights
            weights = np.ones(n_assets) / n_assets
        
        return weights, total_iterations
