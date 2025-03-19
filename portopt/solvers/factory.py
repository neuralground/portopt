"""Factory module for creating different solver types."""

from typing import Dict, Any, Type

from portopt.solvers.base import BaseSolver
from portopt.solvers.classical import ClassicalSolver
from portopt.solvers.approximate import GeneticSolver, SimulatedAnnealingSolver
from portopt.solvers.quantum import QAOASolver, VQESolver
from portopt.solvers.advanced_genetic import AdvancedGeneticSolver
from portopt.solvers.black_litterman import BlackLittermanSolver
from portopt.solvers.factor import FactorSolver


class SolverFactory:
    """Factory for creating and configuring different solver types.
    
    This class provides a unified interface for creating different types of
    portfolio optimization solvers with appropriate default configurations.
    """
    
    def __init__(self):
        """Initialize the solver factory with default solver registrations."""
        self._solvers = {}
        self._default_params = {}
        
        # Register default solvers
        self.register_solver('classical', ClassicalSolver, {
            'max_iterations': 5,
            'initial_penalty': 100.0,
            'penalty_multiplier': 10.0,
            'perturbation_size': 0.05
        })
        
        # Register approximate solvers
        self.register_solver('genetic', GeneticSolver, {
            'population_size': 100,
            'generations': 50,
            'mutation_rate': 0.1,
            'crossover_rate': 0.8
        })
        
        self.register_solver('advanced_genetic', AdvancedGeneticSolver, {
            'population_size': 200,
            'generations': 100,
            'mutation_rate': 0.1,
            'crossover_rate': 0.8,
            'num_islands': 3,
            'adaptive_rates': True,
            'diversity_preservation': True
        })
        
        # Register advanced genetic variants
        self.register_solver('advanced_genetic_multi', AdvancedGeneticSolver, {
            'population_size': 200,
            'generations': 100,
            'mutation_rate': 0.1,
            'crossover_rate': 0.8,
            'num_islands': 3,
            'adaptive_rates': True,
            'diversity_preservation': True,
            'multi_objective': True,
            'risk_weight': 0.5,
            'return_weight': 0.5
        })
        
        self.register_solver('annealing', SimulatedAnnealingSolver, {
            'initial_temp': 100.0,
            'cooling_rate': 0.95,
            'iterations': 1000,
            'perturbation_size': 0.1
        })
        
        # Register Black-Litterman solver
        self.register_solver('black_litterman', BlackLittermanSolver, {
            'risk_aversion': 2.5,
            'tau': 0.05,
            'use_market_caps': True,
            'default_view_confidence': 0.5
        })
        
        # Register Black-Litterman variant with higher risk aversion
        self.register_solver('black_litterman_conservative', BlackLittermanSolver, {
            'risk_aversion': 5.0,
            'tau': 0.05,
            'use_market_caps': True,
            'default_view_confidence': 0.5
        })
        
        # Register factor-based solvers
        self.register_solver('factor', FactorSolver, {
            'risk_aversion': 2.0,
            'specific_risk_penalty': 0.1,
            'alpha': 0.5,
            'regularization_lambda': 1e-4
        })
        
        # Register factor-based solver variant with higher risk aversion
        self.register_solver('factor_conservative', FactorSolver, {
            'risk_aversion': 4.0,
            'specific_risk_penalty': 0.2,
            'alpha': 0.7,  # More weight on factor model vs historical data
            'regularization_lambda': 1e-4
        })
        
        # Register factor-based solver variant with lower risk aversion
        self.register_solver('factor_aggressive', FactorSolver, {
            'risk_aversion': 1.0,
            'specific_risk_penalty': 0.05,
            'alpha': 0.3,  # More weight on historical data vs factor model
            'regularization_lambda': 1e-4
        })
        
        # Register quantum solvers
        self.register_solver(
            'qaoa',
            QAOASolver,
            {
                'depth': 1,
                'shots': 1024,
                'backend_name': 'aer_simulator',
                'optimizer_name': 'COBYLA',
                'max_iterations': 100,
                'max_assets_per_subproblem': 5
            }
        )
        
        self.register_solver(
            'qaoa_deep',
            QAOASolver,
            {
                'depth': 3,
                'shots': 1024,
                'backend_name': 'aer_simulator',
                'optimizer_name': 'COBYLA',
                'max_iterations': 100,
                'max_assets_per_subproblem': 5
            }
        )
        
        self.register_solver(
            'vqe',
            VQESolver,
            {
                'ansatz_name': 'RealAmplitudes',
                'depth': 2,
                'shots': 1024,
                'backend_name': 'aer_simulator',
                'optimizer_name': 'COBYLA',
                'max_iterations': 100,
                'max_assets_per_subproblem': 5
            }
        )
        
        self.register_solver(
            'vqe_twolocal',
            VQESolver,
            {
                'ansatz_name': 'TwoLocal',
                'depth': 2,
                'shots': 1024,
                'backend_name': 'aer_simulator',
                'optimizer_name': 'COBYLA',
                'max_iterations': 100,
                'max_assets_per_subproblem': 5
            }
        )
    
    def register_solver(self, solver_type: str, solver_class: Type[BaseSolver],
                        default_params: Dict[str, Any] = None):
        """Register a new solver type.
        
        Args:
            solver_type: Identifier for the solver type
            solver_class: Class implementing the solver
            default_params: Default parameters for the solver
        """
        self._solvers[solver_type] = solver_class
        self._default_params[solver_type] = default_params or {}
    
    def create_solver(self, solver_type: str, **kwargs) -> BaseSolver:
        """Create a solver of the specified type.
        
        Args:
            solver_type: Type of solver to create
            **kwargs: Parameters to override defaults
            
        Returns:
            Configured solver instance
            
        Raises:
            ValueError: If solver_type is not registered
        """
        if solver_type not in self._solvers:
            raise ValueError(f"Unknown solver type: {solver_type}")
        
        # Combine default parameters with provided overrides
        params = self._default_params[solver_type].copy()
        params.update(kwargs)
        
        # Create and return the solver
        return self._solvers[solver_type](**params)
    
    def get_available_solvers(self) -> Dict[str, Type[BaseSolver]]:
        """Get dictionary of available solver types.
        
        Returns:
            Dictionary mapping solver type names to solver classes
        """
        return self._solvers.copy()
    
    def get_solver_parameters(self, solver_type: str) -> Dict[str, Any]:
        """Get default parameters for a solver type.
        
        Args:
            solver_type: Type of solver
            
        Returns:
            Dictionary of default parameters
            
        Raises:
            ValueError: If solver_type is not registered
        """
        if solver_type not in self._default_params:
            raise ValueError(f"Unknown solver type: {solver_type}")
        
        return self._default_params[solver_type].copy()
