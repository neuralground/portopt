"""Approximate solvers for portfolio optimization."""

import time
import numpy as np
from typing import List, Dict, Any, Tuple, Optional

from portopt.core.problem import PortfolioOptProblem
from portopt.core.result import PortfolioOptResult
from portopt.solvers.base import BaseSolver
from portopt.solvers.constraint_adapter import ConstraintAdapter


class GeneticSolver(BaseSolver):
    """Genetic algorithm solver for portfolio optimization.
    
    This solver uses a genetic algorithm approach to find approximate solutions
    to portfolio optimization problems. It's particularly useful for problems
    with non-convex constraints like cardinality constraints.
    
    Attributes:
        population_size: Size of the population in the genetic algorithm
        generations: Number of generations to evolve
        mutation_rate: Probability of mutation for each gene
        crossover_rate: Probability of crossover between parents
        selection_pressure: Controls the selection pressure (higher = more elitist)
        tournament_size: Size of tournament for selection
    """
    
    def __init__(self, **kwargs):
        """Initialize the genetic algorithm solver.
        
        Args:
            population_size: Size of the population (default: 100)
            generations: Number of generations to evolve (default: 50)
            mutation_rate: Probability of mutation (default: 0.1)
            crossover_rate: Probability of crossover (default: 0.8)
            selection_pressure: Selection pressure parameter (default: 2.0)
            tournament_size: Size of tournament for selection (default: 3)
        """
        super().__init__(**kwargs)
        self.population_size = kwargs.get('population_size', 100)
        self.generations = kwargs.get('generations', 50)
        self.mutation_rate = kwargs.get('mutation_rate', 0.1)
        self.crossover_rate = kwargs.get('crossover_rate', 0.8)
        self.selection_pressure = kwargs.get('selection_pressure', 2.0)
        self.tournament_size = kwargs.get('tournament_size', 3)
    
    def solve(self, problem: PortfolioOptProblem) -> PortfolioOptResult:
        """Solve the portfolio optimization problem using a genetic algorithm.
        
        Args:
            problem: The portfolio optimization problem to solve
            
        Returns:
            PortfolioOptResult containing the optimized weights
        """
        start_time = time.time()
        
        # Preprocess the problem
        processed_problem = self.preprocess_problem(problem)
        
        # Get penalty functions for constraints
        penalties = ConstraintAdapter.to_penalty_functions(processed_problem)
        
        # Get bounds for the weights
        bounds = ConstraintAdapter.create_bounds(processed_problem)
        
        # Initialize population
        population = self._initialize_population(processed_problem, bounds)
        
        # Evaluate initial population
        fitness_values = self._evaluate_population(population, processed_problem, penalties)
        
        # Main evolution loop
        best_individual = None
        best_fitness = float('-inf')
        
        for generation in range(self.generations):
            # Select parents
            parents = self._selection(population, fitness_values)
            
            # Create new population through crossover and mutation
            new_population = []
            
            for i in range(0, len(parents), 2):
                if i + 1 < len(parents):
                    parent1, parent2 = parents[i], parents[i + 1]
                    
                    if np.random.random() < self.crossover_rate:
                        child1, child2 = self._crossover(parent1, parent2)
                    else:
                        child1, child2 = parent1.copy(), parent2.copy()
                    
                    child1 = self._mutation(child1, bounds)
                    child2 = self._mutation(child2, bounds)
                    
                    new_population.extend([child1, child2])
            
            # Ensure population size remains constant
            if len(new_population) < self.population_size:
                # Add some of the best individuals from previous generation
                sorted_indices = np.argsort(fitness_values)[::-1]
                for i in range(self.population_size - len(new_population)):
                    new_population.append(population[sorted_indices[i]].copy())
            
            # Update population
            population = new_population
            
            # Evaluate new population
            fitness_values = self._evaluate_population(population, processed_problem, penalties)
            
            # Track best individual
            max_idx = np.argmax(fitness_values)
            if fitness_values[max_idx] > best_fitness:
                best_fitness = fitness_values[max_idx]
                best_individual = population[max_idx].copy()
        
        # If no valid solution found, use the best invalid one
        if best_individual is None:
            best_individual = population[np.argmax(fitness_values)]
        
        # Normalize weights to ensure sum to one
        best_individual = best_individual / np.sum(best_individual)
        
        # Process weights to handle minimum weight threshold and cardinality
        min_weight = processed_problem.constraints.get('min_weight', 0.0)
        best_individual = self.process_weights(best_individual, min_weight, processed_problem)
        
        # Check if solution is feasible
        constraint_results = self.check_constraints(best_individual, processed_problem)
        feasible = all(constraint_results.values())
        
        # Calculate objective value
        objective_value = self.calculate_objective(best_individual, processed_problem)
        
        # Create result
        solve_time = time.time() - start_time
        result = PortfolioOptResult(
            weights=best_individual,
            objective_value=objective_value,
            solve_time=solve_time,
            feasible=feasible,
            iterations_used=self.generations
        )
        
        return result
    
    def _initialize_population(self, problem: PortfolioOptProblem, bounds: List[Tuple[float, float]]) -> List[np.ndarray]:
        """Initialize a random population of portfolios.
        
        Args:
            problem: The portfolio optimization problem
            bounds: List of (min, max) bounds for each asset
            
        Returns:
            List of weight vectors
        """
        population = []
        n_assets = problem.n_assets
        
        # Check for cardinality constraint
        cardinality = problem.constraints.get('cardinality')
        
        for _ in range(self.population_size):
            if cardinality is not None and cardinality < n_assets:
                # For cardinality constraint, randomly select assets
                weights = np.zeros(n_assets)
                selected_assets = np.random.choice(n_assets, cardinality, replace=False)
                
                # Assign random weights to selected assets
                for asset in selected_assets:
                    lower, upper = bounds[asset]
                    weights[asset] = np.random.uniform(lower, upper)
                
                # Normalize to sum to one
                if np.sum(weights) > 0:
                    weights = weights / np.sum(weights)
            else:
                # Generate random weights within bounds
                weights = np.array([np.random.uniform(lower, upper) for lower, upper in bounds])
                
                # Normalize to sum to one
                weights = weights / np.sum(weights)
            
            population.append(weights)
        
        return population
    
    def _evaluate_population(self, population: List[np.ndarray], problem: PortfolioOptProblem, 
                             penalties: List[Tuple[Any, float]]) -> np.ndarray:
        """Evaluate the fitness of each individual in the population.
        
        Args:
            population: List of weight vectors
            problem: The portfolio optimization problem
            penalties: List of (penalty_function, weight) tuples
            
        Returns:
            Array of fitness values
        """
        fitness_values = np.zeros(len(population))
        
        for i, weights in enumerate(population):
            # Calculate objective value (negative for minimization problems)
            objective = -self.calculate_objective(weights, problem)
            
            # Apply penalties
            penalty_sum = 0.0
            for penalty_func, penalty_weight in penalties:
                penalty_sum += penalty_weight * penalty_func(weights)
            
            # Final fitness (higher is better)
            fitness_values[i] = objective - penalty_sum
        
        return fitness_values
    
    def _selection(self, population: List[np.ndarray], fitness_values: np.ndarray) -> List[np.ndarray]:
        """Select parents for reproduction using tournament selection.
        
        Args:
            population: List of weight vectors
            fitness_values: Array of fitness values
            
        Returns:
            List of selected parents
        """
        selected = []
        
        for _ in range(self.population_size):
            # Tournament selection
            tournament_indices = np.random.choice(len(population), self.tournament_size, replace=False)
            tournament_fitness = fitness_values[tournament_indices]
            winner_idx = tournament_indices[np.argmax(tournament_fitness)]
            selected.append(population[winner_idx].copy())
        
        return selected
    
    def _crossover(self, parent1: np.ndarray, parent2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Perform crossover between two parents.
        
        Args:
            parent1: First parent weight vector
            parent2: Second parent weight vector
            
        Returns:
            Two child weight vectors
        """
        n_assets = len(parent1)
        crossover_point = np.random.randint(1, n_assets)
        
        # Create children by swapping segments
        child1 = np.concatenate([parent1[:crossover_point], parent2[crossover_point:]])
        child2 = np.concatenate([parent2[:crossover_point], parent1[crossover_point:]])
        
        # Normalize to sum to one
        child1 = child1 / np.sum(child1)
        child2 = child2 / np.sum(child2)
        
        return child1, child2
    
    def _mutation(self, individual: np.ndarray, bounds: List[Tuple[float, float]]) -> np.ndarray:
        """Apply mutation to an individual.
        
        Args:
            individual: Weight vector to mutate
            bounds: List of (min, max) bounds for each asset
            
        Returns:
            Mutated weight vector
        """
        mutated = individual.copy()
        n_assets = len(individual)
        
        for i in range(n_assets):
            if np.random.random() < self.mutation_rate:
                # Apply random perturbation
                lower, upper = bounds[i]
                mutated[i] = np.random.uniform(lower, upper)
        
        # Normalize to sum to one
        mutated = mutated / np.sum(mutated)
        
        return mutated
    
    def process_weights(self, weights: np.ndarray, min_weight: float = 0.0, problem: Optional[PortfolioOptProblem] = None) -> np.ndarray:
        """Process weights to ensure they satisfy basic constraints including cardinality.
        
        This method extends the base class implementation to handle cardinality constraints:
        1. Clips weights to be non-negative
        2. If cardinality constraint exists, keeps only the top N assets by weight
        3. Sets weights below threshold to zero
        4. Normalizes weights to sum to 1
        
        Args:
            weights: Raw weights from the optimizer
            min_weight: Minimum weight threshold
            problem: Portfolio optimization problem (optional)
            
        Returns:
            Processed weights
        """
        # Ensure non-negative weights
        weights = np.maximum(weights, 0.0)
        
        # Handle cardinality constraint if problem is provided
        if problem is not None and 'cardinality' in problem.constraints:
            cardinality = problem.constraints['cardinality']
            if cardinality < problem.n_assets:
                # Get indices of top N weights
                top_indices = np.argsort(weights)[-cardinality:]
                
                # Create a mask for top weights
                mask = np.zeros_like(weights, dtype=bool)
                mask[top_indices] = True
                
                # Zero out weights not in top N
                weights = weights * mask
        
        # Set small weights to zero
        weights[weights < min_weight] = 0.0
        
        # Normalize to sum to 1
        if np.sum(weights) > 0:
            weights = weights / np.sum(weights)
            
        return weights


class SimulatedAnnealingSolver(BaseSolver):
    """Simulated annealing solver for portfolio optimization.
    
    This solver uses simulated annealing to find approximate solutions to
    portfolio optimization problems. It's effective for problems with complex
    constraints and non-convex objective functions.
    """
    
    def __init__(self, **kwargs):
        """Initialize the simulated annealing solver.
        
        Args:
            initial_temp: Initial temperature (default: 100.0)
            cooling_rate: Cooling rate per iteration (default: 0.95)
            iterations: Number of iterations (default: 1000)
            perturbation_size: Size of random perturbations (default: 0.1)
        """
        super().__init__(**kwargs)
        self.initial_temp = kwargs.get('initial_temp', 100.0)
        self.cooling_rate = kwargs.get('cooling_rate', 0.95)
        self.iterations = kwargs.get('iterations', 1000)
        self.perturbation_size = kwargs.get('perturbation_size', 0.1)
    
    def solve(self, problem: PortfolioOptProblem) -> PortfolioOptResult:
        """Solve the portfolio optimization problem using simulated annealing.
        
        Args:
            problem: The portfolio optimization problem to solve
            
        Returns:
            PortfolioOptResult containing the optimized weights
        """
        start_time = time.time()
        
        # Preprocess the problem
        processed_problem = self.preprocess_problem(problem)
        
        # Get penalty functions for constraints
        penalties = ConstraintAdapter.to_penalty_functions(processed_problem)
        
        # Get bounds for the weights
        bounds = ConstraintAdapter.create_bounds(processed_problem)
        
        # Initialize with equal weights
        current_solution = np.ones(processed_problem.n_assets) / processed_problem.n_assets
        current_energy = self._calculate_energy(current_solution, processed_problem, penalties)
        
        best_solution = current_solution.copy()
        best_energy = current_energy
        
        # Main annealing loop
        temperature = self.initial_temp
        
        for iteration in range(self.iterations):
            # Generate neighbor solution
            neighbor = self._generate_neighbor(current_solution, bounds)
            
            # Calculate energy of neighbor
            neighbor_energy = self._calculate_energy(neighbor, processed_problem, penalties)
            
            # Decide whether to accept the neighbor
            energy_diff = neighbor_energy - current_energy
            
            if energy_diff < 0 or np.random.random() < np.exp(-energy_diff / temperature):
                current_solution = neighbor
                current_energy = neighbor_energy
                
                # Update best solution if needed
                if current_energy < best_energy:
                    best_solution = current_solution.copy()
                    best_energy = current_energy
            
            # Cool down
            temperature *= self.cooling_rate
        
        # Process weights to handle minimum weight threshold
        min_weight = processed_problem.constraints.get('min_weight', 0.0)
        best_solution = self.process_weights(best_solution, min_weight)
        
        # Check if solution is feasible
        constraint_results = self.check_constraints(best_solution, processed_problem)
        feasible = all(constraint_results.values())
        
        # Calculate objective value
        objective_value = self.calculate_objective(best_solution, processed_problem)
        
        # Create result
        solve_time = time.time() - start_time
        result = PortfolioOptResult(
            weights=best_solution,
            objective_value=objective_value,
            solve_time=solve_time,
            feasible=feasible,
            iterations_used=self.iterations
        )
        
        return result
    
    def _calculate_energy(self, weights: np.ndarray, problem: PortfolioOptProblem, 
                         penalties: List[Tuple[Any, float]]) -> float:
        """Calculate the energy (cost) of a solution.
        
        Args:
            weights: Weight vector
            problem: The portfolio optimization problem
            penalties: List of (penalty_function, weight) tuples
            
        Returns:
            Energy value (lower is better)
        """
        # Calculate objective value
        objective = self.calculate_objective(weights, problem)
        
        # Apply penalties
        penalty_sum = 0.0
        for penalty_func, penalty_weight in penalties:
            penalty_sum += penalty_weight * penalty_func(weights)
        
        # Final energy (lower is better)
        return objective + penalty_sum
    
    def _generate_neighbor(self, solution: np.ndarray, bounds: List[Tuple[float, float]]) -> np.ndarray:
        """Generate a neighboring solution by perturbing the current solution.
        
        Args:
            solution: Current weight vector
            bounds: List of (min, max) bounds for each asset
            
        Returns:
            Perturbed weight vector
        """
        neighbor = solution.copy()
        n_assets = len(solution)
        
        # Randomly select two assets to perturb
        i, j = np.random.choice(n_assets, 2, replace=False)
        
        # Calculate perturbation amount
        perturbation = np.random.uniform(0, self.perturbation_size)
        
        # Apply perturbation while maintaining sum-to-one constraint
        if neighbor[i] - perturbation >= bounds[i][0] and neighbor[j] + perturbation <= bounds[j][1]:
            neighbor[i] -= perturbation
            neighbor[j] += perturbation
        
        return neighbor
