"""Advanced Genetic Algorithm solver for portfolio optimization."""

import time
import numpy as np
from typing import List, Dict, Any, Tuple, Optional, Callable, Union

from portopt.core.problem import PortfolioOptProblem
from portopt.core.result import PortfolioOptResult
from portopt.solvers.base import BaseSolver
from portopt.solvers.constraint_adapter import ConstraintAdapter


class AdvancedGeneticSolver(BaseSolver):
    """Advanced Genetic Algorithm solver for portfolio optimization.
    
    This solver implements a sophisticated genetic algorithm approach with:
    - Multi-objective optimization capabilities
    - Adaptive mutation and crossover rates
    - Island model for better exploration
    - Elitism to preserve best solutions
    - Diverse crossover and mutation operators
    - Niching to maintain population diversity
    
    Attributes:
        population_size: Size of the population in the genetic algorithm
        generations: Number of generations to evolve
        mutation_rate: Initial probability of mutation for each gene
        crossover_rate: Initial probability of crossover between parents
        elitism_ratio: Proportion of best individuals to preserve unchanged
        num_islands: Number of sub-populations for island model
        migration_interval: Generations between migration events
        migration_rate: Proportion of individuals to migrate between islands
        tournament_size: Size of tournament for selection
        adaptive_rates: Whether to use adaptive mutation and crossover rates
        diversity_preservation: Whether to use diversity preservation techniques
        multi_objective: Whether to use multi-objective optimization
        risk_weight: Weight for risk objective in multi-objective optimization
        return_weight: Weight for return objective in multi-objective optimization
        early_stopping: Whether to use early stopping if no improvement
        early_stopping_generations: Generations without improvement before stopping
    """
    
    def __init__(self, **kwargs):
        """Initialize the advanced genetic algorithm solver.
        
        Args:
            population_size: Size of the population (default: 200)
            generations: Number of generations to evolve (default: 100)
            mutation_rate: Initial probability of mutation (default: 0.1)
            crossover_rate: Initial probability of crossover (default: 0.8)
            elitism_ratio: Proportion of best individuals to preserve (default: 0.1)
            num_islands: Number of sub-populations for island model (default: 3)
            migration_interval: Generations between migration events (default: 10)
            migration_rate: Proportion of individuals to migrate (default: 0.1)
            tournament_size: Size of tournament for selection (default: 3)
            adaptive_rates: Whether to use adaptive rates (default: True)
            diversity_preservation: Whether to preserve diversity (default: True)
            multi_objective: Whether to use multi-objective optimization (default: False)
            risk_weight: Weight for risk objective (default: 0.5)
            return_weight: Weight for return objective (default: 0.5)
            early_stopping: Whether to use early stopping (default: True)
            early_stopping_generations: Generations without improvement (default: 20)
        """
        super().__init__(**kwargs)
        self.population_size = kwargs.get('population_size', 200)
        self.generations = kwargs.get('generations', 100)
        self.mutation_rate = kwargs.get('mutation_rate', 0.1)
        self.crossover_rate = kwargs.get('crossover_rate', 0.8)
        self.elitism_ratio = kwargs.get('elitism_ratio', 0.1)
        self.num_islands = kwargs.get('num_islands', 3)
        self.migration_interval = kwargs.get('migration_interval', 10)
        self.migration_rate = kwargs.get('migration_rate', 0.1)
        self.tournament_size = kwargs.get('tournament_size', 3)
        self.adaptive_rates = kwargs.get('adaptive_rates', True)
        self.diversity_preservation = kwargs.get('diversity_preservation', True)
        self.multi_objective = kwargs.get('multi_objective', False)
        self.risk_weight = kwargs.get('risk_weight', 0.5)
        self.return_weight = kwargs.get('return_weight', 0.5)
        self.early_stopping = kwargs.get('early_stopping', True)
        self.early_stopping_generations = kwargs.get('early_stopping_generations', 20)
    
    def solve(self, problem: PortfolioOptProblem) -> PortfolioOptResult:
        """Solve the portfolio optimization problem using advanced genetic algorithm.
        
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
        
        # Initialize island populations
        islands = self._initialize_islands(processed_problem, bounds)
        
        # Track best solution across all islands
        best_individual = None
        best_fitness = float('-inf')
        generations_without_improvement = 0
        
        # Main evolution loop
        for generation in range(self.generations):
            # Evolve each island separately
            for island_idx in range(self.num_islands):
                # Evaluate island population
                if self.multi_objective:
                    risk_values, return_values = self._evaluate_population_multi_objective(
                        islands[island_idx], processed_problem
                    )
                    fitness_values = self._calculate_weighted_fitness(
                        risk_values, return_values, penalties, islands[island_idx]
                    )
                else:
                    fitness_values = self._evaluate_population(
                        islands[island_idx], processed_problem, penalties
                    )
                
                # Apply elitism: preserve best individuals
                elitism_count = int(self.elitism_ratio * self.population_size)
                elite_indices = np.argsort(fitness_values)[-elitism_count:]
                elite_individuals = [islands[island_idx][i].copy() for i in elite_indices]
                
                # Select parents
                parents = self._selection(islands[island_idx], fitness_values)
                
                # Create new population through crossover and mutation
                new_population = []
                
                # Add elite individuals directly
                new_population.extend(elite_individuals)
                
                # Fill the rest with offspring
                while len(new_population) < self.population_size:
                    # Select two parents
                    idx1, idx2 = np.random.choice(len(parents), 2, replace=False)
                    parent1, parent2 = parents[idx1], parents[idx2]
                    
                    # Apply adaptive rates if enabled
                    current_crossover_rate = self.crossover_rate
                    current_mutation_rate = self.mutation_rate
                    
                    if self.adaptive_rates:
                        # Adjust rates based on population diversity
                        diversity = self._calculate_diversity(islands[island_idx])
                        current_mutation_rate = self._adapt_mutation_rate(diversity)
                        current_crossover_rate = self._adapt_crossover_rate(diversity)
                    
                    # Crossover
                    if np.random.random() < current_crossover_rate:
                        child1, child2 = self._crossover(parent1, parent2)
                    else:
                        child1, child2 = parent1.copy(), parent2.copy()
                    
                    # Mutation
                    child1 = self._mutation(child1, bounds, current_mutation_rate)
                    child2 = self._mutation(child2, bounds, current_mutation_rate)
                    
                    # Add children to new population
                    new_population.append(child1)
                    if len(new_population) < self.population_size:
                        new_population.append(child2)
                
                # Update island population
                islands[island_idx] = new_population
            
            # Migration between islands at specified intervals
            if generation % self.migration_interval == 0 and generation > 0:
                islands = self._perform_migration(islands)
            
            # Evaluate all individuals across all islands
            all_individuals = [ind for island in islands for ind in island]
            if self.multi_objective:
                risk_values, return_values = self._evaluate_population_multi_objective(
                    all_individuals, processed_problem
                )
                all_fitness = self._calculate_weighted_fitness(
                    risk_values, return_values, penalties, all_individuals
                )
            else:
                all_fitness = self._evaluate_population(
                    all_individuals, processed_problem, penalties
                )
            
            # Update best individual
            max_idx = np.argmax(all_fitness)
            if all_fitness[max_idx] > best_fitness:
                best_fitness = all_fitness[max_idx]
                best_individual = all_individuals[max_idx].copy()
                generations_without_improvement = 0
            else:
                generations_without_improvement += 1
            
            # Early stopping if no improvement for several generations
            if self.early_stopping and generations_without_improvement >= self.early_stopping_generations:
                break
        
        # If no valid solution found, use the best invalid one
        if best_individual is None:
            all_individuals = [ind for island in islands for ind in island]
            if self.multi_objective:
                risk_values, return_values = self._evaluate_population_multi_objective(
                    all_individuals, processed_problem
                )
                all_fitness = self._calculate_weighted_fitness(
                    risk_values, return_values, penalties, all_individuals
                )
            else:
                all_fitness = self._evaluate_population(
                    all_individuals, processed_problem, penalties
                )
            best_individual = all_individuals[np.argmax(all_fitness)]
        
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
            iterations_used=generation + 1  # Actual number of generations used
        )
        
        return result
    
    def _initialize_islands(self, problem: PortfolioOptProblem, bounds: List[Tuple[float, float]]) -> List[List[np.ndarray]]:
        """Initialize multiple island populations.
        
        Args:
            problem: The portfolio optimization problem
            bounds: List of (min, max) bounds for each asset
            
        Returns:
            List of island populations, each containing weight vectors
        """
        islands = []
        
        # Create each island with a different initialization strategy
        for island_idx in range(self.num_islands):
            island_population = []
            n_assets = problem.n_assets
            
            # Different initialization strategies for different islands
            if island_idx == 0:
                # First island: standard random initialization
                island_population = self._initialize_random_population(problem, bounds)
            elif island_idx == 1:
                # Second island: equal weight initialization with perturbations
                for _ in range(self.population_size):
                    weights = np.ones(n_assets) / n_assets
                    # Add small random perturbations
                    weights += np.random.normal(0, 0.05, n_assets)
                    weights = np.maximum(weights, 0)  # Ensure non-negative
                    weights = weights / np.sum(weights)  # Normalize
                    island_population.append(weights)
            else:
                # Other islands: biased towards high return assets
                expected_returns = problem.exp_returns
                for _ in range(self.population_size):
                    weights = np.zeros(n_assets)
                    # Probability proportional to expected returns
                    probs = np.maximum(expected_returns, 0)
                    if np.sum(probs) > 0:
                        probs = probs / np.sum(probs)
                        weights = np.random.dirichlet(probs * 5 + 0.1)
                    else:
                        weights = np.random.dirichlet(np.ones(n_assets))
                    island_population.append(weights)
            
            islands.append(island_population)
        
        return islands
    
    def _initialize_random_population(self, problem: PortfolioOptProblem, bounds: List[Tuple[float, float]]) -> List[np.ndarray]:
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
                             penalties: List[Tuple[Callable, float]]) -> np.ndarray:
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
    
    def _evaluate_population_multi_objective(self, population: List[np.ndarray], 
                                            problem: PortfolioOptProblem) -> Tuple[np.ndarray, np.ndarray]:
        """Evaluate the population for multiple objectives (risk and return).
        
        Args:
            population: List of weight vectors
            problem: The portfolio optimization problem
            
        Returns:
            Tuple of (risk_values, return_values) arrays
        """
        risk_values = np.zeros(len(population))
        return_values = np.zeros(len(population))
        
        for i, weights in enumerate(population):
            # Calculate expected return
            expected_return = np.dot(weights, problem.exp_returns)
            return_values[i] = expected_return
            
            # Calculate risk (variance)
            risk = weights.T @ problem.cov_matrix @ weights
            risk_values[i] = risk
        
        return risk_values, return_values
    
    def _calculate_weighted_fitness(self, risk_values: np.ndarray, return_values: np.ndarray,
                                   penalties: List[Tuple[Callable, float]], 
                                   population: List[np.ndarray]) -> np.ndarray:
        """Calculate weighted fitness for multi-objective optimization.
        
        Args:
            risk_values: Array of risk values
            return_values: Array of return values
            penalties: List of (penalty_function, weight) tuples
            population: List of weight vectors
            
        Returns:
            Array of weighted fitness values
        """
        # Normalize objectives to [0, 1] range
        if len(risk_values) > 1:
            risk_min, risk_max = np.min(risk_values), np.max(risk_values)
            risk_range = risk_max - risk_min
            if risk_range > 0:
                normalized_risk = (risk_values - risk_min) / risk_range
            else:
                normalized_risk = np.zeros_like(risk_values)
            
            return_min, return_max = np.min(return_values), np.max(return_values)
            return_range = return_max - return_min
            if return_range > 0:
                normalized_return = (return_values - return_min) / return_range
            else:
                normalized_return = np.zeros_like(return_values)
        else:
            normalized_risk = np.zeros_like(risk_values)
            normalized_return = np.zeros_like(return_values)
        
        # Calculate weighted sum (minimize risk, maximize return)
        weighted_fitness = -self.risk_weight * normalized_risk + self.return_weight * normalized_return
        
        # Apply penalties
        for i, weights in enumerate(population):
            penalty_sum = 0.0
            for penalty_func, penalty_weight in penalties:
                penalty_sum += penalty_weight * penalty_func(weights)
            weighted_fitness[i] -= penalty_sum
        
        return weighted_fitness
    
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
        """Perform crossover between two parents using multiple strategies.
        
        Args:
            parent1: First parent weight vector
            parent2: Second parent weight vector
            
        Returns:
            Two child weight vectors
        """
        n_assets = len(parent1)
        
        # Choose crossover strategy randomly
        strategy = np.random.choice(['single_point', 'two_point', 'uniform', 'blend'])
        
        if strategy == 'single_point':
            # Single-point crossover
            crossover_point = np.random.randint(1, n_assets)
            child1 = np.concatenate([parent1[:crossover_point], parent2[crossover_point:]])
            child2 = np.concatenate([parent2[:crossover_point], parent1[crossover_point:]])
            
        elif strategy == 'two_point':
            # Two-point crossover
            points = sorted(np.random.choice(range(1, n_assets), 2, replace=False))
            child1 = np.concatenate([parent1[:points[0]], 
                                    parent2[points[0]:points[1]], 
                                    parent1[points[1]:]])
            child2 = np.concatenate([parent2[:points[0]], 
                                    parent1[points[0]:points[1]], 
                                    parent2[points[1]:]])
            
        elif strategy == 'uniform':
            # Uniform crossover
            mask = np.random.random(n_assets) > 0.5
            child1 = np.where(mask, parent1, parent2)
            child2 = np.where(mask, parent2, parent1)
            
        else:  # blend
            # Blend crossover (interpolation)
            alpha = np.random.random()
            child1 = alpha * parent1 + (1 - alpha) * parent2
            child2 = (1 - alpha) * parent1 + alpha * parent2
        
        # Normalize to sum to one
        child1 = child1 / np.sum(child1) if np.sum(child1) > 0 else child1
        child2 = child2 / np.sum(child2) if np.sum(child2) > 0 else child2
        
        return child1, child2
    
    def _mutation(self, individual: np.ndarray, bounds: List[Tuple[float, float]], 
                 mutation_rate: float) -> np.ndarray:
        """Apply mutation to an individual using multiple strategies.
        
        Args:
            individual: Weight vector to mutate
            bounds: List of (min, max) bounds for each asset
            mutation_rate: Current mutation rate
            
        Returns:
            Mutated weight vector
        """
        mutated = individual.copy()
        n_assets = len(individual)
        
        # Choose mutation strategy randomly
        strategy = np.random.choice(['random_reset', 'swap', 'gaussian'])
        
        if strategy == 'random_reset':
            # Random reset mutation
            for i in range(n_assets):
                if np.random.random() < mutation_rate:
                    lower, upper = bounds[i]
                    mutated[i] = np.random.uniform(lower, upper)
                    
        elif strategy == 'swap':
            # Swap mutation
            if np.random.random() < mutation_rate and n_assets > 1:
                idx1, idx2 = np.random.choice(n_assets, 2, replace=False)
                mutated[idx1], mutated[idx2] = mutated[idx2], mutated[idx1]
                
        else:  # gaussian
            # Gaussian mutation
            for i in range(n_assets):
                if np.random.random() < mutation_rate:
                    # Add Gaussian noise
                    sigma = 0.1  # Standard deviation
                    mutated[i] += np.random.normal(0, sigma)
                    # Ensure bounds
                    lower, upper = bounds[i]
                    mutated[i] = np.clip(mutated[i], lower, upper)
        
        # Ensure non-negative weights
        mutated = np.maximum(mutated, 0.0)
        
        # Normalize to sum to one
        if np.sum(mutated) > 0:
            mutated = mutated / np.sum(mutated)
        else:
            # If all weights became zero, reset to uniform
            mutated = np.ones(n_assets) / n_assets
        
        return mutated
    
    def _perform_migration(self, islands: List[List[np.ndarray]]) -> List[List[np.ndarray]]:
        """Perform migration between islands.
        
        Args:
            islands: List of island populations
            
        Returns:
            Updated list of island populations after migration
        """
        # Number of individuals to migrate from each island
        migrants_count = int(self.migration_rate * self.population_size)
        
        # For each island, select migrants and destinations
        for source_idx in range(self.num_islands):
            # Evaluate current island population
            source_island = islands[source_idx]
            
            # Select best individuals as migrants
            temp_fitness = np.array([np.sum(ind) for ind in source_island])  # Simple fitness for selection
            migrant_indices = np.argsort(temp_fitness)[-migrants_count:]
            migrants = [source_island[i].copy() for i in migrant_indices]
            
            # Determine destination island (next island in circular fashion)
            dest_idx = (source_idx + 1) % self.num_islands
            
            # Replace worst individuals in destination island
            dest_island = islands[dest_idx]
            temp_fitness = np.array([np.sum(ind) for ind in dest_island])
            replace_indices = np.argsort(temp_fitness)[:migrants_count]
            
            for i, migrant in enumerate(migrants):
                if i < len(replace_indices):
                    dest_island[replace_indices[i]] = migrant
        
        return islands
    
    def _calculate_diversity(self, population: List[np.ndarray]) -> float:
        """Calculate diversity of the population.
        
        Args:
            population: List of weight vectors
            
        Returns:
            Diversity measure (higher means more diverse)
        """
        if len(population) <= 1:
            return 0.0
        
        # Convert to numpy array for easier calculations
        pop_array = np.array(population)
        
        # Calculate pairwise distances
        n = len(population)
        distances = np.zeros((n, n))
        
        for i in range(n):
            for j in range(i+1, n):
                distances[i, j] = distances[j, i] = np.sum((pop_array[i] - pop_array[j])**2)
        
        # Average distance as diversity measure
        diversity = np.sum(distances) / (n * (n - 1))
        
        return diversity
    
    def _adapt_mutation_rate(self, diversity: float) -> float:
        """Adapt mutation rate based on population diversity.
        
        Args:
            diversity: Current population diversity
            
        Returns:
            Adapted mutation rate
        """
        # Low diversity -> increase mutation rate
        # High diversity -> decrease mutation rate
        base_rate = self.mutation_rate
        
        # Normalize diversity to [0, 1] range assuming max diversity is around 0.5
        normalized_diversity = min(1.0, diversity / 0.5)
        
        # Adjust rate: higher when diversity is low, lower when diversity is high
        adjusted_rate = base_rate * (1.5 - normalized_diversity)
        
        # Ensure rate stays within reasonable bounds
        return np.clip(adjusted_rate, 0.01, 0.5)
    
    def _adapt_crossover_rate(self, diversity: float) -> float:
        """Adapt crossover rate based on population diversity.
        
        Args:
            diversity: Current population diversity
            
        Returns:
            Adapted crossover rate
        """
        # Low diversity -> decrease crossover rate
        # High diversity -> increase crossover rate
        base_rate = self.crossover_rate
        
        # Normalize diversity to [0, 1] range assuming max diversity is around 0.5
        normalized_diversity = min(1.0, diversity / 0.5)
        
        # Adjust rate: lower when diversity is low, higher when diversity is high
        adjusted_rate = base_rate * (0.5 + normalized_diversity)
        
        # Ensure rate stays within reasonable bounds
        return np.clip(adjusted_rate, 0.5, 0.95)
    
    def process_weights(self, weights: np.ndarray, min_weight: float = 0.0, 
                        problem: Optional[PortfolioOptProblem] = None) -> np.ndarray:
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
        if problem is not None and problem.constraints is not None and 'cardinality' in problem.constraints:
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
