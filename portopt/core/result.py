"""Result definition module."""

import numpy as np

class PortfolioOptResult:
    """Represents the result of a portfolio optimization attempt."""
    
    def __init__(self, weights: np.ndarray, objective_value: float, 
                 solve_time: float, feasible: bool):
        self.weights = weights
        self.objective_value = objective_value
        self.solve_time = solve_time
        self.feasible = feasible
        
    def __str__(self):
        return (f"Portfolio optimization result:\n"
                f"Objective value: {self.objective_value:.6f}\n"
                f"Solve time: {self.solve_time:.3f}s\n"
                f"Feasible: {self.feasible}")

