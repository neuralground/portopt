import numpy as np
from typing import Dict, Any, List, Tuple
import json
from pathlib import Path

class OptimizationDebugger:
    """Debug helper for optimization process."""
    
    def __init__(self, debug_dir: str = "debug"):
        self.debug_dir = Path(debug_dir)
        self.debug_dir.mkdir(parents=True, exist_ok=True)
        self.iteration_data = []
    
    def save_iteration(self, iteration: int, data: Dict[str, Any]) -> None:
        """Save iteration data for debugging."""
        # Convert numpy arrays to lists for JSON serialization
        serializable_data = {}
        for key, value in data.items():
            if isinstance(value, np.ndarray):
                serializable_data[key] = value.tolist()
            else:
                serializable_data[key] = value
        
        self.iteration_data.append({
            'iteration': iteration,
            'data': serializable_data
        })
    
    def save_problem_state(self, problem_data: Dict[str, Any], 
                          filename: str = "problem_state.json") -> None:
        """Save problem state for debugging."""
        serializable_data = {}
        for key, value in problem_data.items():
            if isinstance(value, np.ndarray):
                serializable_data[key] = value.tolist()
            elif isinstance(value, dict):
                serializable_data[key] = {
                    k: v.tolist() if isinstance(v, np.ndarray) else v
                    for k, v in value.items()
                }
            else:
                serializable_data[key] = value
        
        with open(self.debug_dir / filename, 'w') as f:
            json.dump(serializable_data, f, indent=2)
    
    def analyze_convergence(self) -> Dict[str, List[float]]:
        """Analyze convergence patterns in optimization."""
        metrics = {}
        for field in ['objective', 'turnover', 'active_positions']:
            if any(field in d['data'] for d in self.iteration_data):
                metrics[field] = [
                    d['data'].get(field, float('nan')) 
                    for d in self.iteration_data
                ]
        return metrics
    
    def find_constraint_violations(self, 
                                 constraints: Dict[str, float]) -> List[Tuple[int, str, float]]:
        """Find constraint violations across iterations."""
        violations = []
        for i, data in enumerate(self.iteration_data):
            for constraint, limit in constraints.items():
                if constraint in data['data']:
                    value = data['data'][constraint]
                    if value > limit:
                        violations.append((i, constraint, value))
        return violations
    
    def save_report(self, filename: str = "optimization_report.txt") -> None:
        """Generate and save detailed optimization report."""
        with open(self.debug_dir / filename, 'w') as f:
            f.write("Optimization Debug Report\n")
            f.write("=" * 50 + "\n\n")
            
            # Convergence analysis
            f.write("Convergence Analysis:\n")
            metrics = self.analyze_convergence()
            for metric, values in metrics.items():
                f.write(f"\n{metric.title()}:\n")
                f.write(f"  Initial: {values[0]:.6f}\n")
                f.write(f"  Final: {values[-1]:.6f}\n")
                f.write(f"  Min: {min(values):.6f}\n")
                f.write(f"  Max: {max(values):.6f}\n")
            
            # Iteration details
            f.write("\nIteration Details:\n")
            for data in self.iteration_data:
                f.write(f"\nIteration {data['iteration']}:\n")
                for key, value in data['data'].items():
                    if isinstance(value, list):
                        f.write(f"  {key}: [array of size {len(value)}]\n")
                    else:
                        f.write(f"  {key}: {value}\n")

