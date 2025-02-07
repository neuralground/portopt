import logging
import sys
from pathlib import Path
from typing import Optional

def setup_logging(
    level: int = logging.INFO,
    log_file: Optional[str] = None,
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
) -> None:
    """Set up logging configuration.
    
    Args:
        level: Logging level
        log_file: Optional path to log file
        log_format: Format string for log messages
    """
    # Create logger
    root_logger = logging.getLogger("portopt")
    root_logger.setLevel(level)
    
    # Create formatter
    formatter = logging.Formatter(log_format)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # File handler if specified
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

class OptimizationLogger:
    """Logger for optimization process with detailed metrics."""
    
    def __init__(self, name: str):
        self.logger = logging.getLogger(f"portopt.optimization.{name}")
    
    def log_iteration(self, iteration: int, metrics: dict) -> None:
        """Log optimization iteration metrics."""
        self.logger.info(f"Iteration {iteration}:")
        for key, value in metrics.items():
            self.logger.info(f"  {key}: {value}")
    
    def log_constraint_violation(self, constraint: str, value: float, limit: float) -> None:
        """Log constraint violation."""
        self.logger.warning(
            f"Constraint violation - {constraint}: "
            f"value={value:.4f}, limit={limit:.4f}"
        )
    
    def log_solve_complete(self, objective: float, solve_time: float) -> None:
        """Log optimization completion."""
        self.logger.info(
            f"Optimization complete - "
            f"objective={objective:.6f}, "
            f"time={solve_time:.2f}s"
        )

