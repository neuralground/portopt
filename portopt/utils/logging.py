"""
Logging configuration and utilities for portfolio optimization.

This module provides logging functionality specifically designed for portfolio
optimization processes. It includes:

- Configurable logging setup with file and console output
- Specialized logger for optimization processes
- Structured logging of optimization metrics and progress
- Performance tracking and timing information
- Constraint violation reporting

The logging utilities help track the progress and results of optimization runs
and facilitate debugging of complex optimization problems.
"""

import logging
import sys
from pathlib import Path
from typing import Optional

def setup_logging(
    level: str = "WARNING",  # Changed default to WARNING
    log_file: Optional[str] = None,
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
) -> logging.Logger:
    """Set up logging configuration.
    
    Args:
        level: Logging level (default WARNING to suppress most messages)
        log_file: Optional path to log file
        log_format: Format string for log messages
    """
    # Convert string level to logging level
    numeric_level = getattr(logging, level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f'Invalid log level: {level}')
    
    # Create logger
    root_logger = logging.getLogger("portopt")
    root_logger.setLevel(numeric_level)
    
    # Remove any existing handlers
    root_logger.handlers = []
    
    # Create formatter
    formatter = logging.Formatter(log_format)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    console_handler.setLevel(numeric_level)
    root_logger.addHandler(console_handler)
    
    # File handler if specified
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        file_handler.setLevel(numeric_level)
        root_logger.addHandler(file_handler)
    
    return root_logger

class OptimizationLogger:
    """Logger for optimization process with detailed metrics."""
    
    def __init__(self, name: str):
        self.logger = logging.getLogger(f"portopt.optimization.{name}")
    
    def log_iteration(self, iteration: int, metrics: dict) -> None:
        """Log optimization iteration metrics."""
        self.logger.debug(f"Iteration {iteration}:")  # Changed to debug
        for key, value in metrics.items():
            self.logger.debug(f"  {key}: {value}")  # Changed to debug
    
    def log_constraint_violation(self, constraint: str, value: float, limit: float) -> None:
        """Log constraint violation."""
        self.logger.warning(
            f"Constraint violation - {constraint}: "
            f"value={value:.4f}, limit={limit:.4f}"
        )
    
    def log_solve_complete(self, objective: float, solve_time: float) -> None:
        """Log optimization completion."""
        self.logger.debug(  # Changed to debug
            f"Optimization complete - "
            f"objective={objective:.6f}, "
            f"time={solve_time:.2f}s"
        )
