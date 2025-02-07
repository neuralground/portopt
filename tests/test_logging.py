import logging
from portopt.utils.logging import setup_logging, OptimizationLogger

def test_setup_logging(tmp_path):
    log_file = tmp_path / "test.log"
    logger = setup_logging(level="DEBUG", log_file=str(log_file))
    assert logger.level == logging.DEBUG
    assert log_file.exists()

def test_optimization_logger():
    logger = OptimizationLogger("test")
    logger.log_iteration(1, {"objective": 1.0, "constraint_violation": 0.0})
    logger.log_solve_complete(objective=1.0, solve_time=0.5)

