# tests/test_types.py

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Any

@dataclass
class TestResult:
    """Container for test results."""
    test_name: str
    start_time: datetime
    end_time: datetime
    parameters: Dict[str, Any]
    metrics: Dict[str, float]
    constraints_satisfied: Dict[str, bool]
    additional_info: Dict[str, Any]

