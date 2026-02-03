"""
Utility functions for the DataFrame library
"""

from typing import Any, List


def validate_dict(data: Any) -> None:
    """Validate that data is a dictionary."""
    if not isinstance(data, dict):
        raise TypeError(f"Expected dict, got {type(data)}")


def validate_list(data: Any) -> None:
    """Validate that data is a list."""
    if not isinstance(data, list):
        raise TypeError(f"Expected list, got {type(data)}")


def is_numeric(value: Any) -> bool:
    """Check if value is numeric."""
    return isinstance(value, (int, float))


def safe_divide(a: Any, b: Any) -> float:
    """Safely divide two values."""
    if is_numeric(a) and is_numeric(b) and b != 0:
        return a / b
    return 0.0


def get_unique_values(values: List[Any]) -> List[Any]:
    """Get unique values from a list."""
    seen = set()
    unique = []
    for v in values:
        if v not in seen:
            seen.add(v)
            unique.append(v)
    return unique


def format_value(value: Any, max_length: int = 20) -> str:
    """Format a value for display."""
    if value is None:
        return "None"
    s = str(value)
    if len(s) > max_length:
        return s[:max_length-3] + "..."
    return s
