"""
Simple DataFrame Library
A lightweight pandas-like DataFrame implementation for educational purposes.
"""

from .core import DataFrame
from .series import Series
from .operations import (
    filter_rows,
    select_columns,
    group_by,
    join,
    merge,
    concat
)

__version__ = "0.1.0"
__all__ = [
    "DataFrame",
    "Series",
    "filter_rows",
    "select_columns",
    "group_by",
    "join",
    "merge",
    "concat",
]
