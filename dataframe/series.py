"""
Series module - 1-dimensional labeled array
"""

from typing import Any, List, Union, Callable
from .utils import validate_list


class Series:
    """
    A 1-dimensional labeled array capable of holding any data type.
    """

    def __init__(self, data: List[Any], name: str = None):
        """
        Initialize a Series.

        Args:
            data: List of values
            name: Name of the series
        """
        if not isinstance(data, list):
            raise TypeError("data must be a list")
        self._data = data.copy()
        self._name = name

    @property
    def data(self) -> List[Any]:
        """Get the underlying data."""
        return self._data

    @property
    def name(self) -> str:
        """Get the series name."""
        return self._name

    @name.setter
    def name(self, value: str):
        """Set the series name."""
        self._name = value

    def __len__(self) -> int:
        """Return the length of the series."""
        return len(self._data)

    def __getitem__(self, key: int) -> Any:
        """Get item by index."""
        return self._data[key]

    def __setitem__(self, key: int, value: Any):
        """Set item by index."""
        self._data[key] = value

    def __repr__(self) -> str:
        """String representation of the series."""
        name_str = f"'{self._name}'" if self._name else "None"
        data_preview = str(self._data[:10])
        if len(self._data) > 10:
            data_preview = data_preview[:-1] + ", ...]"
        return f"Series(name={name_str}, data={data_preview})"

    def __str__(self) -> str:
        """String representation."""
        return self.__repr__()

    def map(self, func: Callable[[Any], Any]) -> 'Series':
        """
        Apply a function to each element.

        Args:
            func: Function to apply

        Returns:
            New Series with mapped values
        """
        return Series([func(x) for x in self._data], self._name)

    def filter(self, func: Callable[[Any], bool]) -> 'Series':
        """
        Filter elements based on a predicate function.

        Args:
            func: Predicate function returning bool

        Returns:
            New Series with filtered values
        """
        return Series([x for x in self._data if func(x)], self._name)

    def sum(self) -> Union[int, float]:
        """Return the sum of all values."""
        total = 0
        for x in self._data:
            if isinstance(x, (int, float)):
                total += x
        return total

    def mean(self) -> float:
        """Return the mean of all numeric values."""
        numeric_values = [x for x in self._data if isinstance(x, (int, float))]
        if not numeric_values:
            return 0.0
        return sum(numeric_values) / len(numeric_values)

    def min(self) -> Any:
        """Return the minimum value."""
        return min(self._data) if self._data else None

    def max(self) -> Any:
        """Return the maximum value."""
        return max(self._data) if self._data else None

    def count(self) -> int:
        """Return the count of non-None values."""
        return sum(1 for x in self._data if x is not None)

    def unique(self) -> List[Any]:
        """Return unique values."""
        seen = set()
        unique_list = []
        for x in self._data:
            if x not in seen:
                seen.add(x)
                unique_list.append(x)
        return unique_list

    def value_counts(self) -> dict:
        """Return count of unique values."""
        counts = {}
        for x in self._data:
            counts[x] = counts.get(x, 0) + 1
        return dict(sorted(counts.items(), key=lambda item: item[1], reverse=True))

    def to_list(self) -> List[Any]:
        """Convert Series to list."""
        return self._data.copy()
