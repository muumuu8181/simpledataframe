"""
DataFrame operations module - Data manipulation operations
"""

from typing import Any, List, Dict, Callable, Union
from collections import defaultdict
from .core import DataFrame


def filter_rows(df: DataFrame, condition: Callable[[Dict[str, Any]], bool]) -> DataFrame:
    """
    Filter DataFrame rows based on a condition.

    Args:
        df: Source DataFrame
        condition: Function that takes a row dict and returns bool

    Returns:
        New DataFrame with filtered rows
    """
    return df.filter(condition)


def select_columns(df: DataFrame, columns: List[str]) -> DataFrame:
    """
    Select specific columns from DataFrame.

    Args:
        df: Source DataFrame
        columns: List of column names to select

    Returns:
        New DataFrame with selected columns
    """
    return df.select(columns)


class GroupBy:
    """
    GroupBy object for grouped operations.
    """

    def __init__(self, df: DataFrame, by: str):
        """
        Initialize GroupBy.

        Args:
            df: Source DataFrame
            by: Column name to group by
        """
        if by not in df.columns:
            raise ValueError(f"Column '{by}' not found")
        self._df = df
        self._by = by
        self._groups = self._create_groups()

    def _create_groups(self) -> Dict[Any, List[int]]:
        """Create groups based on unique values.

        Optimized with defaultdict and direct data access.
        """
        # Use defaultdict to avoid checking if key exists
        groups: Dict[Any, List[int]] = defaultdict(list)

        # Direct access to underlying data for performance
        column_data = self._df._data[self._by]

        for i, key in enumerate(column_data):
            groups[key].append(i)

        return dict(groups)  # Convert back to regular dict

    @property
    def groups(self) -> Dict[Any, List[int]]:
        """Get the groups dictionary."""
        return self._groups

    def agg(self, aggregations: Dict[str, Union[str, Callable]]) -> DataFrame:
        """
        Aggregate each group with specified functions.

        Args:
            aggregations: Dict of column -> aggregation function
                         ('sum', 'mean', 'min', 'max', 'count') or callable

        Returns:
            New DataFrame with aggregated results
        """
        result_data = {self._by: []}

        # Initialize result columns
        for col, func in aggregations.items():
            if col != self._by:
                result_data[col] = []

        # Cache column data for direct access
        df_data = self._df._data

        for key, indices in self._groups.items():
            result_data[self._by].append(key)

            # Apply aggregations
            for col, func in aggregations.items():
                if col == self._by:
                    continue
                # Direct access to underlying data for performance
                if col in df_data:
                    col_data = df_data[col]
                    values = [col_data[i] for i in indices]
                else:
                    values = []
                result_data[col].append(_apply_aggregation(values, func))

        return DataFrame(result_data)

    def sum(self) -> DataFrame:
        """Sum of each group."""
        agg_dict = {col: 'sum' for col in self._df.columns if col != self._by}
        return self.agg(agg_dict)

    def mean(self) -> DataFrame:
        """Mean of each group."""
        agg_dict = {col: 'mean' for col in self._df.columns if col != self._by}
        return self.agg(agg_dict)

    def count(self) -> DataFrame:
        """Count of each group."""
        agg_dict = {col: 'count' for col in self._df.columns if col != self._by}
        return self.agg(agg_dict)

    def min(self) -> DataFrame:
        """Minimum of each group."""
        agg_dict = {col: 'min' for col in self._df.columns if col != self._by}
        return self.agg(agg_dict)

    def max(self) -> DataFrame:
        """Maximum of each group."""
        agg_dict = {col: 'max' for col in self._df.columns if col != self._by}
        return self.agg(agg_dict)


def _apply_aggregation(values: List[Any], func: Union[str, Callable]) -> Any:
    """Apply aggregation function to values."""
    if isinstance(func, str):
        numeric_values = [v for v in values if isinstance(v, (int, float))]

        if func == 'sum':
            return sum(numeric_values) if numeric_values else 0
        elif func == 'mean':
            return sum(numeric_values) / len(numeric_values) if numeric_values else 0
        elif func == 'min':
            return min(numeric_values) if numeric_values else None
        elif func == 'max':
            return max(numeric_values) if numeric_values else None
        elif func == 'count':
            return len([v for v in values if v is not None])
        else:
            raise ValueError(f"Unknown aggregation: {func}")
    elif callable(func):
        return func(values)
    else:
        raise TypeError(f"Invalid aggregation type: {type(func)}")


def group_by(df: DataFrame, by: str) -> GroupBy:
    """
    Group DataFrame by a column.

    Args:
        df: Source DataFrame
        by: Column name to group by

    Returns:
        GroupBy object
    """
    return GroupBy(df, by)


def join(left: DataFrame, right: DataFrame, on: str, how: str = 'inner') -> DataFrame:
    """
    Join two DataFrames on a column.

    Args:
        left: Left DataFrame
        right: Right DataFrame
        on: Column name to join on
        how: Type of join ('inner', 'left', 'right', 'outer')

    Returns:
        New joined DataFrame
    """
    if on not in left.columns:
        raise ValueError(f"Column '{on}' not found in left DataFrame")
    if on not in right.columns:
        raise ValueError(f"Column '{on}' not found in right DataFrame")

    # Create lookup dict from right DataFrame
    right_lookup = {}
    for i in range(len(right)):
        key = right[on][i]
        if key not in right_lookup:
            right_lookup[key] = []
        right_lookup[key].append(i)

    # Determine which columns come from where
    left_cols = [col for col in left.columns]
    right_cols = [col for col in right.columns if col != on]

    result_data = {col: [] for col in left_cols + right_cols}

    if how == 'inner':
        # Only keys that exist in both DataFrames
        for i in range(len(left)):
            key = left[on][i]
            if key in right_lookup:
                # Add left values
                for col in left_cols:
                    result_data[col].append(left[col][i])
                # Add right values
                right_idx = right_lookup[key][0]
                for col in right_cols:
                    result_data[col].append(right[col][right_idx])

    elif how == 'left':
        # All keys from left DataFrame
        for i in range(len(left)):
            key = left[on][i]
            # Add left values
            for col in left_cols:
                result_data[col].append(left[col][i])
            # Add right values (or None)
            if key in right_lookup:
                right_idx = right_lookup[key][0]
                for col in right_cols:
                    result_data[col].append(right[col][right_idx])
            else:
                for col in right_cols:
                    result_data[col].append(None)

    elif how == 'right':
        # All keys from right DataFrame
        for i in range(len(right)):
            key = right[on][i]
            # Add left values (or None)
            if key in left[on]:
                left_idx = left[on].index(key)
                for col in left_cols:
                    result_data[col].append(left[col][left_idx])
            else:
                for col in left_cols:
                    result_data[col].append(None)
            # Add right values
            for col in right_cols:
                result_data[col].append(right[col][i])

    elif how == 'outer':
        # All keys from both DataFrames
        all_keys = set(left[on]) | set(right[on])
        for key in all_keys:
            if key in left[on] and key in right[on]:
                # Key in both
                left_idx = left[on].index(key)
                right_idx = right[on].index(key)
                for col in left_cols:
                    result_data[col].append(left[col][left_idx])
                for col in right_cols:
                    result_data[col].append(right[col][right_idx])
            elif key in left[on]:
                # Key only in left
                left_idx = left[on].index(key)
                for col in left_cols:
                    result_data[col].append(left[col][left_idx])
                for col in right_cols:
                    result_data[col].append(None)
            else:
                # Key only in right
                right_idx = right[on].index(key)
                for col in left_cols:
                    result_data[col].append(None)
                for col in right_cols:
                    result_data[col].append(right[col][right_idx])

    return DataFrame(result_data)


def merge(left: DataFrame, right: DataFrame, on: str = None, how: str = 'inner') -> DataFrame:
    """
    Merge two DataFrames (alias for join).

    Args:
        left: Left DataFrame
        right: Right DataFrame
        on: Column name to merge on
        how: Type of merge ('inner', 'left', 'right', 'outer')

    Returns:
        New merged DataFrame
    """
    return join(left, right, on, how)


def concat(dfs: List[DataFrame], axis: int = 0) -> DataFrame:
    """
    Concatenate DataFrames.

    Args:
        dfs: List of DataFrames to concatenate
        axis: 0 for row-wise, 1 for column-wise

    Returns:
        New concatenated DataFrame
    """
    if not dfs:
        raise ValueError("No DataFrames provided")

    if axis == 0:
        # Row-wise concatenation
        all_columns = set()
        for df in dfs:
            all_columns.update(df.columns)

        result_data = {col: [] for col in all_columns}

        for df in dfs:
            for col in all_columns:
                if col in df.columns:
                    result_data[col].extend(df[col])
                else:
                    result_data[col].extend([None] * len(df))

        return DataFrame(result_data)

    elif axis == 1:
        # Column-wise concatenation
        if len(set(len(df) for df in dfs)) > 1:
            raise ValueError("All DataFrames must have the same length for axis=1")

        result_data = {}
        for df in dfs:
            result_data.update({col: vals.copy() for col, vals in df._data.items()})

        return DataFrame(result_data)

    else:
        raise ValueError(f"Invalid axis: {axis}. Use 0 or 1")
