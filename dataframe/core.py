"""
Core DataFrame module - 2-dimensional labeled data structure
"""

import csv
from typing import Any, List, Dict, Union, Callable, Optional
from .series import Series
from .utils import validate_dict, validate_list

# Try to import NumPy for optimized operations
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    np = None
    HAS_NUMPY = False


class _ColumnProxy:
    """Proxy object for column data with comparison operators (NumPy-optimized)."""

    def __init__(self, data: Union[List[Any], 'np.ndarray'], df: 'DataFrame'):
        # Support both list and NumPy array input
        if HAS_NUMPY and isinstance(data, np.ndarray):
            self._data = data  # Already a NumPy array
            self._numpy_array = data  # Cache reference
        else:
            self._data = data  # Python list
            self._numpy_array = None
        self._df = df

    def __gt__(self, other):
        if HAS_NUMPY:
            try:
                # Use NumPy array directly if available, otherwise convert
                if isinstance(self._data, np.ndarray):
                    result = self._data > other
                else:
                    result = np.array(self._data) > other
                # Return new ColumnProxy with NumPy result
                proxy = _ColumnProxy(result, self._df)
                proxy._numpy_array = result
                return proxy
            except (TypeError, ValueError):
                return [x > other for x in self._data]
        return [x > other for x in self._data]

    def __lt__(self, other):
        if HAS_NUMPY:
            try:
                if isinstance(self._data, np.ndarray):
                    result = self._data < other
                else:
                    result = np.array(self._data) < other
                proxy = _ColumnProxy(result, self._df)
                proxy._numpy_array = result
                return proxy
            except (TypeError, ValueError):
                return [x < other for x in self._data]
        return [x < other for x in self._data]

    def __ge__(self, other):
        if HAS_NUMPY:
            try:
                if isinstance(self._data, np.ndarray):
                    result = self._data >= other
                else:
                    result = np.array(self._data) >= other
                proxy = _ColumnProxy(result, self._df)
                proxy._numpy_array = result
                return proxy
            except (TypeError, ValueError):
                return [x >= other for x in self._data]
        return [x >= other for x in self._data]

    def __le__(self, other):
        if HAS_NUMPY:
            try:
                if isinstance(self._data, np.ndarray):
                    result = self._data <= other
                else:
                    result = np.array(self._data) <= other
                proxy = _ColumnProxy(result, self._df)
                proxy._numpy_array = result
                return proxy
            except (TypeError, ValueError):
                return [x <= other for x in self._data]
        return [x <= other for x in self._data]

    def __eq__(self, other):
        if HAS_NUMPY:
            try:
                if isinstance(self._data, np.ndarray):
                    result = self._data == other
                else:
                    result = np.array(self._data) == other
                proxy = _ColumnProxy(result, self._df)
                proxy._numpy_array = result
                return proxy
            except (TypeError, ValueError):
                return [x == other for x in self._data]
        return [x == other for x in self._data]

    def __ne__(self, other):
        if HAS_NUMPY:
            try:
                if isinstance(self._data, np.ndarray):
                    result = self._data != other
                else:
                    result = np.array(self._data) != other
                proxy = _ColumnProxy(result, self._df)
                proxy._numpy_array = result
                return proxy
            except (TypeError, ValueError):
                return [x != other for x in self._data]
        return [x != other for x in self._data]

    def __and__(self, other):
        """Bitwise AND for combining boolean masks."""
        if len(other) != len(self._data):
            raise ValueError("Lengths must match")
        return [a and b for a, b in zip(self._data, other)]

    def __or__(self, other):
        """Bitwise OR for combining boolean masks."""
        if len(other) != len(self._data):
            raise ValueError("Lengths must match")
        return [a or b for a, b in zip(self._data, other)]

    def __invert__(self):
        """Bitwise NOT for inverting boolean mask."""
        return [not x for x in self._data]

    def __iter__(self):
        return iter(self._data)

    def __len__(self):
        return len(self._data)

    def __getitem__(self, key):
        return self._data[key]

    def to_list(self):
        """Convert to plain list."""
        return self._data.copy()


class _ILoc:
    """Integer-location based indexer."""

    def __init__(self, df: 'DataFrame'):
        self._df = df

    def __getitem__(self, key: Union[int, slice, List[int]]) -> Any:
        """Get item by integer location."""
        n_rows = len(self._df)

        if isinstance(key, int):
            # Single row
            if key < 0:
                key = n_rows + key
            if key < 0 or key >= n_rows:
                raise IndexError(f"Index {key} out of bounds")
            return {col: self._df._data[col][key] for col in self._df.columns}
        elif isinstance(key, slice):
            # Slice
            start, stop, step = key.indices(n_rows)
            indices = list(range(start, stop, step))
            return self._df._select_rows(indices)
        elif isinstance(key, list):
            # List of indices
            return self._df._select_rows(key)
        raise TypeError(f"Invalid iloc key type: {type(key)}")

    def __setitem__(self, key, value):
        """Set item by integer location (not implemented)."""
        raise NotImplementedError("iloc assignment is not supported")


class DataFrame:
    """
    A 2-dimensional labeled data structure with columns of potentially different types.
    """

    def __init__(self, data: Union[Dict[str, List[Any]], List[Dict[str, Any]]]):
        """
        Initialize a DataFrame.

        Args:
            data: Can be either:
                - Dict of column_name -> list of values
                - List of dicts (row-oriented data)
        """
        if isinstance(data, dict):
            # Column-oriented data - convert to NumPy arrays
            self._data: Dict[str, np.ndarray] = {}
            for k, v in data.items():
                try:
                    # Try to convert to NumPy array
                    self._data[k] = np.array(list(v))
                except (TypeError, ValueError):
                    # For non-numeric data, keep as object array
                    self._data[k] = np.array(list(v), dtype=object)
            self._validate_column_oriented()
        elif isinstance(data, list):
            # Row-oriented data
            self._data = self._from_row_oriented(data)
        else:
            raise TypeError("data must be a dict or list")

    def _validate_column_oriented(self):
        """Validate that all columns have the same length."""
        if not self._data:
            return
        lengths = [len(col) for col in self._data.values()]
        if len(set(lengths)) > 1:
            raise ValueError("All columns must have the same length")

    def _from_row_oriented(self, data: List[Dict[str, Any]]) -> Dict[str, np.ndarray]:
        """Convert row-oriented data to column-oriented NumPy arrays."""
        if not data:
            return {}
        columns = {}
        for row in data:
            for key, value in row.items():
                if key not in columns:
                    columns[key] = []
                columns[key].append(value)

        # Convert lists to NumPy arrays
        result = {}
        for k, v in columns.items():
            try:
                result[k] = np.array(v)
            except (TypeError, ValueError):
                result[k] = np.array(v, dtype=object)
        return result

    @property
    def columns(self) -> List[str]:
        """Get column names."""
        return list(self._data.keys())

    @property
    def shape(self) -> tuple:
        """Get DataFrame shape as (n_rows, n_cols)."""
        n_cols = len(self._data)
        n_rows = len(next(iter(self._data.values()))) if n_cols > 0 else 0
        return (n_rows, n_cols)

    @property
    def values(self) -> List[List[Any]]:
        """Get DataFrame as a 2D list (rows)."""
        if not self._data:
            return []
        n_rows = self.shape[0]
        return [[self._data[col][i].item() if hasattr(self._data[col][i], 'item') else self._data[col][i]
                for col in self.columns] for i in range(n_rows)]

    def __len__(self) -> int:
        """Return number of rows."""
        return self.shape[0]

    def __repr__(self) -> str:
        """String representation of the DataFrame."""
        n_rows, n_cols = self.shape
        col_str = ", ".join(self.columns)
        return f"DataFrame(rows={n_rows}, cols={n_cols}, columns=[{col_str}])"

    def __str__(self) -> str:
        """String representation."""
        return self.__repr__()

    def __getitem__(self, key: Union[str, List[str], slice, List[bool], '_ColumnProxy']) -> Any:
        """Get column(s) or rows by slicing or boolean indexing."""
        if isinstance(key, str):
            # Single column - return NumPy array directly wrapped in ColumnProxy
            col_data = self._data.get(key)
            if col_data is None:
                return _ColumnProxy([], self)
            return _ColumnProxy(col_data, self)
        elif isinstance(key, list):
            # Check if it's a boolean list for filtering
            if len(key) > 0 and isinstance(key[0], bool):
                # Boolean indexing - use optimized method
                if len(key) != len(self):
                    raise ValueError(f"Boolean array length {len(key)} doesn't match DataFrame length {len(self)}")
                return self._select_rows_by_boolean(key)
            else:
                # Multiple columns - keep as NumPy arrays
                return DataFrame({k: self._data[k].copy() for k in key if k in self._data})
        elif isinstance(key, _ColumnProxy):
            # Boolean indexing from ColumnProxy - check for NumPy array
            if HAS_NUMPY and hasattr(key, '_numpy_array') and key._numpy_array is not None:
                # Use NumPy array directly for faster indexing
                indices = np.where(key._numpy_array)[0]
                # Fast path with NumPy indices (no conversion overhead)
                new_data = {}
                for col in self.columns:
                    new_data[col] = self._data[col][indices]
                return DataFrame(new_data)
            else:
                # Fall back to regular boolean indexing
                bool_list = list(key)
                return self._select_rows_by_boolean(bool_list)
        elif isinstance(key, slice):
            # Row slicing
            start = key.start if key.start else 0
            stop = key.stop if key.stop else len(self)
            step = key.step if key.step else 1
            return self.iloc[start:stop:step]
        raise TypeError(f"Invalid key type: {type(key)}")
        raise TypeError(f"Invalid key type: {type(key)}")

    def __setitem__(self, key: str, value: List[Any]):
        """Set or add a column."""
        if len(value) != len(self):
            raise ValueError(f"Column length must match DataFrame length ({len(self)})")
        self._data[key] = list(value)

    def head(self, n: int = 5) -> 'DataFrame':
        """
        Return the first n rows.

        Args:
            n: Number of rows to return

        Returns:
            New DataFrame with first n rows
        """
        return self.iloc[:n]

    def tail(self, n: int = 5) -> 'DataFrame':
        """
        Return the last n rows.

        Args:
            n: Number of rows to return

        Returns:
            New DataFrame with last n rows
        """
        start = max(0, len(self) - n)
        return self.iloc[start:]

    @property
    def iloc(self) -> _ILoc:
        """
        Integer-location based indexer.

        Returns:
            _ILoc indexer object
        """
        return _ILoc(self)

    def _select_rows(self, indices: List[int]) -> 'DataFrame':
        """Select rows by indices."""
        new_data = {}
        for col in self.columns:
            new_data[col] = [self._data[col][i] for i in indices if 0 <= i < len(self._data[col])]
        return DataFrame(new_data)

    def _select_rows_by_boolean(self, bool_list: List[bool]) -> 'DataFrame':
        """
        Select rows by boolean mask (optimized for NumPy-backed storage).

        Args:
            bool_list: List of booleans indicating which rows to keep

        Returns:
            New DataFrame with filtered rows
        """
        # Convert boolean list to NumPy array for indexing
        bool_array = np.array(bool_list, dtype=bool)
        indices = np.where(bool_array)[0]

        # Direct NumPy indexing (no conversion overhead)
        new_data = {}
        for col in self.columns:
            col_array = self._data[col]
            new_data[col] = col_array[indices]
        return DataFrame(new_data)

    def loc(self, condition: Callable[[Dict[str, Any]], bool]) -> 'DataFrame':
        """
        Label-based indexing with a condition function.

        Args:
            condition: Function that takes a row dict and returns bool

        Returns:
            New DataFrame with filtered rows
        """
        indices = []
        for i in range(len(self)):
            row = {col: self._data[col][i] for col in self.columns}
            if condition(row):
                indices.append(i)
        return self._select_rows(indices)

    def filter(self, condition: Callable[[Dict[str, Any]], bool]) -> 'DataFrame':
        """
        Filter rows based on a condition function.

        Args:
            condition: Function that takes a row dict and returns bool

        Returns:
            New DataFrame with filtered rows
        """
        return self.loc(condition)

    def select(self, columns: List[str]) -> 'DataFrame':
        """
        Select specific columns.

        Args:
            columns: List of column names to select

        Returns:
            New DataFrame with selected columns
        """
        return self[columns]

    def sort_values(self, by: str, ascending: bool = True) -> 'DataFrame':
        """
        Sort DataFrame by a column.

        Args:
            by: Column name to sort by
            ascending: Sort ascending if True, descending if False

        Returns:
            New sorted DataFrame
        """
        if by not in self._data:
            raise ValueError(f"Column '{by}' not found")

        # Create list of (value, index) pairs
        pairs = [(self._data[by][i], i) for i in range(len(self))]
        pairs.sort(key=lambda x: x[0], reverse=not ascending)
        sorted_indices = [i for _, i in pairs]

        return self._select_rows(sorted_indices)

    def drop(self, columns: List[str]) -> 'DataFrame':
        """
        Drop specified columns.

        Args:
            columns: List of column names to drop

        Returns:
            New DataFrame without specified columns
        """
        new_data = {k: v for k, v in self._data.items() if k not in columns}
        return DataFrame(new_data)

    def rename(self, mapping: Dict[str, str]) -> 'DataFrame':
        """
        Rename columns.

        Args:
            mapping: Dict of old_name -> new_name

        Returns:
            New DataFrame with renamed columns
        """
        new_data = {}
        for old_name, values in self._data.items():
            new_name = mapping.get(old_name, old_name)
            new_data[new_name] = list(values)
        return DataFrame(new_data)

    def apply(self, func: Callable[[Any], Any], column: str = None) -> 'DataFrame':
        """
        Apply a function to a column or all columns.

        Args:
            func: Function to apply
            column: Column name to apply to (None for all columns)

        Returns:
            New DataFrame with applied function
        """
        if column:
            if column not in self._data:
                raise ValueError(f"Column '{column}' not found")
            new_data = {k: list(v) for k, v in self._data.items()}
            new_data[column] = [func(x) for x in new_data[column]]
        else:
            new_data = {k: [func(x) for x in v] for k, v in self._data.items()}
        return DataFrame(new_data)

    def groupby(self, by: str) -> 'GroupBy':
        """
        Group DataFrame by a column.

        Args:
            by: Column name to group by

        Returns:
            GroupBy object
        """
        from .operations import GroupBy
        return GroupBy(self, by)

    def to_dict(self, orient: str = "records") -> Union[List[Dict[str, Any]], Dict[str, List[Any]]]:
        """
        Convert DataFrame to dict.

        Args:
            orient: 'records' for row-oriented, 'dict' for column-oriented

        Returns:
            Dict representation
        """
        if orient == "records":
            # Return list of dicts (row-oriented)
            result = []
            for i in range(len(self)):
                row = {}
                for col in self.columns:
                    val = self._data[col][i]
                    # Convert NumPy scalar to Python value
                    if hasattr(val, 'item'):
                        row[col] = val.item()
                    else:
                        row[col] = val
                result.append(row)
            return result
        elif orient == "dict":
            # Convert NumPy arrays to lists
            return {k: v.tolist() for k, v in self._data.items()}
        raise ValueError(f"Invalid orient: {orient}. Use 'records' or 'dict'")

    def to_csv(self, path: str, index: bool = False):
        """
        Write DataFrame to CSV file.

        Args:
            path: File path to write to
            index: Whether to write row indices (not implemented, always False)
        """
        with open(path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=self.columns)
            writer.writeheader()
            for i in range(len(self)):
                row = {}
                for col in self.columns:
                    val = self._data[col][i]
                    # Convert NumPy scalar to Python value
                    if hasattr(val, 'item'):
                        row[col] = val.item()
                    else:
                        row[col] = val
                writer.writerow(row)

    @staticmethod
    def read_csv(path: str) -> 'DataFrame':
        """
        Read CSV file into DataFrame.

        Args:
            path: Path to CSV file

        Returns:
            New DataFrame
        """
        data = []
        with open(path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Try to convert numeric strings to numbers
                converted_row = {}
                for key, value in row.items():
                    converted_row[key] = _convert_string(value)
                data.append(converted_row)
        return DataFrame(data)

    def to_parquet(self, path: str):
        """
        Write DataFrame to Parquet file.

        Args:
            path: File path to write to

        Note:
            Requires pyarrow to be installed.
            Install with: pip install pyarrow
        """
        try:
            import pyarrow as pa
            import pyarrow.parquet as pq
        except ImportError:
            raise ImportError(
                "pyarrow is required for Parquet I/O. "
                "Install with: pip install pyarrow"
            )

        # Convert to PyArrow Table
        arrays = []
        for col in self.columns:
            arrays.append(pa.array(self._data[col]))

        table = pa.Table.from_arrays(arrays, names=self.columns)
        pq.write_table(table, path)

    @staticmethod
    def read_parquet(path: str) -> 'DataFrame':
        """
        Read Parquet file into DataFrame.

        Args:
            path: Path to Parquet file

        Returns:
            New DataFrame

        Note:
            Requires pyarrow to be installed.
            Install with: pip install pyarrow
        """
        try:
            import pyarrow.parquet as pq
        except ImportError:
            raise ImportError(
                "pyarrow is required for Parquet I/O. "
                "Install with: pip install pyarrow"
            )

        table = pq.read_table(path)

        # Convert to DataFrame
        data = {}
        for i, col_name in enumerate(table.column_names):
            col_data = table.column(i).to_pylist()
            data[col_name] = col_data

        return DataFrame(data)

    def copy(self) -> 'DataFrame':
        """Create a shallow copy of the DataFrame."""
        return DataFrame({k: v.copy() for k, v in self._data.items()})

    def info(self) -> str:
        """Print concise summary of DataFrame."""
        lines = [
            f"DataFrame Summary",
            f"{'=' * 40}",
            f"Rows: {len(self)}",
            f"Columns: {len(self.columns)}",
            f"{'-' * 40}",
            f"Column names: {', '.join(self.columns)}",
            f"{'=' * 40}",
        ]
        # Get column types
        for col in self.columns:
            values = self._data[col]
            types = set(type(v).__name__ for v in values if v is not None)
            type_str = ', '.join(sorted(types)) if types else 'empty'
            non_null = sum(1 for v in values if v is not None)
            lines.append(f"  {col}: {type_str} (non-null: {non_null}/{len(values)})")
        return '\n'.join(lines)

    def describe(self) -> 'DataFrame':
        """
        Generate descriptive statistics for numeric columns.

        Returns:
            New DataFrame with statistics
        """
        stats = {}
        for col in self.columns:
            values = [v for v in self._data[col] if isinstance(v, (int, float))]
            if values:
                stats[col] = {
                    'count': len(values),
                    'mean': sum(values) / len(values),
                    'min': min(values),
                    'max': max(values),
                }
        return DataFrame(stats)


def _convert_string(value: str) -> Any:
    """Convert string to appropriate type."""
    try:
        return int(value)
    except ValueError:
        try:
            return float(value)
        except ValueError:
            return value
