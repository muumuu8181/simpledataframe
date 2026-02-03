"""
Unit tests for the DataFrame library
"""

import sys
import os
import tempfile
import unittest

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from dataframe.core import DataFrame
from dataframe.series import Series
from dataframe.operations import group_by, join, merge, concat


class TestSeries(unittest.TestCase):
    """Test Series functionality."""

    def test_series_creation(self):
        """Test creating a Series."""
        s = Series([1, 2, 3, 4, 5], name="numbers")
        self.assertEqual(len(s), 5)
        self.assertEqual(s.name, "numbers")
        self.assertEqual(s[0], 1)
        self.assertEqual(s[4], 5)

    def test_series_sum(self):
        """Test Series sum."""
        s = Series([1, 2, 3, 4, 5])
        self.assertEqual(s.sum(), 15)

    def test_series_mean(self):
        """Test Series mean."""
        s = Series([1, 2, 3, 4, 5])
        self.assertEqual(s.mean(), 3.0)

    def test_series_map(self):
        """Test Series map."""
        s = Series([1, 2, 3])
        result = s.map(lambda x: x * 2)
        self.assertEqual(result.data, [2, 4, 6])

    def test_series_filter(self):
        """Test Series filter."""
        s = Series([1, 2, 3, 4, 5])
        result = s.filter(lambda x: x > 2)
        self.assertEqual(result.data, [3, 4, 5])

    def test_series_unique(self):
        """Test Series unique."""
        s = Series([1, 2, 2, 3, 3, 3])
        self.assertEqual(s.unique(), [1, 2, 3])

    def test_series_value_counts(self):
        """Test Series value_counts."""
        s = Series([1, 2, 2, 3, 3, 3])
        counts = s.value_counts()
        self.assertEqual(counts[3], 3)
        self.assertEqual(counts[2], 2)
        self.assertEqual(counts[1], 1)


class TestDataFrame(unittest.TestCase):
    """Test DataFrame functionality."""

    def test_dataframe_creation_column_oriented(self):
        """Test creating DataFrame from column-oriented dict."""
        df = DataFrame({
            'name': ['Alice', 'Bob', 'Charlie'],
            'age': [25, 30, 35],
            'city': ['NYC', 'LA', 'SF']
        })
        self.assertEqual(df.shape, (3, 3))
        self.assertEqual(len(df), 3)
        self.assertEqual(set(df.columns), {'name', 'age', 'city'})

    def test_dataframe_creation_row_oriented(self):
        """Test creating DataFrame from row-oriented list."""
        data = [
            {'name': 'Alice', 'age': 25},
            {'name': 'Bob', 'age': 30},
            {'name': 'Charlie', 'age': 35}
        ]
        df = DataFrame(data)
        self.assertEqual(df.shape, (3, 2))
        self.assertEqual(df['age'], [25, 30, 35])

    def test_get_column(self):
        """Test getting a column."""
        df = DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
        self.assertEqual(df['a'], [1, 2, 3])
        self.assertEqual(df['b'], [4, 5, 6])

    def test_select_columns(self):
        """Test selecting multiple columns."""
        df = DataFrame({
            'a': [1, 2, 3],
            'b': [4, 5, 6],
            'c': [7, 8, 9]
        })
        result = df[['a', 'c']]
        self.assertEqual(result.columns, ['a', 'c'])
        self.assertEqual(result['a'], [1, 2, 3])

    def test_head(self):
        """Test head method."""
        df = DataFrame({'a': range(10)})
        result = df.head(3)
        self.assertEqual(len(result), 3)
        self.assertEqual(result['a'], [0, 1, 2])

    def test_tail(self):
        """Test tail method."""
        df = DataFrame({'a': range(10)})
        result = df.tail(3)
        self.assertEqual(len(result), 3)
        self.assertEqual(result['a'], [7, 8, 9])

    def test_iloc_single_row(self):
        """Test iloc with single index."""
        df = DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
        row = df.iloc[1]
        self.assertEqual(row, {'a': 2, 'b': 5})

    def test_iloc_slice(self):
        """Test iloc with slice."""
        df = DataFrame({'a': [1, 2, 3, 4, 5], 'b': [6, 7, 8, 9, 10]})
        result = df.iloc[1:4]
        self.assertEqual(len(result), 3)
        self.assertEqual(result['a'], [2, 3, 4])

    def test_filter(self):
        """Test filter method."""
        df = DataFrame({'a': [1, 2, 3, 4, 5], 'b': [6, 7, 8, 9, 10]})
        result = df.filter(lambda row: row['a'] > 2)
        self.assertEqual(len(result), 3)
        self.assertEqual(result['a'], [3, 4, 5])

    def test_sort_values(self):
        """Test sort_values method."""
        df = DataFrame({'a': [3, 1, 4, 1, 5], 'b': [6, 7, 8, 9, 10]})
        result = df.sort_values('a')
        self.assertEqual(result['a'], [1, 1, 3, 4, 5])

        result_desc = df.sort_values('a', ascending=False)
        self.assertEqual(result_desc['a'], [5, 4, 3, 1, 1])

    def test_drop(self):
        """Test drop method."""
        df = DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6], 'c': [7, 8, 9]})
        result = df.drop(['b'])
        self.assertEqual(result.columns, ['a', 'c'])

    def test_rename(self):
        """Test rename method."""
        df = DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
        result = df.rename({'a': 'x', 'b': 'y'})
        self.assertEqual(result.columns, ['x', 'y'])

    def test_apply(self):
        """Test apply method."""
        df = DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
        result = df.apply(lambda x: x * 2, 'a')
        self.assertEqual(result['a'], [2, 4, 6])
        self.assertEqual(result['b'], [4, 5, 6])

    def test_to_dict(self):
        """Test to_dict method."""
        df = DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
        result = df.to_dict('records')
        self.assertEqual(result, [
            {'a': 1, 'b': 4},
            {'a': 2, 'b': 5},
            {'a': 3, 'b': 6}
        ])


class TestGroupBy(unittest.TestCase):
    """Test GroupBy functionality."""

    def setUp(self):
        """Set up test DataFrame."""
        self.df = DataFrame({
            'category': ['A', 'B', 'A', 'B', 'A'],
            'value': [10, 20, 30, 40, 50],
            'count': [1, 2, 3, 4, 5]
        })

    def test_groupby_sum(self):
        """Test groupby sum."""
        result = self.df.groupby('category').sum()
        self.assertEqual(len(result), 2)
        # Category A: 10+30+50=90, 1+3+5=9
        # Category B: 20+40=60, 2+4=6

    def test_groupby_mean(self):
        """Test groupby mean."""
        result = self.df.groupby('category').mean()
        self.assertEqual(len(result), 2)

    def test_groupby_agg(self):
        """Test groupby agg."""
        result = self.df.groupby('category').agg({
            'value': 'sum',
            'count': 'mean'
        })
        self.assertEqual(len(result), 2)


class TestJoin(unittest.TestCase):
    """Test join functionality."""

    def test_inner_join(self):
        """Test inner join."""
        left = DataFrame({'id': [1, 2, 3], 'name': ['Alice', 'Bob', 'Charlie']})
        right = DataFrame({'id': [2, 3, 4], 'age': [30, 35, 40]})

        result = join(left, right, on='id', how='inner')
        self.assertEqual(len(result), 2)
        self.assertIn('name', result.columns)
        self.assertIn('age', result.columns)

    def test_left_join(self):
        """Test left join."""
        left = DataFrame({'id': [1, 2, 3], 'name': ['Alice', 'Bob', 'Charlie']})
        right = DataFrame({'id': [2, 3, 4], 'age': [30, 35, 40]})

        result = join(left, right, on='id', how='left')
        self.assertEqual(len(result), 3)


class TestConcat(unittest.TestCase):
    """Test concat functionality."""

    def test_concat_rows(self):
        """Test concatenating rows."""
        df1 = DataFrame({'a': [1, 2], 'b': [3, 4]})
        df2 = DataFrame({'a': [5, 6], 'b': [7, 8]})

        result = concat([df1, df2])
        self.assertEqual(len(result), 4)
        self.assertEqual(result['a'], [1, 2, 5, 6])

    def test_concat_columns(self):
        """Test concatenating columns."""
        df1 = DataFrame({'a': [1, 2], 'b': [3, 4]})
        df2 = DataFrame({'c': [5, 6], 'd': [7, 8]})

        result = concat([df1, df2], axis=1)
        self.assertEqual(len(result.columns), 4)
        self.assertEqual(len(result), 2)


class TestCSV(unittest.TestCase):
    """Test CSV I/O."""

    def test_read_write_csv(self):
        """Test reading and writing CSV."""
        df = DataFrame({
            'name': ['Alice', 'Bob', 'Charlie'],
            'age': [25, 30, 35],
            'city': ['NYC', 'LA', 'SF']
        })

        # Write to temp file
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv') as f:
            temp_path = f.name

        try:
            df.to_csv(temp_path)
            result = DataFrame.read_csv(temp_path)

            self.assertEqual(result.shape, df.shape)
            self.assertEqual(set(result.columns), set(df.columns))
            self.assertEqual(result['name'], ['Alice', 'Bob', 'Charlie'])
        finally:
            os.unlink(temp_path)


class TestParquet(unittest.TestCase):
    """Test Parquet I/O."""

    def test_read_write_parquet(self):
        """Test reading and writing Parquet."""
        try:
            import pyarrow
        except ImportError:
            self.skipTest("pyarrow not installed")

        df = DataFrame({
            'name': ['Alice', 'Bob', 'Charlie'],
            'age': [25, 30, 35],
            'city': ['NYC', 'LA', 'SF']
        })

        # Write to temp file
        with tempfile.NamedTemporaryFile(mode='wb', delete=False, suffix='.parquet') as f:
            temp_path = f.name

        try:
            df.to_parquet(temp_path)
            result = DataFrame.read_parquet(temp_path)

            self.assertEqual(result.shape, df.shape)
            self.assertEqual(set(result.columns), set(df.columns))
            self.assertEqual(result['name'], ['Alice', 'Bob', 'Charlie'])
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)


if __name__ == '__main__':
    unittest.main()
