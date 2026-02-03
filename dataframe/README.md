# Simple DataFrame Library

A lightweight, pandas-like DataFrame implementation for educational purposes. Built from scratch to understand how data manipulation libraries work under the hood.

## Features

- **DataFrame**: 2-dimensional labeled data structure
- **Series**: 1-dimensional labeled array
- **Data Selection**: iloc, filter, head, tail
- **Data Manipulation**: sort, drop, rename, apply
- **GroupBy Operations**: groupby with aggregations (sum, mean, min, max, count)
- **Join Operations**: inner, left, right, outer joins
- **Concatenation**: Row-wise and column-wise concatenation
- **CSV I/O**: Read and write CSV files
- **Type Inference**: Automatic type conversion from CSV

## Installation

```bash
# Clone the repository
cd /path/to/sync_project/20260203

# The library is in the `dataframe` folder
# You can use it directly by importing
```

## Quick Start

```python
from dataframe import DataFrame, Series

# Create a DataFrame
df = DataFrame({
    'name': ['Alice', 'Bob', 'Charlie'],
    'age': [25, 30, 35],
    'city': ['NYC', 'LA', 'SF']
})

print(df)
# DataFrame(rows=3, cols=3, columns=[name, age, city])

# Access columns
print(df['name'])  # ['Alice', 'Bob', 'Charlie']

# Select specific columns
result = df[['name', 'age']]

# Filter rows
result = df.filter(lambda row: row['age'] > 25)

# Sort
result = df.sort_values('age', ascending=False)

# Apply function
result = df.apply(lambda x: x * 2, 'age')
```

## DataFrame Operations

### Selection

```python
# Select specific columns
df[['name', 'age']]

# Get first n rows
df.head(5)

# Get last n rows
df.tail(5)

# Integer-location based indexing
row = df.iloc[0]
rows = df.iloc[1:5]
```

### Filtering

```python
# Filter with condition
result = df.filter(lambda row: row['age'] > 30)

# Alternative with loc
result = df.loc(lambda row: row['city'] == 'NYC')
```

### Sorting

```python
# Sort by column (ascending)
df.sort_values('age')

# Sort by column (descending)
df.sort_values('age', ascending=False)
```

### Manipulation

```python
# Drop columns
df.drop(['unwanted_column'])

# Rename columns
df.rename({'old_name': 'new_name'})

# Apply function to column
df.apply(lambda x: x * 1.1, 'salary')

# Add new column
df['bonus'] = [5000, 3000, 7000]
```

## GroupBy Operations

```python
# Group by a column
grouped = df.groupby('department')

# Aggregations
grouped.sum()
grouped.mean()
grouped.count()
grouped.min()
grouped.max()

# Custom aggregation
grouped.agg({
    'salary': 'sum',
    'age': 'mean'
})
```

## Join Operations

```python
from dataframe.operations import join

# Inner join
result = join(df1, df2, on='id', how='inner')

# Left join
result = join(df1, df2, on='id', how='left')

# Right join
result = join(df1, df2, on='id', how='right')

# Outer join
result = join(df1, df2, on='id', how='outer')
```

## Concatenation

```python
from dataframe.operations import concat

# Concatenate rows (axis=0)
result = concat([df1, df2, df3])

# Concatenate columns (axis=1)
result = concat([df1, df2], axis=1)
```

## CSV I/O

```python
# Write to CSV
df.to_csv('output.csv')

# Read from CSV
df = DataFrame.read_csv('data.csv')
```

## Parquet I/O

```python
# Write to Parquet (requires pyarrow)
df.to_parquet('output.parquet')

# Read from Parquet (requires pyarrow)
df = DataFrame.read_parquet('data.parquet')
```

**Note**: Parquet support requires `pyarrow`. Install with:
```bash
pip install pyarrow
```

## Series Operations

```python
from dataframe import Series

# Create a Series
s = Series([1, 2, 3, 4, 5], name='numbers')

# Statistics
s.sum()      # 15
s.mean()     # 3.0
s.min()      # 1
s.max()      # 5
s.count()    # 5

# Map and filter
s.map(lambda x: x * 2)        # [2, 4, 6, 8, 10]
s.filter(lambda x: x > 2)     # [3, 4, 5]

# Unique values
s.unique()          # [1, 2, 3, 4, 5]
s.value_counts()    # {1: 1, 2: 1, 3: 1, 4: 1, 5: 1}
```

## Examples

Run the example script to see various features in action:

```bash
cd dataframe/examples
python basic_usage.py
```

## Running Tests

```bash
cd dataframe/tests
python test_dataframe.py
```

Or using pytest:

```bash
pytest dataframe/tests/
```

## Project Structure

```
dataframe/
├── __init__.py      # Package initialization
├── core.py          # DataFrame class
├── series.py        # Series class
├── operations.py    # GroupBy, join, concat
├── utils.py         # Utility functions
├── tests/
│   └── test_dataframe.py  # Unit tests
└── examples/
    └── basic_usage.py     # Usage examples
```

## Limitations

This is an educational implementation. For production use, consider:
- **Performance**: This is much slower than pandas
- **Features**: Missing many pandas features (multi-index, time series, etc.)
- **Memory**: Not optimized for large datasets
- **Error Handling**: Basic error handling, may not cover all edge cases

## License

MIT License - feel free to use for learning and experimentation!

## Contributing

This is a learning project. Feel free to:
- Report issues
- Suggest improvements
- Submit pull requests

## Acknowledgments

Inspired by pandas and other data analysis libraries. Built to understand the fundamentals of data manipulation in Python.
