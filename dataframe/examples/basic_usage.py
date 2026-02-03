"""
Basic usage examples for the DataFrame library
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from dataframe import DataFrame, Series
from dataframe.operations import group_by, join, concat


def example_1_basic_creation():
    """Example 1: Creating DataFrames."""
    print("=" * 50)
    print("Example 1: Creating DataFrames")
    print("=" * 50)

    # Column-oriented creation
    df = DataFrame({
        'name': ['Alice', 'Bob', 'Charlie', 'Diana', 'Eve'],
        'age': [25, 30, 35, 28, 32],
        'city': ['NYC', 'LA', 'SF', 'Boston', 'Seattle'],
        'salary': [70000, 80000, 90000, 75000, 85000]
    })

    print(f"\nDataFrame created:")
    print(f"  Shape: {df.shape}")
    print(f"  Columns: {df.columns}")
    print(f"  First 3 rows:")
    for i in range(3):
        row = df.iloc[i]
        print(f"    {row}")


def example_2_selection_filtering():
    """Example 2: Selecting and filtering data."""
    print("\n" + "=" * 50)
    print("Example 2: Selection and Filtering")
    print("=" * 50)

    df = DataFrame({
        'name': ['Alice', 'Bob', 'Charlie', 'Diana', 'Eve'],
        'age': [25, 30, 35, 28, 32],
        'city': ['NYC', 'LA', 'SF', 'Boston', 'Seattle'],
        'salary': [70000, 80000, 90000, 75000, 85000]
    })

    # Select specific columns
    print("\nSelect 'name' and 'salary' columns:")
    result = df[['name', 'salary']]
    print(f"  Columns: {result.columns}")

    # Filter rows
    print("\nFilter people with age > 30:")
    result = df.filter(lambda row: row['age'] > 30)
    print(f"  Count: {len(result)}")
    for i in range(len(result)):
        print(f"    {result.iloc[i]}")

    # Filter by salary
    print("\nFilter people with salary >= 80000:")
    result = df.filter(lambda row: row['salary'] >= 80000)
    print(f"  Count: {len(result)}")


def example_3_sorting_manipulation():
    """Example 3: Sorting and manipulation."""
    print("\n" + "=" * 50)
    print("Example 3: Sorting and Manipulation")
    print("=" * 50)

    df = DataFrame({
        'name': ['Alice', 'Bob', 'Charlie', 'Diana', 'Eve'],
        'age': [25, 30, 35, 28, 32],
        'salary': [70000, 80000, 90000, 75000, 85000]
    })

    # Sort by age
    print("\nSort by age (ascending):")
    result = df.sort_values('age')
    for i in range(len(result)):
        row = result.iloc[i]
        print(f"  {row['name']}: {row['age']}")

    # Sort by salary descending
    print("\nSort by salary (descending):")
    result = df.sort_values('salary', ascending=False)
    for i in range(len(result)):
        row = result.iloc[i]
        print(f"  {row['name']}: ${row['salary']:,}")

    # Apply function to column
    print("\nApply 10% bonus to salary:")
    result = df.apply(lambda x: x * 1.1, 'salary')
    for i in range(len(result)):
        row = result.iloc[i]
        print(f"  {row['name']}: ${row['salary']:,.2f}")


def example_4_groupby():
    """Example 4: GroupBy operations."""
    print("\n" + "=" * 50)
    print("Example 4: GroupBy Operations")
    print("=" * 50)

    df = DataFrame({
        'department': ['Sales', 'Sales', 'Engineering', 'Engineering', 'HR', 'HR'],
        'employee': ['Alice', 'Bob', 'Charlie', 'Diana', 'Eve', 'Frank'],
        'salary': [70000, 80000, 90000, 95000, 60000, 65000],
        'experience': [2, 5, 8, 10, 1, 3]
    })

    print("\nOriginal data:")
    print(f"  Rows: {len(df)}, Columns: {df.columns}")

    # Group by department and sum
    print("\nGroup by department - Sum of salaries:")
    result = df.groupby('department').agg({'salary': 'sum'})
    for i in range(len(result)):
        row = result.iloc[i]
        print(f"  {row['department']}: ${row['salary']:,}")

    # Group by department and get mean
    print("\nGroup by department - Average salary:")
    result = df.groupby('department').agg({'salary': 'mean'})
    for i in range(len(result)):
        row = result.iloc[i]
        print(f"  {row['department']}: ${row['salary']:,.2f}")

    # Group by department and count
    print("\nGroup by department - Employee count:")
    result = df.groupby('department').agg({'employee': 'count'})
    for i in range(len(result)):
        row = result.iloc[i]
        print(f"  {row['department']}: {row['employee']} employees")


def example_5_join():
    """Example 5: Joining DataFrames."""
    print("\n" + "=" * 50)
    print("Example 5: Joining DataFrames")
    print("=" * 50)

    employees = DataFrame({
        'emp_id': [1, 2, 3, 4],
        'name': ['Alice', 'Bob', 'Charlie', 'Diana'],
        'dept_id': [101, 102, 101, 103]
    })

    departments = DataFrame({
        'dept_id': [101, 102, 103],
        'dept_name': ['Engineering', 'Sales', 'HR'],
        'location': ['SF', 'LA', 'NYC']
    })

    print("\nEmployees:")
    print(f"  {employees.shape[0]} employees")

    print("\nDepartments:")
    print(f"  {departments.shape[0]} departments")

    # Inner join
    print("\nInner join - Employees with departments:")
    result = join(employees, departments, on='dept_id', how='inner')
    for i in range(len(result)):
        row = result.iloc[i]
        print(f"  {row['name']} - {row['dept_name']} ({row['location']})")


def example_6_concat():
    """Example 6: Concatenating DataFrames."""
    print("\n" + "=" * 50)
    print("Example 6: Concatenating DataFrames")
    print("=" * 50)

    df1 = DataFrame({
        'name': ['Alice', 'Bob'],
        'age': [25, 30]
    })

    df2 = DataFrame({
        'name': ['Charlie', 'Diana'],
        'age': [35, 28]
    })

    print("\nDataFrame 1:")
    print(f"  {len(df1)} rows")

    print("\nDataFrame 2:")
    print(f"  {len(df2)} rows")

    # Concatenate rows
    print("\nConcatenate rows:")
    result = concat([df1, df2])
    for i in range(len(result)):
        row = result.iloc[i]
        print(f"  {row['name']}: {row['age']}")


def example_7_series():
    """Example 7: Series operations."""
    print("\n" + "=" * 50)
    print("Example 7: Series Operations")
    print("=" * 50)

    # Create a Series
    s = Series([10, 20, 30, 40, 50], name="numbers")

    print(f"\nSeries: {s}")
    print(f"  Sum: {s.sum()}")
    print(f"  Mean: {s.mean()}")
    print(f"  Min: {s.min()}")
    print(f"  Max: {s.max()}")

    # Map operation
    print("\nMap (multiply by 2):")
    result = s.map(lambda x: x * 2)
    print(f"  {result.data}")

    # Filter operation
    print("\nFilter (values > 25):")
    result = s.filter(lambda x: x > 25)
    print(f"  {result.data}")

    # Value counts
    s2 = Series(['a', 'b', 'a', 'c', 'b', 'a'])
    print("\nValue counts:")
    print(f"  {s2.value_counts()}")


def example_8_csv_io():
    """Example 8: CSV I/O."""
    print("\n" + "=" * 50)
    print("Example 8: CSV I/O")
    print("=" * 50)

    df = DataFrame({
        'name': ['Alice', 'Bob', 'Charlie'],
        'age': [25, 30, 35],
        'city': ['NYC', 'LA', 'SF']
    })

    # Write to CSV
    csv_path = '../examples/sample_data.csv'
    df.to_csv(csv_path)
    print(f"\nDataFrame written to {csv_path}")

    # Read from CSV
    loaded_df = DataFrame.read_csv(csv_path)
    print(f"DataFrame loaded from CSV:")
    print(f"  Shape: {loaded_df.shape}")
    print(f"  Columns: {loaded_df.columns}")
    for i in range(len(loaded_df)):
        row = loaded_df.iloc[i]
        print(f"    {row}")


def example_9_parquet_io():
    """Example 9: Parquet I/O."""
    print("\n" + "=" * 50)
    print("Example 9: Parquet I/O")
    print("=" * 50)

    try:
        import pyarrow
    except ImportError:
        print("\nSkipping Parquet example (pyarrow not installed)")
        print("Install with: pip install pyarrow")
        return

    df = DataFrame({
        'name': ['Alice', 'Bob', 'Charlie'],
        'age': [25, 30, 35],
        'salary': [70000, 80000, 90000]
    })

    # Write to Parquet
    parquet_path = '../examples/sample_data.parquet'
    df.to_parquet(parquet_path)
    print(f"\nDataFrame written to {parquet_path}")

    # Read from Parquet
    loaded_df = DataFrame.read_parquet(parquet_path)
    print(f"DataFrame loaded from Parquet:")
    print(f"  Shape: {loaded_df.shape}")
    print(f"  Columns: {loaded_df.columns}")
    for i in range(len(loaded_df)):
        row = loaded_df.iloc[i]
        print(f"    {row}")


def main():
    """Run all examples."""
    print("\n" + "=" * 50)
    print("SIMPLE DATAFRAME LIBRARY - USAGE EXAMPLES")
    print("=" * 50)

    example_1_basic_creation()
    example_2_selection_filtering()
    example_3_sorting_manipulation()
    example_4_groupby()
    example_5_join()
    example_6_concat()
    example_7_series()
    example_8_csv_io()
    example_9_parquet_io()

    print("\n" + "=" * 50)
    print("All examples completed!")
    print("=" * 50)


if __name__ == '__main__':
    main()
