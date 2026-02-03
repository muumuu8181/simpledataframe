"""
Benchmark script for DataFrame operations with 100k rows
"""

import sys
import os
import time
import random

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from dataframe import DataFrame


def generate_random_data(n_rows=100000):
    """Generate random data for benchmarking."""
    print(f"Generating {n_rows:,} rows of random data...")
    start = time.time()

    random.seed(42)  # For reproducibility

    data = {
        'id': list(range(n_rows)),
        'name': [f'User_{i}' for i in range(n_rows)],
        'age': [random.randint(18, 80) for _ in range(n_rows)],
        'salary': [random.randint(30000, 200000) for _ in range(n_rows)],
        'department': [random.choice(['Sales', 'Engineering', 'HR', 'Marketing', 'Finance']) for _ in range(n_rows)],
        'score': [random.uniform(0, 100) for _ in range(n_rows)],
    }

    elapsed = time.time() - start
    print(f"  Data generation: {elapsed:.3f} seconds\n")

    return data, elapsed


def benchmark_dataframe_creation(data):
    """Benchmark DataFrame creation."""
    print("1. Creating DataFrame...")
    start = time.time()
    df = DataFrame(data)
    elapsed = time.time() - start
    print(f"   Time: {elapsed:.3f} seconds")
    print(f"   Shape: {df.shape}")
    print(f"   Columns: {df.columns}\n")
    return df, elapsed


def benchmark_csv_write(df, path):
    """Benchmark CSV write."""
    print(f"2. Writing to CSV: {path}")
    start = time.time()
    df.to_csv(path)
    elapsed = time.time() - start

    # Get file size
    file_size = os.path.getsize(path) / (1024 * 1024)  # MB
    print(f"   Time: {elapsed:.3f} seconds")
    print(f"   File size: {file_size:.2f} MB\n")
    return elapsed, file_size


def benchmark_parquet_write(df, path):
    """Benchmark Parquet write."""
    print(f"3. Writing to Parquet: {path}")
    start = time.time()
    df.to_parquet(path)
    elapsed = time.time() - start

    # Get file size
    file_size = os.path.getsize(path) / (1024 * 1024)  # MB
    print(f"   Time: {elapsed:.3f} seconds")
    print(f"   File size: {file_size:.2f} MB\n")
    return elapsed, file_size


def benchmark_csv_read(path):
    """Benchmark CSV read."""
    print(f"4. Reading from CSV: {path}")
    start = time.time()
    df = DataFrame.read_csv(path)
    elapsed = time.time() - start
    print(f"   Time: {elapsed:.3f} seconds")
    print(f"   Shape: {df.shape}\n")
    return df, elapsed


def benchmark_parquet_read(path):
    """Benchmark Parquet read."""
    print(f"5. Reading from Parquet: {path}")
    start = time.time()
    df = DataFrame.read_parquet(path)
    elapsed = time.time() - start
    print(f"   Time: {elapsed:.3f} seconds")
    print(f"   Shape: {df.shape}\n")
    return df, elapsed


def benchmark_operations(df):
    """Benchmark some DataFrame operations."""
    print("6. Testing DataFrame operations...")

    # Filter
    start = time.time()
    result = df.filter(lambda row: row['age'] > 50)
    filter_time = time.time() - start
    print(f"   Filter (age > 50): {filter_time:.3f} seconds ({len(result):,} rows)")

    # Sort
    start = time.time()
    result = df.sort_values('salary')
    sort_time = time.time() - start
    print(f"   Sort by salary: {sort_time:.3f} seconds")

    # GroupBy
    start = time.time()
    result = df.groupby('department').agg({'salary': 'mean'})
    groupby_time = time.time() - start
    print(f"   GroupBy department: {groupby_time:.3f} seconds\n")

    return filter_time, sort_time, groupby_time


def print_summary(times, sizes):
    """Print summary of benchmark results."""
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"{'Operation':<30} {'Time (s)':<12} {'Notes'}")
    print("-" * 60)

    print(f"{'Data generation':<30} {times['data_gen']:<12.3f}")
    print(f"{'DataFrame creation':<30} {times['df_creation']:<12.3f}")
    print(f"{'CSV write':<30} {times['csv_write']:<12.3f} {sizes['csv']:.2f} MB")
    print(f"{'Parquet write':<30} {times['parquet_write']:<12.3f} {sizes['parquet']:.2f} MB")
    print(f"{'CSV read':<30} {times['csv_read']:<12.3f}")
    print(f"{'Parquet read':<30} {times['parquet_read']:<12.3f}")
    print(f"{'Filter (age > 50)':<30} {times['filter']:<12.3f}")
    print(f"{'Sort by salary':<30} {times['sort']:<12.3f}")
    print(f"{'GroupBy department':<30} {times['groupby']:<12.3f}")

    print("-" * 60)

    # Comparison
    csv_parquet_ratio = sizes['csv'] / sizes['parquet'] if sizes['parquet'] > 0 else 0
    print(f"CSV file size: {sizes['csv']:.2f} MB")
    print(f"Parquet file size: {sizes['parquet']:.2f} MB")
    print(f"Compression ratio: {csv_parquet_ratio:.1f}x (CSV / Parquet)")

    write_speedup = times['csv_write'] / times['parquet_write'] if times['parquet_write'] > 0 else 0
    read_speedup = times['csv_read'] / times['parquet_read'] if times['parquet_read'] > 0 else 0

    print(f"\nWrite speedup (Parquet vs CSV): {write_speedup:.1f}x")
    print(f"Read speedup (Parquet vs CSV): {read_speedup:.1f}x")
    print("=" * 60)


def main():
    """Run benchmark."""
    print("\n" + "=" * 60)
    print("DATAFRAME BENCHMARK - 100,000 ROWS")
    print("=" * 60 + "\n")

    times = {}
    sizes = {}

    # Generate data
    data, times['data_gen'] = generate_random_data(100000)

    # Create DataFrame
    df, times['df_creation'] = benchmark_dataframe_creation(data)

    # Paths
    csv_path = 'benchmark_data.csv'
    parquet_path = 'benchmark_data.parquet'

    # Write benchmarks
    times['csv_write'], sizes['csv'] = benchmark_csv_write(df, csv_path)
    times['parquet_write'], sizes['parquet'] = benchmark_parquet_write(df, parquet_path)

    # Read benchmarks
    _, times['csv_read'] = benchmark_csv_read(csv_path)
    _, times['parquet_read'] = benchmark_parquet_read(parquet_path)

    # Operations benchmark
    times['filter'], times['sort'], times['groupby'] = benchmark_operations(df)

    # Print summary
    print_summary(times, sizes)

    # Cleanup
    if os.path.exists(csv_path):
        os.remove(csv_path)
    if os.path.exists(parquet_path):
        os.remove(parquet_path)
    print("\nCleanup: Temporary files removed.")


if __name__ == '__main__':
    main()
