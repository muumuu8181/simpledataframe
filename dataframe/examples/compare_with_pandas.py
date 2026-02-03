"""
Compare custom DataFrame with pandas - 100k rows benchmark
"""

import sys
import os
import time
import random

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from dataframe import DataFrame

# Try to import pandas
try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False
    print("Warning: pandas not installed. Install with: pip install pandas")


def generate_random_data(n_rows=100000):
    """Generate random data for benchmarking."""
    print(f"\nGenerating {n_rows:,} rows of random data...")
    start = time.time()

    random.seed(42)

    data = {
        'id': list(range(n_rows)),
        'name': [f'User_{i}' for i in range(n_rows)],
        'age': [random.randint(18, 80) for _ in range(n_rows)],
        'salary': [random.randint(30000, 200000) for _ in range(n_rows)],
        'department': [random.choice(['Sales', 'Engineering', 'HR', 'Marketing', 'Finance']) for _ in range(n_rows)],
        'score': [random.uniform(0, 100) for _ in range(n_rows)],
    }

    elapsed = time.time() - start
    print(f"  Data generation: {elapsed:.3f} seconds")

    return data, elapsed


def benchmark_custom_dataframe(data):
    """Benchmark custom DataFrame."""
    print("\n" + "=" * 60)
    print("CUSTOM DATAFRAME BENCHMARK")
    print("=" * 60)

    results = {}

    # Creation
    print("\n1. Creating DataFrame...")
    start = time.time()
    df = DataFrame(data)
    results['creation'] = time.time() - start
    print(f"   Time: {results['creation']:.3f} seconds")
    print(f"   Shape: {df.shape}")

    # Filter
    print("\n2. Filter (age > 50)...")
    start = time.time()
    result = df.filter(lambda row: row['age'] > 50)
    results['filter'] = time.time() - start
    results['filter_count'] = len(result)
    print(f"   Time: {results['filter']:.3f} seconds")
    print(f"   Result: {results['filter_count']:,} rows")

    # Sort
    print("\n3. Sort by salary...")
    start = time.time()
    result = df.sort_values('salary')
    results['sort'] = time.time() - start
    print(f"   Time: {results['sort']:.3f} seconds")

    # GroupBy
    print("\n4. GroupBy department (mean salary)...")
    start = time.time()
    result = df.groupby('department').agg({'salary': 'mean'})
    results['groupby'] = time.time() - start
    print(f"   Time: {results['groupby']:.3f} seconds")
    print(f"   Groups: {len(result)}")

    # Select columns
    print("\n5. Select columns ['name', 'age', 'salary']...")
    start = time.time()
    result = df[['name', 'age', 'salary']]
    results['select'] = time.time() - start
    print(f"   Time: {results['select']:.3f} seconds")

    # Head
    print("\n6. Get head(100)...")
    start = time.time()
    result = df.head(100)
    results['head'] = time.time() - start
    print(f"   Time: {results['head']:.3f} seconds")

    return results


def benchmark_pandas(data):
    """Benchmark pandas DataFrame."""
    if not HAS_PANDAS:
        return None

    print("\n" + "=" * 60)
    print("PANDAS DATAFRAME BENCHMARK")
    print("=" * 60)

    results = {}

    # Creation
    print("\n1. Creating DataFrame...")
    start = time.time()
    df = pd.DataFrame(data)
    results['creation'] = time.time() - start
    print(f"   Time: {results['creation']:.3f} seconds")
    print(f"   Shape: {df.shape}")

    # Filter
    print("\n2. Filter (age > 50)...")
    start = time.time()
    result = df[df['age'] > 50]
    results['filter'] = time.time() - start
    results['filter_count'] = len(result)
    print(f"   Time: {results['filter']:.3f} seconds")
    print(f"   Result: {results['filter_count']:,} rows")

    # Sort
    print("\n3. Sort by salary...")
    start = time.time()
    result = df.sort_values('salary')
    results['sort'] = time.time() - start
    print(f"   Time: {results['sort']:.3f} seconds")

    # GroupBy
    print("\n4. GroupBy department (mean salary)...")
    start = time.time()
    result = df.groupby('department', as_index=False).agg({'salary': 'mean'})
    results['groupby'] = time.time() - start
    print(f"   Time: {results['groupby']:.3f} seconds")
    print(f"   Groups: {len(result)}")

    # Select columns
    print("\n5. Select columns ['name', 'age', 'salary']...")
    start = time.time()
    result = df[['name', 'age', 'salary']]
    results['select'] = time.time() - start
    print(f"   Time: {results['select']:.3f} seconds")

    # Head
    print("\n6. Get head(100)...")
    start = time.time()
    result = df.head(100)
    results['head'] = time.time() - start
    print(f"   Time: {results['head']:.3f} seconds")

    return results


def print_comparison(custom_results, pandas_results):
    """Print comparison between custom DataFrame and pandas."""
    print("\n" + "=" * 60)
    print("COMPARISON SUMMARY")
    print("=" * 60)

    if not pandas_results:
        print("\nPandas not available for comparison.")
        return

    operations = ['creation', 'filter', 'sort', 'groupby', 'select', 'head']
    labels = {
        'creation': 'DataFrame作成',
        'filter': f'フィルタ (age > 50) - {custom_results['filter_count']:,}行',
        'sort': 'ソート (salary)',
        'groupby': 'GroupBy (department)',
        'select': '列選択',
        'head': 'head(100)'
    }

    print(f"\n{'操作':<30} {'自作DataFrame':<15} {'pandas':<15} {'比率的':<10}")
    print("-" * 70)

    for op in operations:
        custom_time = custom_results[op]
        pandas_time = pandas_results[op]

        # Calculate ratio (custom / pandas)
        if pandas_time > 0:
            ratio = custom_time / pandas_time
            ratio_str = f"{ratio:.1f}x"
        else:
            ratio_str = "N/A"

        # Highlight faster one
        if custom_time < pandas_time:
            winner = "自作"
            winner_time = f"{custom_time:.3f}s"
            loser_time = f"{pandas_time:.3f}s"
        else:
            winner = "pandas"
            winner_time = f"{pandas_time:.3f}s"
            loser_time = f"{custom_time:.3f}s"

        print(f"{labels[op]:<30} {custom_time:<15.3f}s {pandas_time:<15.3f}s {ratio_str:<10}")

    print("-" * 70)

    # Overall summary
    custom_total = sum(custom_results[op] for op in operations)
    pandas_total = sum(pandas_results[op] for op in operations)
    overall_ratio = custom_total / pandas_total

    print(f"\n総合計:")
    print(f"  自作DataFrame: {custom_total:.3f}秒")
    print(f"  pandas:        {pandas_total:.3f}秒")
    print(f"  比:           {overall_ratio:.1f}x")

    if overall_ratio < 1.0:
        print(f"\n  結果: 自作DataFrameがpandasの {1/overall_ratio:.1f}倍速い！")
    else:
        print(f"\n  結果: pandasが自作DataFrameの {overall_ratio:.1f}倍速い")

    print("=" * 70)


def main():
    """Run comparison benchmark."""
    print("\n" + "=" * 60)
    print("DATAFRAME COMPARISON - CUSTOM vs PANDAS (100,000 rows)")
    print("=" * 60)

    # Generate data (shared)
    data, _ = generate_random_data(100000)

    # Benchmark custom DataFrame
    custom_results = benchmark_custom_dataframe(data)

    # Benchmark pandas
    pandas_results = benchmark_pandas(data)

    # Print comparison
    print_comparison(custom_results, pandas_results)


if __name__ == '__main__':
    main()
