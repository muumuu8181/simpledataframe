"""
Benchmark filter optimization - 1M rows
"""

import sys
import os
import time
import random

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from dataframe import DataFrame
import pandas as pd

print("=" * 60)
print("FILTER OPTIMIZATION BENCHMARK - 1,000,000 rows")
print("=" * 60)

# Generate data
print("\nGenerating 1,000,000 rows...")
random.seed(42)
data = {
    'id': list(range(1000000)),
    'age': [random.randint(18, 80) for _ in range(1000000)],
    'salary': [random.randint(30000, 200000) for _ in range(1000000)],
}

print("Creating DataFrames...")
df_custom = DataFrame(data)
df_pandas = pd.DataFrame(data)

# Test 1: Old method (lambda)
print("\n" + "-" * 60)
print("Test 1: OLD METHOD (lambda)")
print("-" * 60)

start = time.time()
result_old = df_custom.filter(lambda row: row['age'] > 50)
time_old = time.time() - start
print(f"Custom (lambda): {time_old:.3f} seconds ({len(result_old):,} rows)")

# Test 2: New method (boolean indexing)
print("\n" + "-" * 60)
print("Test 2: NEW METHOD (boolean indexing)")
print("-" * 60)

start = time.time()
result_new = df_custom[df_custom['age'] > 50]
time_new = time.time() - start
print(f"Custom (boolean): {time_new:.3f} seconds ({len(result_new):,} rows)")

# Test 3: Pandas
print("\n" + "-" * 60)
print("Test 3: PANDAS")
print("-" * 60)

start = time.time()
result_pandas = df_pandas[df_pandas['age'] > 50]
time_pandas = time.time() - start
print(f"Pandas: {time_pandas:.3f} seconds ({len(result_pandas):,} rows)")

# Comparison
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"{'Method':<25} {'Time':<12} {'vs Pandas':<12} {'vs Old':<12}")
print("-" * 60)
print(f"{'Old (lambda)':<25} {time_old:<12.3f} {time_old/time_pandas:<12.1f}x 1.0x (baseline)")
print(f"{'New (boolean)':<25} {time_new:<12.3f} {time_new/time_pandas:<12.1f}x {time_old/time_new:<12.1f}x")
print(f"{'Pandas':<25} {time_pandas:<12.3f} 1.0x (baseline) {time_old/time_pandas:<12.1f}x")
print("-" * 60)

speedup = time_old / time_new
print(f"\nSpeedup: {speedup:.1f}x faster than old method!")
print(f"Gap to pandas: {time_new / time_pandas:.1f}x")

if speedup > 10:
    print("\n🎉 Great improvement!")
elif speedup > 5:
    print("\n✅ Good improvement!")
else:
    print("\n⚠️  Need more optimization")
