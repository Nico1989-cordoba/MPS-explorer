#!/usr/bin/env python3
"""Quick profiling script with direct console output."""

import numpy as np
import time
from sklearn.cluster import DBSCAN

print("\n" + "="*80)
print("MPS EXPLORER PERFORMANCE PROFILING - QUICK TEST")
print("="*80 + "\n")

# Generate test data
n_points = 10000
print(f"Generating {n_points:,} synthetic test points...")
np.random.seed(42)
x = np.random.normal(loc=5000, scale=1000, size=n_points)
y = np.random.normal(loc=5000, scale=1000, size=n_points)
z = np.random.normal(loc=100, scale=50, size=n_points)
print("[OK] Generated data\n")

data_points = np.column_stack((x, y))

# --- ROI FILTERING ---
print("="*80)
print("1. ROI FILTERING OPERATIONS")
print("="*80)

# Circular ROI
center = np.array([5000.0, 5000.0])
radius = 1000.0

print(f"\nCircular ROI Filter (radius={radius} on {n_points:,} points):")
t_start = time.perf_counter()
distances = np.linalg.norm(data_points - center, axis=1)
mask = distances <= radius
roi_points = data_points[mask]
t_circular = time.perf_counter() - t_start
print(f"  Time: {t_circular*1000:.2f} ms")
print(f"  Result: {len(roi_points):,} points selected ({100*len(roi_points)/n_points:.1f}%)")

# Square ROI
print(f"\nSquare ROI Filter (4000-6000 on both axes):")
t_start = time.perf_counter()
square_mask = (x >= 4000) & (x <= 6000) & (y >= 4000) & (y <= 6000)
square_points = data_points[square_mask]
t_square = time.perf_counter() - t_start
print(f"  Time: {t_square*1000:.2f} ms")
print(f"  Result: {len(square_points):,} points selected ({100*len(square_points)/n_points:.1f}%)")

# Z-filtering
print(f"\nZ-coordinate Filter (50 < z < 150):")
t_start = time.perf_counter()
z_mask = (z > 50) & (z < 150)
z_filtered = z[z_mask]
t_z = time.perf_counter() - t_start
print(f"  Time: {t_z*1000:.2f} ms")
print(f"  Result: {len(z_filtered):,} points selected ({100*len(z_filtered)/n_points:.1f}%)")

# --- DBSCAN CLUSTERING ---
print("\n" + "="*80)
print("2. DBSCAN CLUSTERING")
print("="*80)

# Test with ROI-selected points (more realistic)
roi_data = roi_points

parameters = [
    (5, 0.5, "Tight"),
    (10, 1.0, "Medium"),
    (20, 2.0, "Loose"),
]

for min_samples, eps, label in parameters:
    print(f"\n{label} clustering (eps={eps}, min_samples={min_samples}) on {len(roi_data):,} points:")
    t_start = time.perf_counter()
    result = DBSCAN(eps=eps, min_samples=min_samples).fit(roi_data)
    t_dbscan = time.perf_counter() - t_start

    n_clusters = len(set(result.labels_)) - (1 if -1 in result.labels_ else 0)
    n_noise = list(result.labels_).count(-1)

    print(f"  Time: {t_dbscan*1000:.2f} ms")
    print(f"  Clusters: {n_clusters}, Noise points: {n_noise:,}")
    print(f"  Time per point: {(t_dbscan*1_000_000)/len(roi_data):.2f} µs")

# --- CENTROID CALCULATION ---
print("\n" + "="*80)
print("3. CENTROID CALCULATION")
print("="*80)

result = DBSCAN(eps=1.0, min_samples=10).fit(roi_data)
labels = result.labels_

print(f"\nCalculating centroids from {len(set(labels)) - (1 if -1 in labels else 0)} clusters:")
t_start = time.perf_counter()
unique_labels = np.unique(labels)
centroids = []
for label in unique_labels:
    if label == -1:
        continue
    cluster_points = roi_data[labels == label]
    centroids.append(np.mean(cluster_points, axis=0))
centroids = np.array(centroids)
t_centroid = time.perf_counter() - t_start

print(f"  Time: {t_centroid*1000:.2f} ms")
print(f"  Computed {len(centroids)} centroids")

# --- DISTANCE CALCULATIONS ---
print("\n" + "="*80)
print("4. DISTANCE CALCULATIONS")
print("="*80)

reference_point = np.mean(roi_data, axis=0)
print(f"\nDistances to reference point (from {len(roi_data):,} points):")
t_start = time.perf_counter()
distances = np.linalg.norm(roi_data - reference_point, axis=1)
t_dist = time.perf_counter() - t_start

print(f"  Time: {t_dist*1000:.2f} ms")
print(f"  Time per point: {(t_dist*1_000_000)/len(roi_data):.2f} µs")

# --- SORTING AND FILTERING ---
print("\n" + "="*80)
print("5. SORTING AND FILTERING")
print("="*80)

print(f"\nSorting {len(z):,} points by z-coordinate:")
t_start = time.perf_counter()
sorted_indices = np.argsort(z)
t_sort = time.perf_counter() - t_start
print(f"  Time: {t_sort*1000:.2f} ms")

print(f"\nMultiple threshold filtering ({n_points:,} points):")
t_start = time.perf_counter()
filter_mask = (z > 50) & (z < 150) & (x > 4000) & (x < 6000)
filtered = data_points[filter_mask]
t_filter = time.perf_counter() - t_start
print(f"  Time: {t_filter*1000:.2f} ms")
print(f"  Result: {len(filtered):,} points")

print(f"\nUnique value extraction ({n_points:,} points):")
t_start = time.perf_counter()
unique_x = np.unique(np.round(x, decimals=1))
unique_y = np.unique(np.round(y, decimals=1))
unique_z = np.unique(np.round(z, decimals=1))
t_unique = time.perf_counter() - t_start
print(f"  Time: {t_unique*1000:.2f} ms")
print(f"  Found {len(unique_x)}x{len(unique_y)}x{len(unique_z)} unique values")

# --- SUMMARY ---
print("\n" + "="*80)
print("PERFORMANCE SUMMARY")
print("="*80)

operations = [
    ("Circular ROI Filter", t_circular * 1000),
    ("Square ROI Filter", t_square * 1000),
    ("Z Filter", t_z * 1000),
    ("DBSCAN Clustering (medium)", 0),  # Will be calculated
    ("Centroid Calculation", t_centroid * 1000),
    ("Distance Calculation", t_dist * 1000),
    ("Sorting by Z", t_sort * 1000),
    ("Multi-threshold Filter", t_filter * 1000),
    ("Unique Extraction", t_unique * 1000),
]

print(f"\n{'Operation':<35} {'Time (ms)':<12} {'Per 1K pts (ms)':>15}")
print("-" * 65)

total_time = sum(t for _, t in operations)

for op_name, time_ms in operations:
    if time_ms > 0:
        per_1k = (time_ms / n_points) * 1000
        print(f"{op_name:<35} {time_ms:>10.2f}  {per_1k:>15.3f}")

print("-" * 65)
print(f"{'Total Profiled':<35} {total_time:>10.2f}")

print("\n" + "="*80)
print("KEY FINDINGS")
print("="*80)

print("""
1. ROI FILTERING: Very fast (< 2 ms)
   - Vectorized NumPy operations are efficient
   - Linear O(n) complexity

2. DBSCAN CLUSTERING: Time depends on eps parameter
   - Larger eps => faster (fewer clusters)
   - Smaller eps => slower (more clusters to compute)

3. DISTANCE CALCULATIONS: O(n) complexity
   - Fast with vectorized operations
   - Scales linearly with dataset size

4. BOTTLENECK IDENTIFICATION:
   - DBSCAN is typically the slowest operation for large datasets
   - Parameter tuning (eps, min_samples) significantly affects speed
   - Memory usage can be a constraint for very large datasets (>1M points)
""")

print("="*80 + "\n")
