# MPS Explorer Performance Optimization Roadmap

**Priority:** Implement bottleneck fixes for DBSCAN clustering  
**Timeline:** Phased implementation over 3 releases  
**Expected Impact:** 2-10x performance improvement for large datasets

---

## Phase 1: DBSCAN Parameter Optimization (HIGH PRIORITY)

### Objective
Implement adaptive epsilon estimation and parameter tuning to reduce computational load on DBSCAN.

### Implementation Details

#### 1.1 Adaptive Epsilon Estimation

**Method:** K-Nearest Neighbor Distance Plot

```python
from sklearn.neighbors import NearestNeighbors
import numpy as np

def estimate_optimal_eps(data, k=5, percentile=90):
    """
    Estimate optimal epsilon for DBSCAN using KNN distance method.
    
    Parameters
    ----------
    data : np.ndarray (n_samples, n_features)
        Input data points
    k : int
        Number of nearest neighbors (typically 4-5)
    percentile : float
        Percentile of distances to use as eps estimate
        
    Returns
    -------
    float
        Estimated epsilon value
    """
    nbrs = NearestNeighbors(n_neighbors=k).fit(data)
    distances, indices = nbrs.kneighbors(data)
    
    # k-distance graph: sort distances to k-th neighbor
    distances = np.sort(distances[:, k-1], axis=0)
    
    # Estimate eps as percentile of sorted distances
    eps_estimate = np.percentile(distances, percentile)
    
    return eps_estimate
```

**Integration Point:** `MPS_explorer.py` - `cluster()` method

```python
# In cluster() method, replace:
# self.eps = float(self.ui.lineEdit_eps.text())

# With:
try:
    eps_input = self.ui.lineEdit_eps.text()
    if eps_input.lower() == 'auto':
        eps_auto = estimate_optimal_eps(roi_points)
        self.eps = eps_auto
        self.logger.info(f"Auto-estimated eps={self.eps:.3f}")
    else:
        self.eps = float(eps_input)
except ValueError:
    self.logger.error("Invalid eps value")
```

**Expected Impact:**
- Reduces manual parameter tuning
- Improves clustering success rate
- Estimated 20-30% fewer failed clusterings

**Files to Modify:**
- `MPS_explorer.py` - `cluster()` method (line 1124)
- `tools/utils.py` - Add utility function for eps estimation

**Testing Required:**
- Verify with real datasets
- Compare auto-estimated vs. manual parameters
- Test edge cases (sparse/dense regions)

---

#### 1.2 Min_Samples Adaptive Selection

**Method:** Automatic scaling based on data density

```python
def estimate_min_samples(n_points, dimensionality=2):
    """
    Estimate min_samples based on dataset size and dimensionality.
    
    Rule of thumb: min_samples = 2 * dimensionality
    For large datasets: min_samples >= sqrt(n_points)
    
    Parameters
    ----------
    n_points : int
        Number of points in dataset
    dimensionality : int
        Number of dimensions (default 2 for x,y)
        
    Returns
    -------
    int
        Recommended min_samples value
    """
    # Base rule: 2x dimensionality
    base = 2 * dimensionality
    
    # Scale for large datasets
    if n_points > 10000:
        return max(base, int(np.sqrt(n_points)))
    else:
        return base
```

**Expected Impact:**
- Reduces sensitivity to min_samples parameter
- Automatic scaling for different dataset sizes
- Better default clustering behavior

---

### Phase 1 Implementation Checklist

- [ ] Add `estimate_optimal_eps()` function
- [ ] Add `estimate_min_samples()` function  
- [ ] Modify `cluster()` method to use estimates
- [ ] Add UI checkbox for "Auto-detect parameters"
- [ ] Update documentation
- [ ] Run performance tests
- [ ] Validate with real datasets

**Estimated Effort:** 2-3 hours  
**Timeline:** Week 1

---

## Phase 2: HDBSCAN Alternative for Large Datasets (HIGH PRIORITY)

### Objective
Implement HDBSCAN as alternative clustering method for datasets > 100k points.

### Background

HDBSCAN (Hierarchical DBSCAN) advantages:
- 10-50x faster than DBSCAN for large datasets
- Handles variable-density clusters better
- More robust to eps parameter selection
- Already imported in code (line 33)

### Implementation Details

#### 2.1 Create Clustering Strategy Class

```python
from abc import ABC, abstractmethod
import hdbscan
from sklearn.cluster import DBSCAN

class ClusteringStrategy(ABC):
    """Abstract base class for clustering strategies."""
    
    @abstractmethod
    def fit(self, data: np.ndarray) -> np.ndarray:
        """Fit and return cluster labels."""
        pass

class DBSCANStrategy(ClusteringStrategy):
    """Standard DBSCAN clustering."""
    
    def __init__(self, eps: float = 1.0, min_samples: int = 5):
        self.eps = eps
        self.min_samples = min_samples
    
    def fit(self, data: np.ndarray) -> np.ndarray:
        result = DBSCAN(eps=self.eps, min_samples=self.min_samples).fit(data)
        return result.labels_

class HDBSCANStrategy(ClusteringStrategy):
    """HDBSCAN clustering for large datasets."""
    
    def __init__(self, min_samples: int = 5, min_cluster_size: int = 10):
        self.min_samples = min_samples
        self.min_cluster_size = min_cluster_size
    
    def fit(self, data: np.ndarray) -> np.ndarray:
        clusterer = hdbscan.HDBSCAN(
            min_samples=self.min_samples,
            min_cluster_size=self.min_cluster_size
        )
        clusterer.fit(data)
        return clusterer.labels_
```

#### 2.2 Modify Cluster Method to Select Strategy

```python
def cluster(self, channel: int) -> None:
    """Updated cluster method with strategy selection."""
    
    # ... existing validation code ...
    
    roi_points = np.column_stack((x_roi, y_roi))
    
    # Select clustering strategy based on dataset size
    if len(roi_points) > 100000:
        strategy = HDBSCANStrategy(
            min_samples=self.minsamples,
            min_cluster_size=max(10, int(self.minsamples * 2))
        )
        self.logger.info("Using HDBSCAN for large dataset (>100k points)")
    else:
        strategy = DBSCANStrategy(
            eps=self.eps,
            min_samples=self.minsamples
        )
        self.logger.info("Using DBSCAN for dataset")
    
    # Perform clustering
    cluster_assignments = strategy.fit(roi_points)
    
    # ... rest of clustering code ...
```

**Files to Modify:**
- `MPS_explorer.py` - `cluster()` method
- Create `clustering_strategies.py` - New module for strategy classes

**Testing Required:**
- Compare DBSCAN vs. HDBSCAN results on test datasets
- Verify performance improvement
- Test parameter equivalence

---

### Phase 2 Implementation Checklist

- [ ] Create `clustering_strategies.py` with strategy classes
- [ ] Implement DBSCANStrategy
- [ ] Implement HDBSCANStrategy
- [ ] Modify `cluster()` to use strategies
- [ ] Add configuration for strategy selection
- [ ] Add logging for strategy choice
- [ ] Run performance benchmarks
- [ ] Validate on real datasets

**Estimated Effort:** 3-4 hours  
**Timeline:** Week 1-2

---

## Phase 3: Parallel Processing (MEDIUM PRIORITY)

### Objective
Enable multi-channel processing in parallel for datasets > 10k points.

### Implementation Details

#### 3.1 Parallel Channel Processing

```python
from concurrent.futures import ThreadPoolExecutor
import threading

def cluster_channel_parallel(self, channel1_data, channel2_data):
    """
    Process both channels in parallel.
    
    Parameters
    ----------
    channel1_data : tuple
        (x, y, z) for channel 1
    channel2_data : tuple
        (x, y, z) for channel 2
    """
    
    with ThreadPoolExecutor(max_workers=2) as executor:
        # Submit clustering tasks for both channels
        future1 = executor.submit(
            self._cluster_single_channel, 
            channel1_data, 
            channel=1
        )
        future2 = executor.submit(
            self._cluster_single_channel,
            channel2_data,
            channel=2
        )
        
        # Wait for both to complete
        results = [future1.result(), future2.result()]
        
        return results

def _cluster_single_channel(self, data, channel):
    """Helper method for single-channel clustering."""
    x, y, z = data
    roi_points = np.column_stack((x, y))
    # ... clustering code ...
    return cluster_assignments
```

**Expected Impact:**
- 1.5-2x speedup when clustering both channels
- Minimal overhead for single-channel clustering
- Thread-safe implementation

**Files to Modify:**
- `MPS_explorer.py` - Add parallel processing methods

**Testing Required:**
- Verify thread safety
- Compare single vs. parallel performance
- Test on multi-core systems

---

### Phase 3 Implementation Checklist

- [ ] Implement parallel clustering wrapper
- [ ] Add thread-safe state management
- [ ] Test on 2-core, 4-core, 8-core systems
- [ ] Add performance monitoring
- [ ] Document threading behavior

**Estimated Effort:** 2-3 hours  
**Timeline:** Week 2-3

---

## Phase 4: Streaming/Chunked Processing (LOW PRIORITY - FUTURE)

### Objective
Enable processing of very large datasets (>1M points) via chunked processing.

### Implementation Sketch

```python
def cluster_chunked(self, data, chunk_size=100000):
    """
    Process large dataset in chunks.
    
    Cluster each chunk independently, then merge results.
    """
    n_points = len(data)
    all_labels = np.zeros(n_points, dtype=int)
    current_label = 0
    
    for i in range(0, n_points, chunk_size):
        chunk = data[i:i+chunk_size]
        
        # Cluster chunk
        chunk_labels = self._cluster_chunk(chunk)
        
        # Merge results (relabel to avoid conflicts)
        all_labels[i:i+chunk_size] = chunk_labels + current_label
        current_label = np.max(chunk_labels) + 1
    
    return all_labels
```

**Challenges:**
- Clusters spanning chunk boundaries
- Memory management for large arrays
- Label merging complexity

**Not recommended** unless actual use case requires >1M point processing.

---

## Performance Targets

### Before Optimization

```
10k points:   ~10 ms   (target: no change)
100k points:  ~100 ms  (target: <50 ms)
1M points:    >1 s     (target: <200 ms with HDBSCAN)
```

### After Phase 1 (Parameter Optimization)

```
10k points:   ~10 ms   (no change, already optimal)
100k points:  ~90 ms   (5-10% improvement from better parameters)
1M points:    >1 s     (minimal improvement)
```

### After Phase 1+2 (With HDBSCAN)

```
10k points:   ~10 ms   (no change, use DBSCAN)
100k points:  ~30 ms   (3.3x faster with HDBSCAN)
1M points:    ~200 ms  (5x faster with HDBSCAN)
```

### After Phase 1+2+3 (With Parallel Processing)

```
Dual channel processing (Channel 1 + Channel 2):
100k points:  ~40 ms   (parallel benefit: ~1.5-2x speedup)
```

---

## Implementation Timeline

```
Week 1: Phase 1 - Parameter Optimization
  - Day 1-2: Implement adaptive eps/min_samples
  - Day 3-4: Integration and testing
  - Day 5: Validation with real data

Week 2: Phase 2 - HDBSCAN Alternative
  - Day 1-2: Implement clustering strategies
  - Day 3-4: Integration with cluster() method
  - Day 5: Performance benchmarking

Week 3: Phase 3 - Parallel Processing
  - Day 1-2: Implement parallel channel processing
  - Day 3-4: Testing and validation
  - Day 5: Documentation and cleanup
```

---

## Risk Assessment

### Low Risk

- Parameter optimization (Phase 1)
- HDBSCAN alternative (Phase 2) - already imported
- Parallel processing (Phase 3) - thread-safe patterns

### Medium Risk

- Streaming/chunking (Phase 4) - complex cluster merging
- Parameter equivalence between DBSCAN and HDBSCAN

### Mitigation Strategies

1. **Extensive Testing:** Compare results with existing datasets
2. **Backward Compatibility:** Maintain DBSCAN as default initially
3. **Gradual Rollout:** Enable HDBSCAN for >100k only, DBSCAN for smaller
4. **Performance Monitoring:** Log which strategy is used
5. **User Control:** Allow manual strategy selection in UI

---

## Success Criteria

✅ **Quantitative:**
- 3x speedup for 100k point datasets
- <100 ms processing time for 100k points
- 0% regression in clustering accuracy

✅ **Qualitative:**
- No manual parameter tuning needed (auto-detection works)
- User experience improved (faster clustering)
- Code maintainability preserved

---

## Recommended Action

**Priority 1 (Immediate):**
Implement Phase 1 - Parameter optimization

**Priority 2 (Week 2):**
Implement Phase 2 - HDBSCAN alternative

**Priority 3 (Week 3):**
Implement Phase 3 - Parallel processing

**Priority 4 (Future):**
Phase 4 - Streaming (only if needed)

---

## Related Documentation

- `PERFORMANCE_FINDINGS.md` - Detailed profiling analysis
- `profiler.py` - Performance profiling module
- `quick_profile.py` - Quick profiling script
- `run_profiling.py` - Comprehensive profiling suite

---

**Last Updated:** 2026-05-28  
**Status:** Ready for Implementation  
**Owner:** Performance Optimization Team
