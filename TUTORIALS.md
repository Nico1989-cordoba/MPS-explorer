# MPS Explorer - Tutorials & Workflows

**Learn by doing with step-by-step tutorials**

---

## 📖 Table of Contents

1. [Tutorial 1: Basic Workflow](#tutorial-1-basic-workflow)
2. [Tutorial 2: Understanding Parameters](#tutorial-2-understanding-parameters)
3. [Tutorial 3: Batch Processing](#tutorial-3-batch-processing)
4. [Tutorial 4: GPU Acceleration](#tutorial-4-gpu-acceleration)
5. [Tutorial 5: Research Analysis](#tutorial-5-research-analysis)
6. [Tutorial 6: Method Development](#tutorial-6-method-development)

---

## Tutorial 1: Basic Workflow

**Time:** 5-10 minutes  
**Goal:** Complete your first clustering analysis  
**Difficulty:** Beginner  

### Step 1: Start the Application

```bash
python MPS_explorer.py
```

**Expected:** Application window opens with empty display area

### Step 2: Load an Image

1. Click **"Load Data"** button (or File → Open)
2. Select an image file (recommended: sample data)
3. Wait for image to load (2-5 seconds)

**What to look for:**
- Image displays in the viewer
- Image appears in gray or color
- Status bar shows "Image loaded"

### Step 3: Select a Region of Interest (ROI)

1. **Click and drag** on the image to draw a rectangle
2. **Release** to finalize selection
3. **ROI coordinates** appear in the control panel

**Tips:**
- Select regions with visible structure
- Avoid completely empty areas
- Start with medium-sized regions (500-5000 points)

### Step 4: Cluster the Data

1. **Verify** epsilon is set to "auto" and min_samples is "auto"
2. **Click** "Cluster" button
3. **Wait** for clustering (usually <500ms)

**What to observe:**
- Status message shows processing time
- Cluster count appears in results panel
- Noise percentage displayed
- Quality suggestion provided

### Step 5: Review the Results

**Check these numbers:**

- **Cluster Count:** 0-20 is typical for good data
- **Noise Points:** <20% is good
- **Processing Time:** <500ms is normal
- **Quality Suggestion:** "Good clustering" is ideal

**What different results mean:**

```
Good Results:
- Cluster count: 5-20
- Noise: 0-20%
- Quality: "Good clustering"
→ Analysis complete, can proceed

Poor Results:
- Cluster count: 0 or 100+
- Noise: >40%
- Quality: "Consider adjusting epsilon"
→ Try different ROI or parameters
```

### Step 6: Save Your Results (Optional)

1. Click **"Save Results"**
2. Choose location and filename
3. Select format (CSV or JSON)
4. Click "Save"

**What was saved:**
- Cluster assignments for each point
- Parameters used (epsilon, min_samples)
- Processing time
- Data statistics

### Step 7: Analyze Another Region

1. Select different ROI on the image
2. Click "Cluster"
3. Observe that results are **30% faster** (parameter caching!)

**Why faster?**
- First clustering estimated parameters
- Second similar ROI reuses cached parameters
- No re-estimation needed = faster processing

---

## Tutorial 2: Understanding Parameters

**Time:** 15-20 minutes  
**Goal:** Understand how epsilon and min_samples affect clustering  
**Difficulty:** Beginner to Intermediate

### Background: What Are Parameters?

```
Epsilon (ε): Maximum distance for points in same cluster
- Smaller = smaller clusters
- Larger = larger clusters

Min Samples: Minimum points needed for a cluster
- Smaller = more lenient
- Larger = more strict
```

### Experiment 1: Effect of Epsilon

1. Load an image and select an ROI
2. **Uncheck "Auto"** to use manual parameters
3. Run clustering with different epsilon values

**Test these epsilon values:**

```
First run:  epsilon = 0.5   (very small)
Second run: epsilon = 1.0   (small)
Third run:  epsilon = 2.0   (medium)
Fourth run: epsilon = 5.0   (large)
Fifth run:  epsilon = 10.0  (very large)
```

**Observe:**
- Cluster count at each epsilon
- How clusters grow with larger epsilon
- Noise percentage at each level

**Record results:**

| Epsilon | Clusters | Noise % | Observation |
|---------|----------|---------|-------------|
| 0.5 | ___ | ___ | |
| 1.0 | ___ | ___ | |
| 2.0 | ___ | ___ | |
| 5.0 | ___ | ___ | |
| 10.0 | ___ | ___ | |

**Expected pattern:**
- Larger epsilon → Fewer clusters
- Larger epsilon → Less noise

### Experiment 2: Effect of Min_Samples

1. Use same ROI as before
2. Keep epsilon constant (e.g., 1.0)
3. Vary min_samples

**Test these min_samples values:**

```
First run:  min_samples = 3   (lenient)
Second run: min_samples = 5   (default)
Third run:  min_samples = 10  (strict)
Fourth run: min_samples = 20  (very strict)
```

**Observe:**
- How cluster count changes
- Effect on noise percentage
- Threshold effects

**Expected pattern:**
- Larger min_samples → Fewer/smaller clusters
- Larger min_samples → More noise (stricter criteria)

### Experiment 3: Finding Optimal Parameters

1. Load image with clear structure
2. **Enable "Auto"** parameters
3. Click "Cluster"
4. **Note the automatic values:**

```
Automatic epsilon: ___
Automatic min_samples: ___
Cluster count: ___
Noise %: ___
Quality: ___
```

5. Compare with your manual experiments
6. **Automatic parameters usually better than manual!**

### Learning Outcomes

After this tutorial, you should understand:
- ✅ What epsilon and min_samples do
- ✅ How they affect clustering results
- ✅ Why automatic selection works well
- ✅ When manual adjustment might help

---

## Tutorial 3: Batch Processing

**Time:** 20-30 minutes  
**Goal:** Efficiently analyze multiple similar regions  
**Difficulty:** Intermediate

### Scenario: Analyzing Multiple ROIs

Imagine you have a single image with 10 similar regions you need to analyze.

**Without optimization:** 10 × 50ms estimation = 500ms wasted  
**With caching:** 1 × 50ms + 9 × 0ms = Only 50ms estimation total! ✅

### Step 1: Prepare Your Image

1. Load image with multiple similar regions
2. Plan which regions you'll analyze
3. Estimate region sizes (should be similar)

### Step 2: First ROI (Cache Building)

1. Select **ROI #1**
2. Ensure "Auto" parameters enabled
3. Click "Cluster"
4. **System automatically:**
   - Analyzes data distribution
   - Estimates epsilon and min_samples
   - **Saves parameters to cache** ✅

**Time:** ~50ms for parameter estimation

### Step 3: Similar ROIs (Cache Benefits)

1. Select **ROI #2** (similar size/location as ROI #1)
2. Click "Cluster"
3. **System automatically:**
   - Detects similar data signature
   - **Retrieves cached parameters** ✅
   - Skips estimation (instant)

**Time:** ~0ms for parameter estimation (instant!)

### Step 4: Repeat for Remaining ROIs

1. Select **ROI #3** through **ROI #10**
2. Each one automatically uses cached parameters
3. Processing is **30% faster** per ROI

**Full workflow timing:**

```
ROI #1: 50ms estimation + 50ms clustering = 100ms total
ROI #2: 0ms estimation + 50ms clustering = 50ms total
ROI #3: 0ms estimation + 50ms clustering = 50ms total
...
ROI #10: 0ms estimation + 50ms clustering = 50ms total
─────────────────────────────────────────────────────
Total: ~500ms (vs 1000ms without caching)
Saved: ~500ms! (50% improvement!)
```

### Step 5: Document Your Analysis

For each ROI, record:

| ROI # | Clusters | Noise % | Time | Notes |
|-------|----------|---------|------|-------|
| 1 | ___ | ___ | ~100ms | Cache built |
| 2 | ___ | ___ | ~50ms | Cached |
| 3 | ___ | ___ | ~50ms | Cached |
| ... | ... | ... | ... | ... |
| 10 | ___ | ___ | ~50ms | Cached |

### Step 6: Use "Cluster Both"

For maximum speed:

1. If analyzing dual-channel data
2. Click **"Cluster Both"** instead of "Cluster"
3. Both channels processed in parallel
4. **1.5-2.5x faster than sequential**

**Total workflow with "Cluster Both":**

```
ROI #1: 100ms
ROI #2-10: 35ms each × 9 = 315ms
─────────────────────────────
Total: ~415ms (vs 1000ms without optimization)
Speedup: 2.4x! 🚀
```

### Benefits of Batch Processing

✅ **Parameter caching** - 30% faster per ROI  
✅ **Parallel "Cluster Both"** - 1.5-2.5x faster  
✅ **Consistent parameters** - Same ROI size = same settings  
✅ **Time saved** - Process 10 ROIs faster than 2  
✅ **Reproducible** - Document parameters used  

---

## Tutorial 4: GPU Acceleration

**Time:** 15-30 minutes  
**Goal:** Enable and use GPU acceleration  
**Difficulty:** Intermediate

### Prerequisites

- NVIDIA GPU (not applicable for others)
- CUDA Toolkit 11.0+
- Administrative access (for installation)

### Step 1: Check if GPU Available

**Before installation, check if GPU exists:**

```bash
nvidia-smi
```

**Expected output:**
```
Wed May 28 10:30:00 2026
+------+ GPU Name | ... | Memory | ... |
| NVIDIA GeForce RTX 3080 | ...
...
```

**If command not found:** GPU drivers not installed. Visit NVIDIA website.

### Step 2: Install GPU Support

```bash
pip install cuml pynvml
```

**Expected output:**
```
Successfully installed cuml-24.04
Successfully installed pynvml-...
```

### Step 3: Verify GPU Detection

Start application and watch startup messages:

```bash
python MPS_explorer.py
```

**Look for message:**
```
GPU detected: NVIDIA GeForce RTX 3080 (8000MB)
RAPIDS HDBSCAN available for 10-100x speedup.
```

**Or verify programmatically:**

```bash
python -c "from tools.gpu_clustering import create_gpu_clustering_manager; m = create_gpu_clustering_manager(); print(m.gpu_info)"
```

### Step 4: Compare CPU vs GPU Performance

**Load a large dataset (>100k points) and cluster:**

1. **CPU clustering:**
   ```
   Cluster using CPU (no GPU)
   Time: 2000ms (typical for 100k points)
   ```

2. **GPU clustering:**
   ```
   Cluster using GPU
   Time: 100ms (with GPU)
   Speedup: 20x! 🚀
   ```

### Step 5: Use GPU with Application

GPU is **automatically detected and used** - no configuration needed!

Just use the application normally:
1. Load image
2. Select ROI
3. Click "Cluster"
4. **GPU automatically used for large datasets**

**Performance by dataset size:**

| Size | CPU | GPU | Speedup |
|------|-----|-----|---------|
| 10k | 100ms | 50ms | 2x |
| 50k | 800ms | 50ms | 16x |
| 100k | 2000ms | 100ms | 20x |
| 500k | 15000ms | 500ms | 30x |

### Step 6: GPU + Batch Processing

**Maximum optimization:**

1. Load image with large ROIs
2. Batch analyze with "Cluster Both"
3. GPU accelerates each clustering
4. Parameter caching saves estimation

**Combined speedup:**

```
ROI #1: 100ms (estimation) + 100ms (GPU) = 200ms
ROI #2: 0ms (cached) + 100ms (GPU) = 100ms
ROI #3: 0ms (cached) + 100ms (GPU) = 100ms
...
ROI #10: 0ms (cached) + 100ms (GPU) = 100ms
─────────────────────────────────────────────
Total: ~1100ms vs 10000ms without optimization
Speedup: 9x overall! 🚀🚀🚀
```

### Troubleshooting GPU

**GPU not detected:**

1. Run `nvidia-smi` to verify GPU exists
2. Install/update NVIDIA drivers
3. Install CUDA Toolkit
4. Re-install cuml: `pip install --upgrade cuml`

**GPU slower than CPU (shouldn't happen but if it does):**

1. Data too small (<10k points) - GPU overhead dominates
2. GPU busy with other tasks - close other applications
3. Use CPU for small data, GPU for large data (automatic)

---

## Tutorial 5: Research Analysis

**Time:** 30-45 minutes  
**Goal:** Workflow for reproducible research publications  
**Difficulty:** Advanced

### Context: Reproducible Science

For publications, you need:
- ✅ Documented parameters
- ✅ Reproducible results
- ✅ Quality metrics
- ✅ Complete workflow description

### Step 1: Document Your Setup

Create a lab notebook entry:

```
Experiment: Analysis of sample XYZ
Date: 2026-05-28
Analyzer: [Your name]
File: sample_xyz_scan.h5

Settings Used:
- Epsilon: auto (automatic)
- Min Samples: auto (automatic)
- GPU: [enabled/disabled]
- Analysis time: [recorded by system]
```

### Step 2: Run Analysis with Automatic Parameters

1. Load your research image
2. **Ensure "Auto" parameters enabled** (recommended)
3. Select first ROI carefully (document coordinates)
4. Click "Cluster"
5. **Document results:**

```
ROI #1:
- Location: (x1, x2, y1, y2) = (100, 200, 150, 250)
- Size: 50×100 pixels
- Clusters: 12
- Noise: 8.5%
- Processing time: 125ms
- Automatic epsilon: 0.75
- Automatic min_samples: 5
- Quality: "Good clustering"
```

### Step 3: Quality Control

Before accepting results:

```
✅ Cluster count 5-20?
✅ Noise <20%?
✅ Quality assessment positive?
✅ Results match expected patterns?
✅ No visual artifacts in image?

If NO to any: Document issue and try different ROI
```

### Step 4: Save Results for Publication

**Save in both formats:**

1. **CSV format** - For supplementary data
2. **JSON format** - For archival and reproducibility

```bash
# Save CSV
File: sample_xyz_roi1_clusters.csv
Contents: Cluster assignments for each point

# Save JSON
File: sample_xyz_roi1_metadata.json
Contents: Parameters, statistics, timestamps
```

### Step 5: Document for Supplementary Materials

Create supplementary document:

```markdown
## Clustering Analysis Methods

### Parameters
- Epsilon: [automatic / value]
- Min Samples: [automatic / value]
- Algorithm: [DBSCAN / HDBSCAN]
- Processing: [CPU / GPU]

### Quality Metrics
- Cluster count: [number]
- Noise percentage: [percent]
- Processing time: [milliseconds]

### Reproducibility
- Software: MPS Explorer v1.0
- Python: [version]
- System: [Windows/Mac/Linux]
- GPU: [yes/no]

### Results
[Table of cluster assignments for each ROI]
[Visualization of clustering results]
```

### Step 6: Verify Reproducibility

**On different computer or time:**

1. Load same image
2. Select **same ROI coordinates**
3. Click "Cluster"
4. **Compare results:**

```
Expected: Identical clustering (same parameters, same results)
Actual: ___

Match: [YES / NO]
```

**If NO:** Investigate differences. Check:
- ✅ Same automatic parameters?
- ✅ Same data loaded?
- ✅ Same software version?

---

## Tutorial 6: Method Development

**Time:** 45-60 minutes  
**Goal:** Test parameter sensitivity and develop optimal workflow  
**Difficulty:** Advanced

### Scenario: Finding Best Settings for New Data Type

You have a new image type (different staining, microscope, etc.) and need to find optimal clustering parameters.

### Step 1: Create Test Dataset

1. Prepare representative image
2. Select 5-10 ROIs of consistent size
3. **Document expectations:**

```
Expected clustering characteristics:
- Typical cluster size: [small/medium/large]
- Expected cluster count: [estimate]
- Data density: [sparse/medium/dense]
```

### Step 2: Test Automatic Parameters

Run all ROIs with automatic parameters:

```python
# Test with auto parameters
for roi_number in range(1, 6):
    select_roi(roi_number)
    enable_auto_parameters()
    click_cluster()
    record_results(roi_number)
```

**Record results table:**

| ROI | Auto Eps | Auto MinSamp | Clusters | Noise % | Time | Quality |
|-----|----------|--------------|----------|---------|------|---------|
| 1 | ___ | ___ | ___ | ___ | ___ | ___ |
| 2 | ___ | ___ | ___ | ___ | ___ | ___ |
| 3 | ___ | ___ | ___ | ___ | ___ | ___ |
| 4 | ___ | ___ | ___ | ___ | ___ | ___ |
| 5 | ___ | ___ | ___ | ___ | ___ | ___ |

**Average automatic values:**
```
Average epsilon: ___
Average min_samples: ___
Consistency: ___
```

### Step 3: Test Parameter Variations

Take the automatic epsilon and test variations:

```
Base epsilon from auto: X
Test epsilon values: X/2, X*0.75, X, X*1.25, X*2
```

**For each variation, run on ROI #1:**

```python
test_epsilon_values = [x/2, x*0.75, x, x*1.25, x*2]
for eps in test_epsilon_values:
    select_roi(1)
    set_epsilon(eps)
    click_cluster()
    record_results()
```

**Record sensitivity table:**

| Epsilon | Cluster Count | Noise % | Quality | Note |
|---------|---------------|---------|---------|------|
| X/2 | ___ | ___ | ___ | Too small? |
| X*0.75 | ___ | ___ | ___ | Small |
| X | ___ | ___ | ___ | Optimal |
| X*1.25 | ___ | ___ | ___ | Large |
| X*2 | ___ | ___ | ___ | Too large? |

### Step 4: Repeat for Min Samples

Similar test for min_samples:

```
Base min_samples: Y
Test values: Y/2, Y*0.75, Y, Y*1.25, Y*2
```

### Step 5: Identify Optimal Parameters

Based on your tests:

```
Optimal epsilon range: [___, ___]
Optimal min_samples range: [___, ___]

Recommended single values:
- Epsilon: ___
- Min Samples: ___

Rationale:
- Consistency across ROIs: [good/fair/poor]
- Quality assessment: [positive/neutral/negative]
- Cluster count stability: [high/medium/low]
```

### Step 6: Cross-Validation

Test final parameters on new ROIs (not used in development):

```python
# Test on ROI #6-10 (validation set)
for roi_number in range(6, 11):
    select_roi(roi_number)
    set_epsilon(optimal_eps)
    set_min_samples(optimal_min_samples)
    click_cluster()
    record_results()
```

**Validation results:**

| ROI | Clusters | Noise % | Quality | As Expected? |
|-----|----------|---------|---------|-------------|
| 6 | ___ | ___ | ___ | [YES/NO] |
| 7 | ___ | ___ | ___ | [YES/NO] |
| 8 | ___ | ___ | ___ | [YES/NO] |
| 9 | ___ | ___ | ___ | [YES/NO] |
| 10 | ___ | ___ | ___ | [YES/NO] |

### Step 7: Document Final Method

```markdown
## Optimized Clustering Method for [Data Type]

### Optimal Parameters
- Epsilon: [value]
- Min Samples: [value]

### Performance Characteristics
- Average clusters: [number]
- Average noise: [percent]
- Processing time: [milliseconds]
- Consistency: [quality assessment]

### Applicability
- Works well for: [size, structure, density characteristics]
- Potential issues: [documented edge cases]
- Recommended validation: [number of test cases]

### Reproducibility Notes
- Software version: [version]
- Tested on: [data types/characteristics]
- Parameter stability: [assessment]
```

---

## Summary

These 6 tutorials progress from basic (5 min) to advanced (60 min):

1. **Basic Workflow** (5 min) - Get started
2. **Understanding Parameters** (15 min) - Learn the concepts
3. **Batch Processing** (20 min) - Efficient analysis
4. **GPU Acceleration** (15 min) - Speed boost
5. **Research Analysis** (30 min) - Reproducible science
6. **Method Development** (45 min) - Optimize for your data

**Suggested progression:**
- Complete Tutorial 1 immediately
- Do Tutorials 2-3 this week
- Try Tutorials 4-6 as needed for your work

---

**Ready to get started? See QUICK_START.md for immediate next steps!**
