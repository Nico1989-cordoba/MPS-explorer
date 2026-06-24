# Polygon ROI - Quick Start Guide

## Quick Overview

You can now use **polygonal ROIs** to manually trace the outline of axons instead of being limited to circles or squares!

---

## How to Use

### Step 1: Load Data & Create Scatter Plot
```
1. Click "Browse" to load your HDF5/CSV file
2. Click "Scatter" to visualize the data
```

### Step 2: Select Polygon ROI
```
In the ROI section, select: [Circular] [Square] [Polygon] ← Click this
```

### Step 3: Intelligent Polygon Appears
```
An intelligent polygon appears automatically!
- If you have ConvexHull data: 10-20 vertices wrapping dense region
- Otherwise: 12-vertex circle as fallback
```

### Step 4: Edit the Polygon
You now have several options:

#### Drag Vertices
- Click and drag any vertex to move it
- Shape adjusts in real-time

#### Add Vertices
- Right-click on any polygon edge
- New vertex appears at that location
- Drag to adjust position

#### Remove Vertices  
- Ctrl + Click on a vertex to delete it
- Must keep at least 3 vertices

#### Move Entire Polygon
- Drag from the center
- All vertices move together

### Step 5: Fine-Tune
- Keep editing until polygon outlines your axon
- You can have 50-100+ vertices if needed
- As complex as you want!

### Step 6: Set Z-Range (Optional)
```
Enter Z-min and Z-max if you want to filter by depth
```

### Step 7: Cluster!
```
Click "Cluster on Ch1" or "Cluster on Ch2"
- Ray-casting filters points inside polygon (<1 second, even with 100 vertices)
- Rest of clustering proceeds normally
- Results saved same as before
```

---

## Examples

### Example 1: Simple Axon Outline
```
1. Load SMLM data
2. Select [Polygon] ROI
3. Polygon appears (ConvexHull initialized)
4. Drag a few vertices to match axon shape
5. Click Cluster
```
**Result**: Clusters for that specific axon, no noise from neighboring structures

### Example 2: Complex L-Shaped Axon
```
1. Start with intelligent polygon
2. Add vertices on the "corner" area
3. Drag those vertices to create the L-shape
4. Fine-tune other vertices
5. Click Cluster
```
**Result**: Accurate clustering of complex axon geometry

### Example 3: Multiple Fine Adjustments
```
1. Rough polygon outline (initial ConvexHull)
2. Add vertices in high-curvature regions
3. Drag each vertex to match actual boundary
4. Final polygon has 50-80 vertices, very accurate
5. Click Cluster
```
**Result**: Highly accurate clustering with smooth curved boundaries

---

## Key Features

| Feature | Benefit |
|---------|---------|
| **Smart Init** | ConvexHull starts you with good approximation |
| **Hybrid Editing** | Drag existing vertices or add/remove new ones |
| **Any Complexity** | From 3 vertices (triangle) to 100+ for very detailed traces |
| **Real-Time** | See changes as you edit |
| **Fast Filtering** | <1 second even with 100 vertices and 1M points |
| **Z-Range** | Still works with polygon ROI |
| **Same Clustering** | Uses DBSCAN/HDBSCAN same as before |

---

## Performance

| Dataset Size | Polygon Vertices | Filter Time | Status |
|---|---|---|---|
| 1M points | 20 vertices | ~200ms | Instant |
| 1M points | 100 vertices | ~500ms | < 1 second |
| 100K points | 50 vertices | ~50ms | Very fast |

All timing logged to help you understand the process.

---

## Tips & Tricks

### Tip 1: Start Rough, Refine Later
- Let ConvexHull do the initial work
- Then manually adjust vertices as needed
- Faster than drawing from scratch

### Tip 2: Add Vertices in Curved Regions
- If your axon has curves, add more vertices there
- Keeps polygons accurate without overcomplicating straight sections

### Tip 3: Use Z-Range Too
- Combine polygon XY outline with Z-range filtering
- Triple-filters your data: polygon + z-range + optional channel2

### Tip 4: Experiment Fearlessly
- Edit, cluster, see results
- Edit more, cluster again
- Changes are instant, no big performance hit

---

## Troubleshooting

### Problem: Polygon looks strange at startup
**Solution**: ConvexHull initialization might be picking wrong region. Just edit it! Drag vertices to match your axon.

### Problem: Polygon is a circle, not my axon shape
**Solution**: If ConvexHull wasn't available, fallback is 12-vertex circle. Add vertices and drag them to match your axon boundary.

### Problem: Polygon ROI not appearing
**Solution**: Make sure you:
1. ✅ Loaded data (scatter plot shows points)
2. ✅ Selected [Polygon] radio button
3. ✅ Clicked "Scatter" to create the ROI

### Problem: Clustering seems slow with polygon
**Solution**: Likely caused by many vertices (100+) or large dataset (1M+ points). Both are normal - process can take up to 1 second. Check debug log for timing.

### Problem: Points are still wrong after editing polygon
**Solution**: 
1. Make sure polygon is completely around your axon
2. Check Z-range isn't filtering out your points
3. Try with [Circular] ROI to verify data quality

---

## What's Different from Circle/Square?

| Feature | Circle | Square | Polygon |
|---|---|---|---|
| Shape | Fixed circle | Fixed square | Any shape you draw |
| Complexity | Simple | Simple | As complex as needed |
| Flexibility | Resize/move | Rotate/move | Drag each vertex individually |
| Use Case | Quick overview | Roughly square axons | **Precise irregular boundaries** ✨ |

---

## Advanced Features (Optional)

### Spline Smoothing
If you want smooth curves (instead of straight edges between vertices):
- This happens automatically behind the scenes
- Creates realistic axon outlines
- Optional feature in config

### Multiple Edit Sessions
- Edit polygon → Cluster → See results
- Edit again → Cluster → See new results
- Changes don't affect your original data

---

## Questions?

Check `POLYGONAL_ROI_IMPLEMENTATION.md` for technical details:
- Algorithm explanation (ray-casting)
- Performance benchmarks
- Configuration options
- Troubleshooting guide

---

**Enjoy precise axon clustering with polygon ROIs!** 🎯
