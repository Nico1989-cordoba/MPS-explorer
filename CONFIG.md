# MPS Explorer — Configuration System

## Overview

A centralized configuration system for tuning MPS Explorer parameters without modifying code. All application constants (ROI sizing, histogram binning, point sizes, colors) are defined in external `config.yaml` or `config.json` files.

**Status**: ✅ **Fully implemented and integrated**

---

## Quick Start

### Default Configuration

The default `config.yaml` file contains all MPS Explorer constants:

```yaml
roi:
  diameter_scale_factor: 1.3
  extent_divisor: 10
  color_rgb: [255, 0, 0]
  z_order: 10

histogram:
  default_knn_bins: 30
  bins_2d: 400
  bins_z: 500
  max_lateral_distance_nm: 800

visualization:
  point_size_noise: 3
  point_size_good_cluster: 5
  point_size_centroid: 10
```

### Modifying Constants

Simply edit `config.yaml` to change any parameter. No code modifications needed:

```yaml
# Increase 2D histogram resolution
histogram:
  bins_2d: 600  # Changed from 400

# Make ROI circles larger
roi:
  diameter_scale_factor: 1.5  # Changed from 1.3
```

Restart the application for changes to take effect.

---

## Configuration Sections

### ROI (Region of Interest) Configuration

```yaml
roi:
  # Scale factor for circular ROI diameter
  # Range: > 0 (typically 1.0 - 2.0)
  # Effect: Larger values = bigger selection circles
  diameter_scale_factor: 1.3

  # Divisor for initial ROI extent
  # Range: > 0 (typically 5 - 20)
  # Effect: extent = image_dimension / extent_divisor
  # Larger values = smaller initial ROI
  extent_divisor: 10

  # RGB color for ROI circle outline
  # Format: [Red, Green, Blue] where each is 0-255
  # Examples: [255, 0, 0] = Red, [0, 255, 0] = Green, [0, 0, 255] = Blue
  color_rgb: [255, 0, 0]

  # Z-order (stacking) for ROI rendering
  # Range: 0-100 (higher = on top)
  # Default: 10 (renders above most graphics)
  z_order: 10
```

### Histogram Configuration

```yaml
histogram:
  # Default bins for K-NN distance histogram
  # Range: 5 - 100
  # Effect: More bins = finer resolution but slower computation
  default_knn_bins: 30

  # Bins for 2D XY histogram (cluster centroid positions)
  # Range: 50 - 1000
  # Effect: More bins = higher spatial resolution
  # Note: High values (>500) may be slow on older computers
  bins_2d: 400

  # Bins for Z-axis histogram (axial distribution)
  # Range: 50 - 1000
  # Effect: More bins = finer axial resolution
  bins_z: 500

  # Maximum lateral distance (nm) for KNN histogram
  # Range: 100 - 2000
  # Effect: Sets x-axis range of KNN distance plot
  # Larger values = more spread out histogram
  max_lateral_distance_nm: 800
```

### Visualization Configuration

```yaml
visualization:
  # Point size for noise points (DBSCAN label = -1)
  # Range: 1 - 20
  # Typical: 2-5 (smaller to de-emphasize)
  point_size_noise: 3

  # Point size for good cluster members
  # Range: 1 - 20
  # Typical: 4-8 (medium prominence)
  point_size_good_cluster: 5

  # Point size for cluster centroids
  # Range: 1 - 30
  # Typical: 8-15 (largest for visibility)
  # Note: Set larger than point_size_good_cluster for visual hierarchy
  point_size_centroid: 10
```

---

## Environment Variable Overrides

For advanced users and CI/CD integration, override any config value via environment variables:

### Syntax

Environment variables use the pattern: `MPS_<SECTION>_<KEY>=value`

Examples:

```bash
# ROI configuration
export MPS_ROI_DIAMETER_SCALE_FACTOR=1.5
export MPS_ROI_COLOR_RGB="[0, 255, 0]"

# Histogram configuration
export MPS_HISTOGRAM_BINS_2D=600
export MPS_HISTOGRAM_MAX_LATERAL_DISTANCE_NM=1000

# Visualization configuration
export MPS_VISUALIZATION_POINT_SIZE_CENTROID=15
```

### Windows Command Line

```cmd
:: Set environment variables in Windows
set MPS_ROI_DIAMETER_SCALE_FACTOR=1.5
set MPS_HISTOGRAM_BINS_2D=600
set MPS_VISUALIZATION_POINT_SIZE_CENTROID=15

:: Run MPS Explorer
python MPS_explorer.py
```

### Python Script

```python
import os

# Set environment variables before importing MPS_explorer
os.environ['MPS_ROI_DIAMETER_SCALE_FACTOR'] = '1.5'
os.environ['MPS_HISTOGRAM_BINS_2D'] = '600'

# Now import (loads config with overrides)
from MPS_explorer import MPS_explorer
```

### Priority Order

Configuration values are loaded with this priority (highest to lowest):

1. **Environment variables** (highest priority)
   - Example: `MPS_ROI_DIAMETER_SCALE_FACTOR=1.5`
2. **Configuration file** (YAML or JSON)
   - Example: `config.yaml` or `config.json`
3. **Built-in defaults** (lowest priority)
   - Hardcoded fallback values in config_loader.py

This means environment variables override the config file, which overrides defaults.

---

## Configuration File Formats

### YAML Format (Recommended)

**File**: `config.yaml`

```yaml
# Human-friendly format with comments
roi:
  diameter_scale_factor: 1.3  # Scale ROI circles

histogram:
  bins_2d: 400  # 2D histogram resolution
```

**Advantages**:
- Easy to read and edit
- Supports comments
- Structured hierarchy

**Limitations**:
- Requires PyYAML: `pip install pyyaml`
- If not installed, falls back to simple YAML parser

### JSON Format (Fallback)

**File**: `config.json`

```json
{
  "roi": {
    "diameter_scale_factor": 1.3,
    "extent_divisor": 10,
    "color_rgb": [255, 0, 0],
    "z_order": 10
  },
  "histogram": {
    "default_knn_bins": 30,
    "bins_2d": 400,
    "bins_z": 500,
    "max_lateral_distance_nm": 800
  },
  "visualization": {
    "point_size_noise": 3,
    "point_size_good_cluster": 5,
    "point_size_centroid": 10
  }
}
```

**Advantages**:
- Works without external dependencies
- Standard format (widely supported)

**Limitations**:
- No comments allowed
- More verbose than YAML

### File Selection

The loader automatically detects and loads from:

1. `config.yaml` (if exists)
2. `config.json` (if YAML doesn't exist)
3. Built-in defaults (if neither file exists)

---

## Typical Tuning Scenarios

### Scenario 1: Higher Resolution 2D Histogram

**Goal**: Get finer spatial detail in cluster centroid positions

```yaml
histogram:
  bins_2d: 800  # From 400 (doubles resolution)
```

**Trade-off**: Slightly slower rendering (negligible on modern computers)

### Scenario 2: Adjust ROI Circle Size

**Goal**: Make ROI selection circles larger for better visibility

```yaml
roi:
  diameter_scale_factor: 1.7  # From 1.3
```

**Effect**: ROI circles now 30% larger

### Scenario 3: Emphasize Cluster Centroids

**Goal**: Make cluster centroids much more visually prominent

```yaml
visualization:
  point_size_centroid: 20     # From 10
  point_size_good_cluster: 5  # Keep same
  point_size_noise: 3         # Keep same
```

**Effect**: Clear visual hierarchy: centroids >> good clusters >> noise

### Scenario 4: Focus on Specific Distance Range

**Goal**: View KNN distances in narrower range (local clustering)

```yaml
histogram:
  max_lateral_distance_nm: 300  # From 800
  default_knn_bins: 50          # Increase for detail
```

**Effect**: KNN histogram compresses to 300 nm, showing local structure

### Scenario 5: Different Color Scheme

**Goal**: Use green ROI circles instead of red

```yaml
roi:
  color_rgb: [0, 255, 0]  # Green (was [255, 0, 0])
```

---

## Advanced Configuration

### Batch Configuration for Multiple Users

Create different config files for different use cases:

```
configs/
├── config_default.yaml     # Standard settings
├── config_hires.yaml       # High-resolution mode
├── config_fast.yaml        # Fast rendering mode
└── config_presentation.yaml # Presentation mode
```

Load specific config via environment variable:

```python
import os
os.environ['MPS_CONFIG_FILE'] = 'configs/config_hires.yaml'
from MPS_explorer import MPS_explorer
```

Or pass to config loader:

```python
from config_loader import load_config

config = load_config('configs/config_hires.yaml')
```

### CI/CD Integration

In automated testing/deployment:

```bash
#!/bin/bash
# Set test configuration
export MPS_HISTOGRAM_BINS_2D=200    # Lower for fast testing
export MPS_VISUALIZATION_POINT_SIZE_CENTROID=8

# Run application
python MPS_explorer.py --test-mode
```

---

## Validation and Error Handling

### Type Validation

The configuration system validates types automatically:

```yaml
# ✅ Valid
histogram:
  bins_2d: 400        # Integer
  max_lateral_distance_nm: 800  # Integer

# ✅ Valid (converted to int)
histogram:
  bins_2d: 400.0      # Float (converted to int)

# ❌ Invalid (not a number)
histogram:
  bins_2d: "high"     # String (ERROR)
```

### Range Validation

Built-in validation ensures sensible values:

```python
# From config_loader.py validate_config()
if roi_config.get('diameter_scale_factor', 0) <= 0:
    raise ValueError("roi.diameter_scale_factor must be > 0")

if hist_config.get('bins_2d', 0) <= 0:
    raise ValueError("histogram.bins_2d must be > 0")
```

### Missing Values

Missing values use built-in defaults:

```python
# If config.yaml is missing 'bins_2d'
# This default is used:
bins_2d = config.get('bins_2d', 400)  # Default: 400
```

---

## Troubleshooting

### Configuration Not Loading

**Problem**: Changes in `config.yaml` don't take effect

**Solution**:
1. Save the file (Ctrl+S)
2. Fully restart the application
3. Check for syntax errors in YAML:
   ```bash
   python -c "from config_loader import load_config; print(load_config())"
   ```

### Invalid Configuration Error

**Problem**: Application crashes with configuration error

**Check**:
1. Verify YAML syntax (proper indentation, colons)
2. Ensure values are correct types:
   - Numbers: `400` (not `"400"`)
   - Lists: `[255, 0, 0]` (not `255, 0, 0`)
   - Strings: `"value"` (if needed)

**Example Valid YAML**:
```yaml
roi:
  diameter_scale_factor: 1.3  # ✅ Correct (number)
  color_rgb: [255, 0, 0]       # ✅ Correct (list)
```

**Example Invalid YAML**:
```yaml
roi:
  diameter_scale_factor: "1.3"  # ❌ String instead of number
  color_rgb: 255, 0, 0          # ❌ Missing brackets
```

### File Permission Error

**Problem**: Cannot save configuration file

**Solution**:
1. Check file is not read-only:
   ```bash
   chmod 644 config.yaml  # Linux/macOS
   attrib -r config.yaml  # Windows
   ```
2. Ensure directory is writable
3. Run editor as administrator (Windows)

---

## Configuration Statistics

### Current Configuration

| Section | Parameter | Value | Type | Range |
|---------|-----------|-------|------|-------|
| ROI | diameter_scale_factor | 1.3 | float | > 0 |
| ROI | extent_divisor | 10 | int | > 0 |
| ROI | color_rgb | [255, 0, 0] | list[int] | 0-255 |
| ROI | z_order | 10 | int | 0-100 |
| Histogram | default_knn_bins | 30 | int | 5-100 |
| Histogram | bins_2d | 400 | int | 50-1000 |
| Histogram | bins_z | 500 | int | 50-1000 |
| Histogram | max_lateral_distance_nm | 800 | int | 100-2000 |
| Visualization | point_size_noise | 3 | int | 1-20 |
| Visualization | point_size_good_cluster | 5 | int | 1-20 |
| Visualization | point_size_centroid | 10 | int | 1-30 |

**Total Configurable Parameters**: 11

---

## Best Practices

### 1. **Keep Defaults When Unsure**

The default values are tuned for typical SMLM data. Only change if you have a specific reason.

### 2. **Document Your Changes**

Add comments in `config.yaml` explaining custom values:

```yaml
roi:
  # Custom for high-density clusters (2x typical density)
  diameter_scale_factor: 1.0  # Reduced from default 1.3
  extent_divisor: 5           # Reduced from default 10 (smaller initial ROI)
```

### 3. **Use Version Control**

Track configuration changes:

```bash
git add config.yaml
git commit -m "Increase histogram resolution for publication-quality plots"
```

### 4. **Test Before Production**

For important configurations, test on sample data first:

```python
# Test configuration
from config_loader import load_config, validate_config

config = load_config()
assert validate_config(config), "Configuration validation failed"
```

### 5. **Backup Default Configuration**

Keep a copy of the original:

```bash
cp config.yaml config.yaml.default
```

---

## Migration Guide

### From Hardcoded Constants (Old Code)

**Before** (hardcoded in Python):
```python
ROI_DIAMETER_SCALE_FACTOR = 1.3  # In MPS_explorer.py
HISTOGRAM_2D_BINS = 400           # In MPS_explorer.py
```

**After** (externalized in config):
```python
# In config.yaml
roi:
  diameter_scale_factor: 1.3

histogram:
  bins_2d: 400
```

**Advantage**: Change values without editing Python code!

---

## Summary

The configuration system provides:

✅ **Externalized constants** (no code edits needed)  
✅ **YAML + JSON support** (flexible formats)  
✅ **Environment variable overrides** (CI/CD friendly)  
✅ **Type validation** (automatic error detection)  
✅ **Built-in defaults** (safe fallbacks)  
✅ **Comprehensive logging** (tracks which config is loaded)  
✅ **Easy tuning** (change values in seconds)  

---

**Last Updated**: 2026-05-28  
**Status**: ✅ Production-ready  
**Configuration File**: `config.yaml`  
**Fallback Format**: `config.json`  
**Validation**: Automatic type & range checking  
**Override Method**: Environment variables (MPS_* prefix)
