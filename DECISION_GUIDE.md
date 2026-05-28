# MPS Explorer - Decision Guide

**"What do I do when...?" Quick Reference**

Use this guide to find answers to specific situations and challenges.

---

## 🚀 Getting Started

### "I just installed MPS Explorer, what should I do?"

👉 **See QUICK_START.md** for 5-minute introduction

**Path:**
1. Load sample data (File → Open)
2. Select a small region first
3. Click "Cluster"
4. Review results
5. Try different regions to learn

---

### "I have a new image file, how do I start?"

👉 **Step-by-step:**

1. **Click "Load Data"** (or File → Open)
2. **Select your image file**
   - Supported: .h5, .hdf5, .tif, .png, .jpg
   - Check file exists and is readable
3. **Wait for image to display** (~2-5 seconds)
4. **Select ROI** by drawing rectangle
5. **Click "Cluster"**
6. **Review results**

**If image won't load:**
- Check file format
- Try converting to different format
- See Troubleshooting: "Image Won't Load"

---

## 🎯 During Analysis

### "Should I use automatic or manual parameters?"

👉 **Use automatic (recommended)**

```
Epsilon = "auto"      ← Default, recommended
Min Samples = "auto"  ← Default, recommended
```

**When to use automatic:**
- ✅ First time users (99% of cases)
- ✅ Want consistent results
- ✅ Want 30% performance boost from caching
- ✅ Don't know optimal parameters

**When to use manual:**
- ❌ Only if auto gives bad results
- ❌ Specific clustering behavior needed
- ❌ Testing parameter sensitivity
- ❌ Advanced analysis

---

### "I'm getting results I don't like. What should I do?"

👉 **Decision tree:**

```
START: Clustering complete

→ Q1: Is cluster count 0 or 1?
   ├─ YES → Problem: Underclustering
   │        Solution A: Try different ROI
   │        Solution B: Use auto parameters
   │        Solution C: See "No Clusters" section
   └─ NO → Continue

→ Q2: Is cluster count >100?
   ├─ YES → Problem: Overclustering
   │        Solution A: Use auto parameters
   │        Solution B: Increase epsilon
   │        Solution C: See "Too Many Clusters" section
   └─ NO → Continue

→ Q3: Is noise percentage >40%?
   ├─ YES → Problem: Too much noise
   │        Solution A: Check data quality
   │        Solution B: Try different ROI
   │        Solution C: See "Noisy Results" section
   └─ NO → DONE - Results look good!
```

---

### "How do I choose between 'Cluster' and 'Cluster Both'?"

👉 **Quick decision:**

| Situation | Use |
|-----------|-----|
| Single channel analysis | "Cluster" |
| Want fastest results | "Cluster Both" |
| Need both channels | "Cluster Both" |
| Testing/debugging | "Cluster" |
| Production/batch | "Cluster Both" |

**Default recommendation:** Use "Cluster Both" - 1.5-2.5x faster with same quality

---

### "I'm analyzing multiple similar ROIs. How do I go faster?"

👉 **Use parameter caching:**

1. **First ROI:**
   - Select ROI
   - Click "Cluster"
   - Parameters automatically estimated and saved

2. **Second similar ROI:**
   - Select new ROI (same size/location range)
   - Click "Cluster"
   - Automatic: Cached parameters reused
   - Result: ~30% faster

3. **Continue for more ROIs:**
   - Same process for each ROI
   - Cache speeds up subsequent clustering
   - Parameters automatically applied

**No configuration needed** - caching works automatically!

---

### "Should I enable GPU acceleration?"

👉 **Decision:**

**Enable GPU if:**
- ✅ Have NVIDIA GPU
- ✅ Analyzing large datasets (>100k points)
- ✅ Processing many ROIs
- ✅ Want fastest possible results
- ✅ Processing takes >1 second

**Skip GPU if:**
- ❌ No NVIDIA GPU
- ❌ Working with small datasets (<10k)
- ❌ Don't need extreme speed
- ❌ Prefer simple CPU-only solution

**Installation (if interested):**
```bash
pip install cuml pynvml
python MPS_explorer.py
```

Automatic detection - if GPU available, used transparently!

---

## 💾 Saving & Exporting

### "How do I save my clustering results?"

👉 **After successful clustering:**

1. **Click "Save Results"** (or File → Save)
2. **Choose location**
3. **Enter filename**
4. **Choose format** (CSV or JSON)
5. **File saved** ✅

**What gets saved:**
- Cluster assignments for each point
- Parameter values used
- Processing time
- Data statistics

---

### "What format should I use - CSV or JSON?"

👉 **Quick comparison:**

| Format | Best For | Size | Readability |
|--------|----------|------|-------------|
| CSV | Importing to Excel/MATLAB | Small | Human-readable |
| JSON | Complete metadata | Medium | Developer-friendly |

**Recommendation:**
- Scientific analysis: **CSV** (compatible with tools)
- Data archival: **JSON** (preserves all details)
- Quick inspection: **CSV** (readable in Excel)

---

## ⚠️ When Results Are Unexpected

### "Clustering found 0 clusters (all noise)"

👉 **Diagnosis → Solution:**

**Step 1: Verify data**
- Is ROI selected properly?
- Does ROI contain visible structure?
- Is image loaded correctly?

**Step 2: Try automatic parameters**
- Uncheck "Auto" checkbox, then recheck
- Click "Cluster" again
- Usually fixes the issue

**Step 3: Try different ROI**
- Select larger or more structured region
- Select area with more points
- Avoid pure noise regions

**Step 4: Check data quality**
- Is image too dark/bright?
- Is preprocessing correct?
- Is file format valid?

**If still failing:**
- See full troubleshooting in USER_GUIDE.md

---

### "Clustering found 1000+ clusters (fragmented)"

👉 **Diagnosis → Solution:**

**Step 1: Use automatic parameters**
- Checkbox "Auto" should be enabled
- Click "Cluster" again
- Automatic selection usually fixes this

**Step 2: Increase epsilon manually**
- Try values: 1.0, 2.0, 5.0, 10.0
- Start with smallest, increase if needed
- Click "Cluster"

**Step 3: Increase min_samples manually**
- Try values: 10, 20, 30
- Prevents small clusters from forming

**Step 4: Try different ROI**
- Too-dense data causes fragmentation
- Try less dense region
- Or accept many small clusters

---

### "Results are inconsistent (different each time)"

👉 **Diagnosis → Solution:**

**Common causes:**
- Different parameters each time
- Different ROI selections
- Manual parameters not saved

**Solution:**
1. **Use automatic parameters** (checkbox enabled)
2. **Same ROI** produces same results
3. **Same parameters** = identical clustering
4. **System remembers** previous parameters

**Verify consistency:**
1. Cluster same ROI twice
2. Results should be identical
3. If not identical → Check parameters

---

### "Application is slow (takes >2 seconds)"

👉 **Diagnosis → Solution:**

**Check 1: Data size**
- Very large ROI (millions of points)?
- Solution: Select smaller ROI

**Check 2: GPU available?**
- No GPU? Try: `pip install cuml`
- Enable GPU for 10-100x speedup

**Check 3: Background processes**
- Too many applications running?
- Close unnecessary programs
- Check Task Manager

**Check 4: Use "Cluster Both"**
- Parallel processing is faster
- Use for both channels

**Quick fix (fastest):**
1. Install GPU support: `pip install cuml`
2. Restart application
3. System auto-detects GPU
4. 10-100x speedup for large data

---

## 🐛 Common Problems

### "I keep getting 'File not supported' error"

👉 **Check file format:**

**Supported formats:**
- ✅ `.h5`, `.hdf5` (HDF5 files)
- ✅ `.tif`, `.tiff` (TIFF images)
- ✅ `.png` (PNG images)
- ✅ `.jpg`, `.jpeg` (JPEG images)

**If your file isn't supported:**
1. Convert to TIFF or PNG
2. Use file converter (e.g., ImageMagick)
3. Or convert programmatically:
   ```python
   from PIL import Image
   img = Image.open('original.bmp')
   img.save('converted.png')
   ```

---

### "ImportError: No module named 'X'"

👉 **Missing dependency:**

**Solution:**
```bash
pip install -r requirements.txt
```

**If specific module missing:**
```bash
pip install hdbscan numpy scikit-learn PyQt5
```

---

### "GPU not detected even though I have NVIDIA GPU"

👉 **Troubleshooting steps:**

1. **Check GPU exists:**
   ```bash
   nvidia-smi
   ```
   Should list your GPU

2. **Install RAPIDS:**
   ```bash
   pip install cuml pynvml
   ```

3. **Check detection:**
   ```bash
   python -c "from tools.gpu_clustering import create_gpu_clustering_manager; m = create_gpu_clustering_manager(); print(m.gpu_info)"
   ```

4. **If still not detected:**
   - Update NVIDIA drivers
   - Install CUDA Toolkit (11.0+)
   - Check GPU compatibility
   - Application falls back to CPU automatically

**Important:** GPU acceleration is optional. CPU works fine without it!

---

## 📊 Understanding Results

### "What does 'Cluster Count = 8' mean?"

👉 **Interpretation:**

- **Clusters = 8** means 8 distinct groups found
- **Quality depends on data:**
  - 0-3 clusters: Very consolidated
  - 3-20 clusters: Well-structured (typical good result)
  - 20-100 clusters: Dense or fragmented
  - 100+ clusters: Very dense or overclustering

**What to do:**
- 0-1 clusters: Try different ROI or parameters
- 3-20: Usually good (typical range)
- 20-100: May be acceptable or need adjustment
- 100+: Likely overclustering, adjust parameters

---

### "What percentage of noise is acceptable?"

👉 **Noise interpretation:**

| Noise % | Assessment | Action |
|---------|-----------|--------|
| 0-10% | Excellent | Keep results |
| 10-20% | Good | Acceptable |
| 20-40% | Fair | Consider adjusting |
| 40%+ | Poor | Likely wrong data/parameters |

**If noise too high:**
- Try automatic parameters
- Select different ROI
- Check data preprocessing
- Review data quality

---

## 🎓 Learning Path

### "I'm new to clustering, what should I learn?"

👉 **Recommended learning order:**

1. **Read QUICK_START.md** (5 min)
   - Basic workflow
   - First clustering

2. **Try application** (10 min)
   - Load sample data
   - Select ROIs
   - Run clustering
   - Review results

3. **Read Clustering Explained** in USER_GUIDE.md (5 min)
   - What clustering is
   - How it works
   - Quality indicators

4. **Read Parameters section** in USER_GUIDE.md (10 min)
   - Understand epsilon
   - Understand min_samples
   - Automatic vs. manual

5. **Experiment** (20+ min)
   - Try different ROIs
   - Try different parameters
   - See how things change

6. **Read full USER_GUIDE.md** (if interested)
   - Advanced features
   - GPU acceleration
   - Batch processing

---

### "How long until I'm comfortable using MPS Explorer?"

👉 **Timeline:**

| Milestone | Time | What You Can Do |
|-----------|------|-----------------|
| After 5 min | QUICK_START | Basic clustering |
| After 30 min | + experimenting | Understand parameters |
| After 1 hour | + USER_GUIDE | Use all features confidently |
| After 1 day | + practice | Train others |

---

## 🎯 Tips by Use Case

### Research Paper Analysis

**Steps:**
1. Use automatic parameters (reproducibility)
2. Document all parameters used
3. Save results and parameters
4. Keep metadata for supplementary
5. Enable GPU if analyzing large datasets

**Key settings:**
- ✅ Automatic parameters
- ✅ Same ROI selection criteria
- ✅ Document everything

---

### Large-Scale Batch Processing

**Steps:**
1. Analyze first batch with auto parameters
2. Parameters cached automatically
3. Subsequent batches reuse parameters
4. 30% faster due to parameter caching
5. Parallel processing ("Cluster Both")

**Key settings:**
- ✅ Automatic parameters
- ✅ Use "Cluster Both"
- ✅ Batch similar datasets together
- ✅ Enable GPU if available

---

### Method Development / Testing

**Steps:**
1. Try automatic parameters first
2. Manually adjust to test sensitivity
3. Document parameter effects
4. Test on multiple ROIs
5. Validate results against gold standard

**Key settings:**
- ❌ Don't use automatic (you're testing parameters)
- ✅ Manual adjustments encouraged
- ✅ Try wide parameter ranges
- ✅ Document all experiments

---

## 📞 Getting Help

### "Where do I find help?"

👉 **Help resources:**

| Question | Resource |
|----------|----------|
| How to start? | QUICK_START.md |
| Full documentation? | USER_GUIDE.md |
| Troubleshooting? | USER_GUIDE.md → Troubleshooting |
| Decision making? | This guide (DECISION_GUIDE.md) |
| Advanced features? | USER_GUIDE.md → Advanced Features |
| Found a bug? | GitHub Issues |
| Need GPU help? | GPU_ACCELERATION_GUIDE.md |

### "How do I report a problem?"

👉 **Report on GitHub:**

1. Go to: github.com/luhalac/MPS-explorer/issues
2. Click "New Issue"
3. Describe problem clearly
4. Include error message
5. Provide reproduction steps
6. Include system info (Python version, OS, etc.)

---

## ✅ Checklist: Before You Ask for Help

Before reporting an issue, verify:

- [ ] Installed Python 3.8+
- [ ] Ran `pip install -r requirements.txt`
- [ ] Tried with automatic parameters
- [ ] Checked file format (HDF5, TIFF, PNG, JPG)
- [ ] Verified ROI selection
- [ ] Reviewed error message
- [ ] Checked USER_GUIDE.md troubleshooting
- [ ] Tried restarting application
- [ ] Tried different image/ROI

If all checked and still failing → Ready to report!

---

## 🎯 Summary

**Choose your path:**

```
├─ "I have 5 minutes" → QUICK_START.md
├─ "I have 30 minutes" → QUICK_START.md + try application
├─ "I have 1 hour" → USER_GUIDE.md
├─ "I need help deciding" → DECISION_GUIDE.md (this file)
├─ "Something's broken" → Troubleshooting sections
├─ "I want GPU acceleration" → GPU_ACCELERATION_GUIDE.md
└─ "I still need help" → GitHub Issues
```

---

**MPS Explorer - Decision Guide v1.0**  
**Find answers to "What do I do when...?" questions**
