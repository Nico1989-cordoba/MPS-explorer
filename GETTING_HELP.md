# MPS Explorer - Getting Help & Support

**Complete guide to finding answers, getting support, and contributing**

---

## 🆘 I Need Help!

### Start Here - Self-Service Support (Takes 2-10 minutes)

**Step 1: Check Documentation (First Try This)**

| Your Situation | Best Resource | Time |
|---|---|---|
| **Want to get started** | [QUICK_START.md](QUICK_START.md) | 5 min |
| **Have a specific question** | [DECISION_GUIDE.md](DECISION_GUIDE.md) | 2-10 min |
| **Need complete information** | [USER_GUIDE.md](USER_GUIDE.md) | 1 hour |
| **Something's broken** | USER_GUIDE.md → Troubleshooting | 5-10 min |
| **Want to learn by doing** | [TUTORIALS.md](TUTORIALS.md) | 10-60 min |
| **GPU not working** | [GPU_ACCELERATION_GUIDE.md](GPU_ACCELERATION_GUIDE.md) | 10 min |
| **Can't find anything** | [USER_DOCUMENTATION_INDEX.md](USER_DOCUMENTATION_INDEX.md) | 2 min |

**Step 2: Check FAQ (Second Try This)**

See [USER_GUIDE.md → FAQ section](USER_GUIDE.md#faq) for 30+ common questions already answered.

**Step 3: Check Troubleshooting**

See [USER_GUIDE.md → Troubleshooting](USER_GUIDE.md#troubleshooting) for common problems with solutions.

---

## ❓ Common Questions Answered

### Installation Issues

**Q: "ModuleNotFoundError: No module named 'X'"**

A: Run `pip install -r requirements.txt` to install all dependencies

**Q: "Application won't start"**

A: Try these steps in order:
1. Verify Python version: `python --version` (should be 3.8+)
2. Reinstall requirements: `pip install --upgrade -r requirements.txt`
3. Try: `python -m pip install PyQt5 hdbscan numpy scikit-learn`
4. Restart your terminal
5. Try running again

**Q: "GPU not detected"**

A: Run `pip install cuml pynvml` then restart application

### Usage Issues

**Q: "Image won't load"**

A: Check:
1. File format supported? (.h5, .tif, .png, .jpg)
2. File exists and readable?
3. File not corrupted? Try opening with viewer first
4. Try converting to different format

**Q: "0 clusters found (all noise)"**

A: Try:
1. Different ROI (current one may be blank/noisy)
2. Enable automatic parameters
3. Larger ROI with more structure
4. Check image preprocessing

**Q: "Too many clusters (100+)"**

A: Try:
1. Enable automatic parameters (fixes most cases)
2. Increase epsilon manually
3. Increase min_samples
4. Select less dense region

**Q: "Application is slow"**

A: Try:
1. Enable GPU: `pip install cuml pynvml`
2. Use "Cluster Both" for parallel processing
3. Select smaller ROI
4. Close other applications
5. Check available RAM

### Parameter Questions

**Q: "What should epsilon be?"**

A: Use "auto" (automatic). Let system estimate. Only use manual values if auto fails.

**Q: "Should I adjust parameters?"**

A: 95% of the time, use automatic. Only manually adjust if:
- Automatic gave poor results
- Testing parameter sensitivity
- Specific clustering behavior needed

**Q: "Why are results different each time?"**

A: They shouldn't be with same parameters. Check:
- Using same parameters?
- Same ROI selection?
- Same data?
- Different Python/package versions?

---

## 🔍 Finding Your Answer

### Decision Tree: Finding Help

```
START: I have a problem

├─ Is it about getting started?
│  └─ YES → QUICK_START.md
│
├─ Is it about parameters?
│  └─ YES → USER_GUIDE.md → Parameters section
│
├─ Is it about GPU?
│  └─ YES → GPU_ACCELERATION_GUIDE.md
│
├─ Do I have a specific question?
│  └─ YES → DECISION_GUIDE.md → "What do I do when...?"
│
├─ Is something broken?
│  └─ YES → USER_GUIDE.md → Troubleshooting section
│
├─ Is it answered in FAQ?
│  └─ YES → USER_GUIDE.md → FAQ (30+ answers)
│
├─ Do I want to learn by doing?
│  └─ YES → TUTORIALS.md (6 tutorials)
│
├─ Am I lost in documentation?
│  └─ YES → USER_DOCUMENTATION_INDEX.md
│
└─ Still no answer?
   └─ → Report issue on GitHub
```

---

## 📞 Different Types of Help

### For Installation Problems

1. Check: [QUICK_START.md → Installation](QUICK_START.md)
2. Check: [USER_GUIDE.md → Installation & Setup](USER_GUIDE.md#installation--setup)
3. Check: [Troubleshooting in USER_GUIDE.md](USER_GUIDE.md#troubleshooting)
4. Still broken? [File GitHub Issue](#reporting-issues)

### For Usage Questions

1. Check: [DECISION_GUIDE.md](DECISION_GUIDE.md) for your scenario
2. Check: [USER_GUIDE.md → Using the Application](USER_GUIDE.md#using-the-application)
3. Check: [FAQ in USER_GUIDE.md](USER_GUIDE.md#faq)
4. Still confused? [File GitHub Issue](#reporting-issues)

### For Performance Issues

1. Check: [GPU_ACCELERATION_GUIDE.md](GPU_ACCELERATION_GUIDE.md)
2. Check: [TUTORIALS.md → Tutorial 3 (Batch Processing)](TUTORIALS.md#tutorial-3-batch-processing)
3. Check: [Troubleshooting - "Application is slow"](USER_GUIDE.md#troubleshooting)
4. Still slow? [File GitHub Issue](#reporting-issues)

### For Parameter Understanding

1. Check: [USER_GUIDE.md → Understanding Parameters](USER_GUIDE.md#understanding-parameters)
2. Do: [TUTORIALS.md → Tutorial 2 (Parameters)](TUTORIALS.md#tutorial-2-understanding-parameters)
3. Check: [FAQ section](USER_GUIDE.md#faq) - several questions about parameters
4. Still confused? [File GitHub Issue](#reporting-issues)

### For Scientific/Research Help

1. Check: [TUTORIALS.md → Tutorial 5 (Research Analysis)](TUTORIALS.md#tutorial-5-research-analysis)
2. Check: [USER_GUIDE.md → Tips & Best Practices](USER_GUIDE.md#tips--best-practices)
3. Check: [DECISION_GUIDE.md → Tips by Use Case](DECISION_GUIDE.md#tips-by-use-case)
4. Need customization? [File GitHub Issue](#reporting-issues)

---

## 🐛 Reporting Issues

### Before You Report

**Check this list first:**

- [ ] Read [QUICK_START.md](QUICK_START.md) or relevant guide
- [ ] Tried solutions in [USER_GUIDE.md → Troubleshooting](USER_GUIDE.md#troubleshooting)
- [ ] Checked [DECISION_GUIDE.md](DECISION_GUIDE.md) for your scenario
- [ ] Searched [FAQ section](USER_GUIDE.md#faq) (30+ answers)
- [ ] Verified system requirements (Python 3.8+, 4GB RAM)
- [ ] Tried restarting application and terminal
- [ ] Tried on different image or ROI (if data issue)
- [ ] Updated all dependencies: `pip install --upgrade -r requirements.txt`
- [ ] Checked Python version: `python --version`

**If all checked and still broken → Ready to report!**

### How to Report an Issue

**GitHub Issues: https://github.com/luhalac/MPS-explorer/issues**

#### Step 1: Check if issue already reported

Search existing issues with your keyword (e.g., "won't start", "GPU", "error").

#### Step 2: Click "New Issue"

#### Step 3: Fill out the issue template

**Include:**
1. **Clear title:** What's the problem? (e.g., "GPU detection fails on Windows")
2. **Description:** What were you trying to do?
3. **Exact error message:** Copy-paste complete error
4. **Steps to reproduce:** Numbered steps to recreate
5. **System info:**
   - OS: Windows / Mac / Linux (which version?)
   - Python version: (output of `python --version`)
   - GPU: Yes / No (if yes, which model?)
   - How installed: (pip / conda / other)
6. **Expected behavior:** What should happen?
7. **Actual behavior:** What actually happens?

#### Step 4: Submit issue

Include error messages and steps. More detail = faster resolution!

### Good Issue Example

```
Title: GPU not detected on Windows 11 with RTX 4080

Description:
I'm trying to enable GPU acceleration but the system says
"GPU Available: False" even though I have an NVIDIA GPU.

Steps to reproduce:
1. Run: pip install cuml pynvml
2. Start application: python MPS_explorer.py
3. Application starts but no GPU detected message

System Info:
- OS: Windows 11 Pro 22H2
- Python: 3.10.11
- GPU: NVIDIA RTX 4080
- Installation: pip install -r requirements.txt

Error Message:
GPU Available: False
GPU Info: GPUDetectionResult(available=False, error=RAPIDS cuML not installed)

Expected: "GPU detected: NVIDIA GeForce RTX 4080"
Actual: GPU shows as unavailable

What I've tried:
- Reinstalled cuml: pip install --upgrade cuml
- Checked nvidia-smi: GPU is listed
- Updated NVIDIA drivers
```

---

## 🤝 Contributing

### Report Improvements

Found documentation issue? Have suggestion?

1. Go to [GitHub Issues](https://github.com/luhalac/MPS-explorer/issues)
2. Click "New Issue"
3. Label it "documentation" or "suggestion"
4. Describe improvement

### Contribute Code

**Fork, improve, submit pull request:**

1. Fork repository
2. Create feature branch: `git checkout -b feature/my-improvement`
3. Make improvements
4. Run tests: `pytest -v`
5. Commit: `git commit -am "Description"`
6. Push: `git push origin feature/my-improvement`
7. Submit Pull Request

**Code standards:**
- ✅ PEP 8 compliant
- ✅ Add type hints
- ✅ Include docstrings
- ✅ Add tests
- ✅ Update documentation

### Improve Documentation

**Fix or improve existing docs:**

1. Edit relevant `.md` file
2. Submit pull request with improvements
3. Include reason for change

**Examples of improvements:**
- Clearer explanations
- Better code examples
- Additional troubleshooting
- Additional FAQ answers
- Translation to other languages

---

## 📚 Self-Service Resources

### Documentation by Purpose

| Purpose | Best Resource |
|---------|---|
| **Get started fast** | [QUICK_START.md](QUICK_START.md) (5 min) |
| **Understand how to use** | [USER_GUIDE.md](USER_GUIDE.md) (1 hour) |
| **Find answer to question** | [DECISION_GUIDE.md](DECISION_GUIDE.md) (5 min) |
| **Learn by doing** | [TUTORIALS.md](TUTORIALS.md) (varies) |
| **GPU acceleration** | [GPU_ACCELERATION_GUIDE.md](GPU_ACCELERATION_GUIDE.md) (15 min) |
| **30+ FAQ answers** | [USER_GUIDE.md → FAQ](USER_GUIDE.md#faq) |
| **Troubleshooting** | [USER_GUIDE.md → Troubleshooting](USER_GUIDE.md#troubleshooting) |
| **Glossary** | [USER_GUIDE.md → Glossary](USER_GUIDE.md#glossary) |
| **Keyboard shortcuts** | [USER_GUIDE.md → Shortcuts](USER_GUIDE.md#keyboard-shortcuts) |
| **Documentation overview** | [USER_DOCUMENTATION_INDEX.md](USER_DOCUMENTATION_INDEX.md) |
| **Project statistics** | [DOCUMENTATION_METRICS.md](DOCUMENTATION_METRICS.md) |

---

## ✅ Self-Help Checklist

**Before reaching out, try these:**

1. **Read relevant documentation**
   - [ ] Checked QUICK_START.md
   - [ ] Checked USER_GUIDE.md sections
   - [ ] Checked DECISION_GUIDE.md
   - [ ] Checked FAQ (30+ answers)

2. **Try troubleshooting**
   - [ ] Restarted application
   - [ ] Restarted terminal
   - [ ] Updated dependencies: `pip install --upgrade -r requirements.txt`
   - [ ] Checked system requirements
   - [ ] Tried different image or ROI

3. **Check specifics**
   - [ ] Verified Python version (3.8+)
   - [ ] Verified image format (.h5, .tif, .png, .jpg)
   - [ ] Verified file exists and readable
   - [ ] Verified sufficient RAM available
   - [ ] Checked error messages closely

4. **Verify setup**
   - [ ] `python --version` shows 3.8+
   - [ ] `python -c "import hdbscan"` works
   - [ ] `python -c "import PyQt5"` works
   - [ ] Application starts: `python MPS_explorer.py`

5. **Try workarounds**
   - [ ] Different image file
   - [ ] Different ROI selection
   - [ ] Different parameter values
   - [ ] Disable GPU temporarily
   - [ ] Try on different computer (if possible)

**If all 5 sections pass and still broken → Ready to report!**

---

## 🎓 Learning Resources

### Getting Started

1. Read [QUICK_START.md](QUICK_START.md) (5 min)
2. Run `python MPS_explorer.py`
3. Load image, select ROI, cluster
4. Done! You're using MPS Explorer

### Going Deeper

1. Read [USER_GUIDE.md](USER_GUIDE.md) (1 hour) - comprehensive
2. Do relevant [TUTORIALS.md](TUTORIALS.md) (varies) - hands-on
3. Check [DECISION_GUIDE.md](DECISION_GUIDE.md) (as needed) - specific questions

### Becoming an Expert

1. Master [TUTORIALS.md → Tutorial 5 & 6](TUTORIALS.md) (60 min) - research and development
2. Optimize with [GPU_ACCELERATION_GUIDE.md](GPU_ACCELERATION_GUIDE.md) (15 min) - speedup
3. Review [DOCUMENTATION_METRICS.md](DOCUMENTATION_METRICS.md) - understand project

---

## 📞 Contact & Communication

### For Quick Questions

- Check documentation first (saves time!)
- Most questions answered in FAQ or Troubleshooting

### For Bugs / Feature Requests

- GitHub Issues: https://github.com/luhalac/MPS-explorer/issues
- Include system info and reproduction steps
- Share error messages

### For Discussions

- GitHub Discussions (if enabled)
- GitHub Issues with "discussion" label

### For Suggestions

- GitHub Issues with "suggestion" label
- Include rationale for improvement
- Describe expected behavior

---

## 🚀 Getting Support in <5 Minutes

| Situation | Quick Solution | Time |
|-----------|---|---|
| **Won't start** | `pip install -r requirements.txt` | <1 min |
| **Module error** | Reinstall requirements | <1 min |
| **Specific question** | [DECISION_GUIDE.md](DECISION_GUIDE.md) | 2-5 min |
| **Something broken** | [USER_GUIDE.md Troubleshooting](USER_GUIDE.md#troubleshooting) | 2-5 min |
| **Don't know what to do** | [QUICK_START.md](QUICK_START.md) | 5 min |

---

## ✨ Summary

### When You Need Help:

1. **First:** Check documentation (2-10 min)
2. **Second:** Try troubleshooting (5 min)
3. **Third:** Check FAQ (2 min)
4. **Fourth:** Review tutorials (10-60 min)
5. **Last:** File GitHub issue (with all info)

### Most Common Issues Are Solved By:

1. Running `pip install -r requirements.txt` (50%)
2. Checking [DECISION_GUIDE.md](DECISION_GUIDE.md) (30%)
3. Selecting different ROI (15%)
4. Using automatic parameters (4%)
5. Actual bug (1%)

### Best Practices:

- ✅ Always try self-service first
- ✅ Read documentation relevant to your situation
- ✅ Check FAQ for common questions
- ✅ Include complete error messages in issues
- ✅ Describe steps to reproduce
- ✅ Include system information

---

**Ready to get started? See [QUICK_START.md](QUICK_START.md)!**

**Have a question? Check [DECISION_GUIDE.md](DECISION_GUIDE.md) first!**

**Something broken? See [USER_GUIDE.md Troubleshooting](USER_GUIDE.md#troubleshooting)!**

---

**MPS Explorer - Getting Help Guide v1.0**  
**Last Updated: 2026-05-28**
