# MyPy Type Checking Setup Guide

## Overview

MyPy is a static type checker for Python that helps catch type-related bugs early. MPS Explorer uses MyPy to verify type annotations are correct and catch potential runtime errors.

**Status**: ✅ Type hints complete, MyPy CI/CD integrated

---

## Quick Start

### Run Type Checking Locally

Before committing code, run the type checker:

```bash
# Using the provided script (recommended)
python check_types.py

# Or run MyPy directly
python -m mypy MPS_explorer.py config_loader.py logging_config.py --ignore-missing-imports
```

### Installation

MyPy is not installed by default. Install it for local development:

```bash
# Install MyPy and required dependencies
pip install mypy numpy

# Or install from requirements-dev.txt
pip install -r requirements-dev.txt
```

---

## How MyPy Works

### Type Checking Basics

MyPy validates type annotations by analyzing your code without running it:

```python
# Correct: type matches value
def add(x: int, y: int) -> int:
    return x + y

result: int = add(5, 3)  # OK

# Error: type mismatch
result: int = add("a", "b")  # MyPy error: expected int, got str
```

### Why Type Checking Matters

1. **Early Bug Detection**: Catch mistakes before they become runtime errors
2. **Better IDE Support**: Type hints enable autocomplete and refactoring
3. **Self-Documenting Code**: Types serve as inline documentation
4. **Easier Maintenance**: Future developers understand expected types

---

## Local Setup

### Installation

```bash
# Option 1: Install MyPy alone
pip install mypy

# Option 2: Install with development dependencies
pip install -r requirements-dev.txt

# Option 3: Install from source (for latest features)
pip install git+https://github.com/python/mypy.git
```

### Running Type Checks

#### Method 1: Using the provided script (Recommended)

```bash
# Standard type checking
python check_types.py

# Strict mode (more thorough)
python check_types.py --strict

# Verbose output
python check_types.py --verbose
```

#### Method 2: Direct MyPy command

```bash
# Basic type checking
python -m mypy MPS_explorer.py config_loader.py logging_config.py --ignore-missing-imports

# With strict mode
python -m mypy MPS_explorer.py config_loader.py logging_config.py --ignore-missing-imports --strict

# Check only specific file
python -m mypy MPS_explorer.py --ignore-missing-imports

# Generate detailed report
python -m mypy MPS_explorer.py --ignore-missing-imports --show-error-codes
```

### Configuration

MyPy configuration is in `.mypy.ini`:

```ini
[mypy]
python_version = 3.10
warn_return_any = True
warn_unused_configs = True
warn_redundant_casts = True
warn_unused_ignores = True
strict_optional = True
ignore_missing_imports = True
```

**Key Options**:
- `python_version`: Target Python version (3.10+)
- `warn_return_any`: Warn when returning Any type
- `strict_optional`: Enforce strict None/Optional handling
- `ignore_missing_imports`: Don't error on missing library stubs

---

## IDE Integration

### VS Code

Install the MyPy extension for real-time type checking:

```bash
# Install extension (command line)
code --install-extension ms-python.vscode-pylance

# Or search "Pylance" in VS Code extensions
```

**Configuration** (`settings.json`):
```json
{
  "python.linting.mypyEnabled": true,
  "python.linting.mypyArgs": [
    "--ignore-missing-imports",
    "--no-error-summary"
  ]
}
```

### PyCharm

Type checking is built-in:

1. Go to: **Settings → Python → Python Type Checker**
2. Select: **MyPy**
3. Configure MyPy path if needed

### Vim/Neovim

Use with `coc.nvim` or `ALE`:

```vim
" Using coc.nvim
" Install: CocInstall coc-pyright

" Using ALE
let g:ale_linters = {'python': ['mypy']}
let g:ale_python_mypy_options = '--ignore-missing-imports'
```

---

## Understanding MyPy Errors

### Common Error Messages

#### Type Mismatch

```
error: Incompatible types in assignment (expression has type "str", variable has type "int")
```

**Fix**: Ensure assigned value matches variable type

```python
# Wrong
x: int = "hello"

# Correct
x: int = 42
x: str = "hello"
```

#### Missing Type Annotation

```
error: Need type annotation for "result"
```

**Fix**: Add type hint to variable declaration

```python
# Wrong
result = some_function()

# Correct
result: int = some_function()
```

#### Optional Type Issues

```
error: Item "None" is not subscriptable
```

**Fix**: Check for None before using (or assert not None)

```python
# Wrong
data: Optional[list] = get_data()
value = data[0]  # Error: might be None

# Correct
data: Optional[list] = get_data()
if data is not None:
    value = data[0]  # OK, data is not None

# Or use assert
assert data is not None
value = data[0]  # OK, asserted not None
```

#### Returning Wrong Type

```
error: Incompatible return value type (got "None", expected "int")
```

**Fix**: Ensure function returns correct type

```python
# Wrong
def get_count() -> int:
    pass  # Returns None, but declared int

# Correct
def get_count() -> int:
    return 0
```

### Suppressing Errors (When Necessary)

```python
# Suppress single line
x = some_func()  # type: ignore

# Suppress entire function
# type: ignore
def untyped_function():
    pass

# Suppress specific error code
x: int = "hello"  # type: ignore[assignment]
```

**Note**: Use `type: ignore` sparingly. Prefer fixing the underlying issue.

---

## CI/CD Integration

### GitHub Actions Workflow

The project includes an automated workflow (`.github/workflows/type-check.yml`) that:

- Runs on every push to `main` or `develop`
- Runs on every pull request
- Can be triggered manually
- Tests on Python 3.10 and 3.11
- Reports results in PR checks

**Workflow Status in GitHub**:
- ✅ Green check: All type checks passed
- ❌ Red X: Type checking found errors
- Status visible in PR checks section

### Local Pre-commit Check

Run before committing:

```bash
# Type checking
python check_types.py

# Tests
pytest test_mps_explorer.py -v

# Both together
python check_types.py && pytest test_mps_explorer.py -v
```

### Fixing CI/CD Failures

If GitHub Actions reports type checking failures:

1. **Check the error** in GitHub Actions output
2. **Run locally** to see detailed errors:
   ```bash
   python check_types.py
   ```
3. **Fix the issue** (update type hints or code)
4. **Verify** locally before pushing:
   ```bash
   python check_types.py
   ```

---

## Type Hints in MPS Explorer

### What's Typed

✅ Complete type hints for:
- Instance attributes (50+ annotated)
- Method signatures (26 critical methods)
- Return types (all core methods)
- Parameter types (all function parameters)

### Type Annotation Examples

```python
from typing import Optional, List
from numpy.typing import NDArray
import numpy as np

# Instance attribute
self.cluster_centroids: Optional[NDArray[np.float64]] = None

# Function with type hints
def cluster(self, channel: int) -> None:
    """Run DBSCAN clustering on data."""
    pass

# Return various types
def get_config() -> dict[str, Any]:
    """Load configuration."""
    pass

def filter_points(points: NDArray[np.float64], radius: float) -> Optional[NDArray[np.float64]]:
    """Filter points by radius."""
    pass
```

### NumPy Type Hints

MPS Explorer uses modern NumPy type hints:

```python
from numpy.typing import NDArray
import numpy as np

# Specific dtype
data: NDArray[np.float64]  # array of float64

# Any dtype
data: NDArray[np.any_]  # array of any numeric type

# Optional array
data: Optional[NDArray[np.float64]]  # array or None
```

---

## Troubleshooting

### MyPy Not Found

```
FileNotFoundError: [WinError 2] The system cannot find the file specified
```

**Solution**: Install MyPy

```bash
pip install mypy
```

### Python Version Error

```
Python 3.7 is not supported (must be 3.10 or higher)
```

**Solution**: Update `.mypy.ini` or use Python 3.10+

```bash
# Check Python version
python --version

# Upgrade Python if needed
# See https://www.python.org/downloads/
```

### Missing Import Stubs

```
error: Skipping analyzing "pyqtgraph": found no overloads for "__getitem__"
```

**Solution**: This is expected for PyQtGraph. Use `--ignore-missing-imports` (already configured)

### Too Many Errors

If MyPy reports many errors initially:

1. **Don't panic** - this is normal when adding type checking
2. **Start gradual** - focus on one file at a time
3. **Use issues** - see "Understanding MyPy Errors" section
4. **Disable strict mode** - use default checking, not `--strict`

---

## Best Practices

### 1. Always Specify Types

```python
# Good
def process_data(data: list[int]) -> int:
    return sum(data)

# Avoid
def process_data(data):  # Type missing
    return sum(data)
```

### 2. Use Optional for Nullable Values

```python
# Good
def get_user(uid: int) -> Optional[User]:
    """Return user or None if not found."""
    pass

# Avoid
def get_user(uid: int) -> User:  # Might return None!
    pass
```

### 3. Type Hints for Complex Returns

```python
# Good
def parse_config() -> dict[str, Any]:
    """Return configuration dictionary."""
    pass

# Good
def get_clusters() -> tuple[NDArray[np.float64], NDArray[np.int64]]:
    """Return centroids and labels."""
    pass

# Avoid
def get_clusters():  # Type unclear
    pass
```

### 4. Keep Type Hints Updated

When changing code, update type hints:

```python
# If changing parameter type
def process(data: list[str]) -> None:  # Changed from list[int]
    pass

# If changing return type
def get_result() -> Optional[int]:  # Added Optional
    pass
```

---

## Integration with Other Tools

### With pytest

Type checking and tests run separately but complementarily:

```bash
# Run tests
pytest test_mps_explorer.py -v

# Run type checking
python -m mypy MPS_explorer.py --ignore-missing-imports

# Both (check script does this)
python check_types.py && pytest test_mps_explorer.py -v
```

### With pre-commit hooks (Optional)

Set up automatic checking before commits:

```bash
# Install pre-commit
pip install pre-commit

# Create .pre-commit-config.yaml with mypy hook
# Automatically run on git commit
```

---

## Advanced Configuration

### Enabling Stricter Checking

For higher type safety, gradually enable stricter options:

```ini
[mypy]
# Current (lenient)
strict_optional = True

# Can enable later (stricter)
check_untyped_defs = True
disallow_untyped_defs = True
disallow_any_unimported = True
```

### Custom MyPy Plugins

Advanced users can write custom plugins or use existing ones:

```ini
[mypy]
plugins = mypy_plugin_name
```

See [MyPy documentation](https://mypy.readthedocs.io/) for details.

---

## Resources

### Documentation

- [MyPy Official Documentation](https://mypy.readthedocs.io/)
- [Python Type Hints (PEP 484)](https://www.python.org/dev/peps/pep-0484/)
- [NumPy Type Hints](https://numpy.org/doc/stable/reference/typing.html)

### Tools

- [MyPy GitHub Repository](https://github.com/python/mypy)
- [Python Type Checking Guide](https://docs.python-guide.org/writing/tests/)
- [Type Hint Cheat Sheet](https://mypy.readthedocs.io/en/stable/cheat_sheet_py3.html)

### Community

- Stack Overflow: [tag:mypy](https://stackoverflow.com/questions/tagged/mypy)
- Python Discourse: [typing discussions](https://discuss.python.org/)

---

## Summary

MyPy provides:
- ✅ Static type checking without runtime overhead
- ✅ Early detection of type-related bugs
- ✅ Better IDE support and autocomplete
- ✅ Self-documenting code through types
- ✅ CI/CD integration for quality assurance

**Quick commands**:
```bash
# Install
pip install mypy numpy

# Check locally (before committing)
python check_types.py

# CI/CD runs automatically on push/PR
```

**Status**: Type hints fully implemented, MyPy checks integrated in CI/CD ✅

For questions or issues with type checking, see the [MyPy documentation](https://mypy.readthedocs.io/).
