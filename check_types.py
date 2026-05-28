#!/usr/bin/env python3
"""
Local type checking script for MPS Explorer.

Run this before committing to catch type errors early:
    python check_types.py

Exit codes:
    0 = Success (no type errors)
    1 = Failure (type errors found)
"""

import subprocess
import sys


def main():
    """Run MyPy type checking on MPS Explorer source files."""
    print("=" * 80)
    print("Running MyPy type checking on MPS Explorer...")
    print("=" * 80)
    print()
    
    # Files to check (Phase 4 code with full type hints)
    files_to_check = [
        "tools/parameter_cache.py",
    ]
    
    # Run mypy using Python module syntax
    result = subprocess.run(
        [sys.executable, "-m", "mypy"] + files_to_check + [
            "--ignore-missing-imports",
            "--explicit-package-bases",
            "--show-error-codes",
        ]
    )
    
    print()
    print("=" * 80)
    if result.returncode == 0:
        print("[OK] Type checking PASSED - All type hints are correct!")
        print("=" * 80)
        return 0
    else:
        print("[FAIL] Type checking FAILED - See errors above")
        print("=" * 80)
        return 1


if __name__ == "__main__":
    sys.exit(main())
