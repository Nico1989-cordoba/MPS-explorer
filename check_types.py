#!/usr/bin/env python3
"""
Local Type Checking Script for MPS Explorer

This script runs MyPy type checking on the MPS Explorer codebase to catch
type-related errors before committing. Run this script before pushing changes
to ensure type safety.

Usage:
    python check_types.py                   # Standard checking
    python check_types.py --strict          # Strict mode (more checks)
    python check_types.py --verbose         # Verbose output

Exit codes:
    0: Type checking passed (success)
    1: Type checking failed (errors found)

Installation:
    pip install mypy numpy
"""

import subprocess
import sys
import argparse
from pathlib import Path


def run_mypy(strict: bool = False, verbose: bool = False) -> int:
    """
    Run MyPy type checker on MPS Explorer Python files.

    Parameters
    ----------
    strict : bool
        If True, enable strict mode with additional checks.
    verbose : bool
        If True, show verbose output from MyPy.

    Returns
    -------
    int
        Exit code (0 = success, 1 = failure)
    """
    # Files to check
    files_to_check = [
        "MPS_explorer.py",
        "config_loader.py",
        "logging_config.py",
    ]

    # Build MyPy command (use python -m for cross-platform compatibility)
    cmd = ["python", "-m", "mypy"]
    cmd.extend(files_to_check)
    cmd.append("--ignore-missing-imports")

    if strict:
        cmd.append("--strict")
        print("[LOCK] Running in STRICT mode (more thorough checking)")

    if verbose:
        cmd.append("--verbose")
        print("[INFO] Running in VERBOSE mode")

    print("\n" + "="*70)
    print("Running MyPy Type Checking for MPS Explorer")
    print("="*70)
    print(f"Command: {' '.join(cmd)}\n")

    try:
        # Run MyPy using shell on Windows for compatibility
        cmd_str = " ".join(cmd)
        result = subprocess.run(cmd_str, shell=True)

        print("\n" + "="*70)
        if result.returncode == 0:
            print("[OK] SUCCESS: Type checking passed!")
            print("="*70)
            print("\nAll type annotations are correct. Ready to commit!")
            return 0
        else:
            print("[FAIL] FAILURE: Type checking found errors!")
            print("="*70)
            print("\nPlease fix the errors shown above before committing.")
            return 1

    except FileNotFoundError:
        print("\n[ERROR] MyPy is not installed!")
        print("="*70)
        print("\nPlease install MyPy:")
        print("  pip install mypy numpy")
        return 1
    except Exception as e:
        print(f"\n[ERROR] An error occurred: {e}")
        print("="*70)
        return 1


def check_files_exist() -> bool:
    """
    Check that required files exist in the current directory.

    Returns
    -------
    bool
        True if all files exist, False otherwise
    """
    required_files = [
        "MPS_explorer.py",
        "config_loader.py",
        "logging_config.py",
    ]

    missing = []
    for file in required_files:
        if not Path(file).exists():
            missing.append(file)

    if missing:
        print("[ERROR] Missing required files:")
        for file in missing:
            print(f"  - {file}")
        print("\nPlease run this script from the MPS-explorer project root directory.")
        return False

    return True


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Type check MPS Explorer codebase with MyPy",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python check_types.py              # Standard type checking
  python check_types.py --strict     # Strict mode with more checks
  python check_types.py --verbose    # Verbose output
        """
    )

    parser.add_argument(
        "--strict",
        action="store_true",
        help="Enable strict mode for more thorough type checking"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show verbose output from MyPy"
    )

    args = parser.parse_args()

    # Check that we're in the right directory
    if not check_files_exist():
        return 1

    # Run MyPy
    exit_code = run_mypy(strict=args.strict, verbose=args.verbose)

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
