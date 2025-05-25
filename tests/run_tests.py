#!/usr/bin/env python
"""Test runner for MCDA library tests.

This script discovers and runs all tests in the test suite.
"""

import unittest
import sys
import os

# Add the parent directory to the path so that imports work correctly
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def run_tests():
    """Discover and run all tests in the test suite."""
    print("\n" + "=" * 70)
    print("Running MCDA Tests")
    print("=" * 70)
    
    # Discover all tests in the 'tests' directory
    test_loader = unittest.TestLoader()
    test_suite = test_loader.discover(start_dir=os.path.dirname(__file__), pattern="test_*.py")
    
    # Run the tests
    test_runner = unittest.TextTestRunner(verbosity=2)
    result = test_runner.run(test_suite)
    
    # Print summary
    print("\n" + "=" * 70)
    print(f"Tests Run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Skipped: {len(result.skipped)}")
    print("=" * 70)
    
    return len(result.failures) + len(result.errors)


if __name__ == "__main__":
    exit_code = run_tests()
    sys.exit(exit_code)
