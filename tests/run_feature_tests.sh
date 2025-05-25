#!/bin/bash
# Run all feature tests for MCDA library

# Change to project root directory
cd "$(dirname "$0")"

# Set up display width
export COLUMNS=120

echo "Running MCDA Feature Tests..."
echo "================================"

python -m unittest discover -s features -p "test_*.py" -v

# Check for test failures
if [ $? -eq 0 ]; then
    echo "\nAll tests passed!\n"
    exit 0
else
    echo "\nSome tests failed. See details above.\n"
    exit 1
fi
