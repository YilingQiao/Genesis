#!/bin/bash
# Run Genesis tests with full coverage (Python + Kernel)
#
# Usage:
#   ./scripts/run_coverage.sh                    # Run all tests
#   ./scripts/run_coverage.sh tests/test_*.py   # Run specific tests
#   ./scripts/run_coverage.sh -k "test_rigid"   # Run tests matching pattern

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

# Use Python/pytest from current environment (or override via env vars)
# This finds the Python that's currently in PATH
PYTHON="${PYTHON:-$(command -v python3 || command -v python)}"
PYTEST="${PYTEST:-$(command -v pytest || echo "$PYTHON -m pytest")}"

# Coverage output directories
COVERAGE_DATA_DIR=".coverage_data"
COVERAGE_REPORT_DIR="coverage_report"
KERNEL_COVERAGE_FILE="kernel_coverage.json"

echo "=============================================="
echo "Genesis Coverage Test Runner"
echo "=============================================="
echo "Python: $PYTHON"
echo "Pytest: $PYTEST"
echo ""

# Create coverage data directory
mkdir -p "$COVERAGE_DATA_DIR"

# Install coverage dependencies if needed
echo "Checking dependencies..."
$PYTHON -c "import coverage" 2>/dev/null || $PYTHON -m pip install coverage[toml] pytest-cov

# Build test arguments
TEST_ARGS="$@"
if [ -z "$TEST_ARGS" ]; then
    TEST_ARGS="tests/"
fi

echo ""
echo "Step 1: Running tests with Python coverage..."
echo "----------------------------------------------"

# Run pytest with coverage
# Note: We disable xdist parallelism for more accurate coverage
# Use --numprocesses=1 for single-process, or remove for parallel (may miss some coverage)
$PYTEST \
    --cov=genesis \
    --cov-report=term-missing \
    --cov-report=html:$COVERAGE_REPORT_DIR/python \
    --cov-report=json:$COVERAGE_REPORT_DIR/python_coverage.json \
    --cov-config=pyproject.toml \
    --numprocesses=1 \
    -p tests.coverage.conftest_plugin \
    --kernel-coverage \
    --kernel-coverage-report="$KERNEL_COVERAGE_FILE" \
    $TEST_ARGS || true

echo ""
echo "Step 2: Generating combined report..."
echo "----------------------------------------------"

# Generate combined coverage report
$PYTHON tests/coverage/generate_report.py \
    --coverage-data "$COVERAGE_DATA_DIR" \
    --kernel-data "$KERNEL_COVERAGE_FILE" \
    --output "$COVERAGE_REPORT_DIR"

echo ""
echo "=============================================="
echo "Coverage reports generated!"
echo "=============================================="
echo ""
echo "View reports:"
echo "  Combined:  $COVERAGE_REPORT_DIR/index.html"
echo "  Python:    $COVERAGE_REPORT_DIR/python/index.html"
echo "  Kernel:    $KERNEL_COVERAGE_FILE"
echo ""
