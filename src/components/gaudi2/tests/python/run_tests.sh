#!/bin/bash
#
# run_tests.sh
#
# Test runner for PAPI Gaudi2 component Python tests.
# Sets up the environment and runs each test script.
#
# Usage:
#   PT_HPU_LAZY_MODE=1 bash run_tests.sh
#
# Environment variables:
#   PAPI_DIR  - Path to PAPI install directory (auto-detected if not set)
#

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Auto-detect PAPI install directory
if [ -z "$PAPI_DIR" ]; then
    PAPI_DIR="$(cd "$SCRIPT_DIR/../../../../install" 2>/dev/null && pwd)"
fi

if [ ! -d "$PAPI_DIR" ]; then
    echo "ERROR: PAPI install directory not found: $PAPI_DIR"
    echo "Set PAPI_DIR environment variable or build PAPI first."
    exit 1
fi

echo "Using PAPI_DIR: $PAPI_DIR"

# Setup environment
export LD_LIBRARY_PATH="$PAPI_DIR/lib:${LD_LIBRARY_PATH:-}"
export PATH="$PAPI_DIR/bin:$PATH"

# Test list
TESTS=(
    "test_component_and_events.py"
    "test_start_stop_read.py"
    "test_multidevice.py"
)

# Run tests
TOTAL=0
PASSED=0
FAILED=0

for test in "${TESTS[@]}"; do
    TOTAL=$((TOTAL + 1))
    echo "Running: $test"
    echo -e "-------------------------------------\n"

    python3 "$SCRIPT_DIR/$test"
    rc=$?

    if [ $rc -eq 0 ]; then
        PASSED=$((PASSED + 1))
        echo -e "Result: \e[32mPASSED\e[0m"
    else
        FAILED=$((FAILED + 1))
        echo -e "Result: \e[31mFAILED\e[0m (exit code $rc)"
    fi

    echo -e "-------------------------------------\n"
done

# Summary
echo ""
echo "OVERALL SUMMARY"
echo -e "-------------------------------------\n"
echo "  Total:  $TOTAL"
echo "  Passed: $PASSED"
echo "  Failed: $FAILED"
echo -e "-------------------------------------\n"

if [ $FAILED -gt 0 ]; then
    exit 1
fi

exit 0
