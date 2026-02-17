#!/usr/bin/env python3
"""
test_component_and_events.py

Test script for PAPI Gaudi2 component availability and event enumeration.
Verifies that the gaudi2 component is loaded, enabled, and exposes the
expected native events using PAPI command-line utilities.

Prerequisites:
  - PAPI built and installed with gaudi2 component
  - papi_component_avail and papi_native_avail in PATH

Usage:
    python3 test_component_and_events.py
"""

import subprocess
import sys
import os
import shutil

# Constants
# Known events that must be present in the gaudi2 event catalog
REQUIRED_EVENTS = [
    "TPC_KERNEL_EXECUTED",
    "TPC_STALL",
    "EDMA_DESC_PUSH",
    "MME_NUM_OUTER_PRODUCTS",
]

# Helper Functions

def run_command(cmd):
    """Run a command and return (returncode, stdout, stderr)"""
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=30,
        )
        return result.returncode, result.stdout, result.stderr
    except FileNotFoundError:
        return -1, "", f"Command not found: {cmd[0]}"
    except subprocess.TimeoutExpired:
        return -1, "", f"Command timed out: {' '.join(cmd)}"

# Test: Component Availability

def test_component_avail():
    """Verify gaudi2 component is listed and not disabled"""
    print("=" * 70)
    print("Test: Component Availability (papi_component_avail)")
    print("=" * 70)

    if not shutil.which("papi_component_avail"):
        print("  SKIP: papi_component_avail not found in PATH")
        return None

    rc, stdout, stderr = run_command(["papi_component_avail"])
    if rc != 0:
        print(f"  FAILED: papi_component_avail returned {rc}")
        if stderr:
            print(f"  stderr: {stderr.strip()}")
        return False

    # Check that gaudi2 component appears in the output
    found_gaudi2 = False
    is_disabled = False

    for line in stdout.splitlines():
        if "gaudi2" in line.lower():
            found_gaudi2 = True
            print(f"  Found: {line.strip()}")
            if "disabled" in line.lower():
                is_disabled = True

    if not found_gaudi2:
        print("  FAILED: gaudi2 component not found in papi_component_avail output")
        return False

    if is_disabled:
        print("  FAILED: gaudi2 component is disabled")
        return False

    print("  gaudi2 component is available and enabled")
    return True

# Test: Native Event Enumeration

def test_native_events():
    """Verify gaudi2 native events are correctly enumerated"""
    print("\n" + "=" * 70)
    print("Test: Native Event Enumeration (papi_native_avail)")
    print("=" * 70)

    if not shutil.which("papi_native_avail"):
        print("  SKIP: papi_native_avail not found in PATH")
        return None

    rc, stdout, stderr = run_command(["papi_native_avail"])
    if rc != 0:
        print(f"  FAILED: papi_native_avail returned {rc}")
        if stderr:
            print(f"  stderr: {stderr.strip()}")
        return False

    output_lines = stdout.splitlines()

    # Check for required events
    print("\n  Checking required events:")
    all_found = True
    for event_name in REQUIRED_EVENTS:
        found = any(event_name in line for line in output_lines)
        status = "FOUND" if found else "MISSING"
        print(f"    {event_name:<35} {status}")
        if not found:
            all_found = False

    if not all_found:
        print("\n  FAILED: Some required events are missing")
        return False

    # Check that device qualifiers appear
    print("\n  Checking device qualifiers:")
    has_device_qualifier = any(":device=" in line for line in output_lines)
    if has_device_qualifier:
        print("    :device= qualifier found")
    else:
        print("    WARNING: No :device= qualifiers found in output")

    # Count total events listed
    event_count = 0
    for line in output_lines:
        # prefix or event name pattern
        stripped = line.strip()
        if stripped and any(stripped.startswith(prefix) for prefix in
                           ["TPC_", "EDMA_", "MME_"]):
            event_count += 1

    print(f"\n  Total event entries found: {event_count}")

    # contain event name | description)
    has_descriptions = any("|" in line and len(line.split("|")) >= 2
                           for line in output_lines
                           if any(ev in line for ev in REQUIRED_EVENTS))
    if has_descriptions:
        print("  Event descriptions present")

    print("\n  All required events found")
    return True

# Main

def main():
    print("PAPI Gaudi2 Component & Event Enumeration Test")
    print("=" * 70)
    print()

    results = {}

    results["Component Availability"] = test_component_avail()
    results["Native Event Enumeration"] = test_native_events()

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    passed = 0
    failed = 0
    skipped = 0

    for name, result in results.items():
        if result is None:
            status = "SKIPPED"
            skipped += 1
        elif result:
            status = "PASSED"
            passed += 1
        else:
            status = "FAILED"
            failed += 1
        print(f"  {name:<40} {status}")

    print(f"\n  Total: {passed} passed, {failed} failed, {skipped} skipped")
    print("=" * 70)

    if failed > 0:
        print("\nFAILED")
        return 1

    print("\nPASSED")
    return 0

if __name__ == "__main__":
    sys.exit(main())
