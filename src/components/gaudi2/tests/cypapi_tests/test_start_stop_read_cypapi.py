#!/usr/bin/env python3
"""
test_start_stop_read_cypapi.py

Test script for PAPI Gaudi2 component full measurement lifecycle using cyPAPI.
Exercises the complete PAPI workflow: init, create eventset, add events,
start, run workload, stop, read values, cleanup, and shutdown on a
single Gaudi2 device.

Prerequisites:
  - PAPI built and installed with gaudi2 component
  - cyPAPI installed (pip install -e /path/to/cyPAPI)
  - PyTorch with Habana support

Usage:
    PT_HPU_LAZY_MODE=1 python3 test_start_stop_read_cypapi.py
"""

import sys
import os

import cypapi as cyp

# PyTorch Initialization and Workload

_pytorch_initialized = False
_torch = None
_hthpu = None
_device = None

def init_pytorch():
    """Initialize PyTorch and acquire the Gaudi2 device BEFORE PAPI init"""
    global _pytorch_initialized, _torch, _hthpu, _device

    if _pytorch_initialized:
        return True

    try:
        import torch
        import habana_frameworks.torch.hpu as hthpu

        _torch = torch
        _hthpu = hthpu

        print("  Initializing PyTorch HPU...")
        _device = torch.device("hpu")

        # Force device initialization by creating a small tensor
        _ = torch.zeros(1, device=_device)
        hthpu.synchronize()

        _pytorch_initialized = True
        print("  PyTorch HPU initialized successfully")
        return True

    except ImportError as e:
        print(f"  PyTorch/Habana not available: {e}")
        return False
    except Exception as e:
        print(f"  Failed to initialize PyTorch HPU: {e}")
        return False

def run_pytorch_workload():
    """Run a PyTorch matmul workload on Gaudi2"""
    global _torch, _hthpu, _device

    if not _pytorch_initialized:
        print("  ERROR: PyTorch not initialized")
        return False

    try:
        dtype = _torch.float32
        size = 1024
        a = _torch.randn(size, size, dtype=dtype, device=_device)
        b = _torch.randn(size, size, dtype=dtype, device=_device)

        # Warm-up
        for _ in range(3):
            c = _torch.matmul(a, b)
        _hthpu.synchronize()

        # Actual workload
        for _ in range(10):
            c = _torch.matmul(a, b)
        _hthpu.synchronize()

        return True

    except Exception as e:
        print(f"  Workload failed: {e}")
        return False

# Main Test

def main():
    # Step 0: Initialize PyTorch FIRST to acquire the device
    print("[0] Initializing PyTorch HPU (must be done before PAPI)...")
    if not init_pytorch():
        print("    SKIP: PyTorch HPU not available")
        return 0

    # Step 1: Initialize cyPAPI
    print("\n[1] Initializing cyPAPI...")
    try:
        cyp.cyPAPI_library_init(cyp.PAPI_VER_CURRENT)
    except Exception as e:
        print(f"  cyPAPI init failed: {e}")
        print("\nFAILED")
        return 1

    if cyp.cyPAPI_is_initialized() != 1:
        print("  ERROR: cyPAPI not initialized")
        print("\nFAILED")
        return 1
    print("  cyPAPI initialized successfully")

    # Step 2: Create event set
    print("\n[2] Creating event set...")
    try:
        eventset = cyp.CypapiCreateEventset()
    except Exception as e:
        print(f"  Failed to create eventset: {e}")
        cyp.cyPAPI_shutdown()
        print("\nFAILED")
        return 1
    print(f"  Event set created: {eventset}")

    # Step 3: Add Gaudi2 events
    print("\n[3] Adding Gaudi2 events...")
    events = [
        "gaudi2:::TPC_KERNEL_EXECUTED:device=0",
        "gaudi2:::TPC_STALL:device=0",
        "gaudi2:::TPC_VECTOR_PIPE_EXEC:device=0",
        "gaudi2:::TPC_ICACHE_HIT:device=0",
        "gaudi2:::TPC_DCACHE_HIT:device=0",
    ]

    added_events = []
    for event in events:
        try:
            eventset.add_named_event(event)
            print(f"    Added: {event}")
            added_events.append(event)
        except Exception as e:
            print(f"    Failed: {event} ({e})")

    if len(added_events) == 0:
        print("\n  ERROR: No events added. Is the gaudi2 component built?")
        eventset.destroy_eventset()
        cyp.cyPAPI_shutdown()
        print("\nFAILED")
        return 1

    # Step 4: Start counting
    print(f"\n[4] Starting counters ({len(added_events)} events)...")
    try:
        eventset.start()
    except Exception as e:
        print(f"  Failed to start: {e}")
        eventset.cleanup_eventset()
        eventset.destroy_eventset()
        cyp.cyPAPI_shutdown()
        print("\nFAILED")
        return 1
    print("  Counters started")

    # Step 5: Run workload
    print("\n[5] Running PyTorch workload...")
    if not run_pytorch_workload():
        print("  WARNING: Workload failed, counters may be zero")

    # Step 6: Stop and read
    print("\n[6] Stopping counters and reading values...")
    try:
        values = eventset.stop()
    except Exception as e:
        print(f"  Failed to stop: {e}")
        eventset.cleanup_eventset()
        eventset.destroy_eventset()
        cyp.cyPAPI_shutdown()
        print("\nFAILED")
        return 1

    print("\n  Results:")
    print("  " + "-" * 55)
    all_valid = True
    for event, value in zip(added_events, values):
        event_name = event.split(":::")[-1]
        print(f"  {event_name:<40}: {value:>12,}")
        if value < 0:
            print(f"    WARNING: Negative counter value for {event_name}")
            all_valid = False
    print("  " + "-" * 55)

    if not all_valid:
        print("\n  ERROR: Some counter values are invalid (negative)")

    # Step 7: Cleanup
    print("\n[7] Cleaning up...")
    eventset.cleanup_eventset()
    eventset.destroy_eventset()
    cyp.cyPAPI_shutdown()
    print("  Done")

    if all_valid:
        print("PASSED")
    else:
        print("FAILED")

    return 0 if all_valid else 1

if __name__ == "__main__":
    sys.exit(main())
