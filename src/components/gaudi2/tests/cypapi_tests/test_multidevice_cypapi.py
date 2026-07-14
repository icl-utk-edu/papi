#!/usr/bin/env python3
"""
test_multidevice_cypapi.py

Test script for PAPI Gaudi2 component multi-device support using cyPAPI.
Tests device qualifiers, monitoring the same event across multiple devices,
mixed events across devices, invalid device handling, and multiple reads
during workload execution.

Prerequisites:
  - PAPI built and installed with gaudi2 component
  - cyPAPI installed (pip install -e /path/to/cyPAPI)
  - PyTorch with Habana support
  - One or more Gaudi2 devices available

Usage:
    PT_HPU_LAZY_MODE=1 python3 test_multidevice_cypapi.py
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

# Helper Functions

def detect_devices_by_probing():
    """Detect the number of Gaudi2 devices that PAPI can actually use by
    attempting to add an event with increasing device indices.
    Returns (num_devices, device_ids) where device_ids is a 0-based list."""
    device_ids = []
    for d in range(16):
        try:
            eventset = cyp.CypapiCreateEventset()
            eventset.add_named_event(f"gaudi2:::TPC_KERNEL_EXECUTED:device={d}")
            device_ids.append(d)
            eventset.cleanup_eventset()
            eventset.destroy_eventset()
        except Exception:
            try:
                eventset.destroy_eventset()
            except Exception:
                pass
            break
    return len(device_ids), device_ids

def cleanup_eventset(eventset):
    """Cleanup and destroy an eventset"""
    if eventset is not None:
        try:
            eventset.cleanup_eventset()
            eventset.destroy_eventset()
        except Exception:
            pass

# Sub-test 1: Single event on a specific device

def test_single_device_qualifier(device_id):
    """Test adding a single event with an explicit :device=N qualifier"""
    print(f"Sub-test 1: Single event with :device={device_id} qualifier")
    try:
        eventset = cyp.CypapiCreateEventset()
    except Exception as e:
        print(f"  Failed to create eventset: {e}")
        return False

    event_name = f"gaudi2:::TPC_KERNEL_EXECUTED:device={device_id}"
    print(f"  Adding event: {event_name}")
    try:
        eventset.add_named_event(event_name)
    except Exception as e:
        print(f"  Failed to add event: {e}")
        eventset.destroy_eventset()
        return False
    print(f"  Event added successfully")

    print(f"  Starting counters...")
    try:
        eventset.start()
    except Exception as e:
        print(f"  Failed to start: {e}")
        cleanup_eventset(eventset)
        return False

    print(f"  Running workload...")
    run_pytorch_workload()

    try:
        values = eventset.stop()
    except Exception as e:
        print(f"  Failed to stop: {e}")
        cleanup_eventset(eventset)
        return False

    print(f"\n  Result:")
    print(f"    TPC_KERNEL_EXECUTED:device={device_id} = {values[0]:>15,}")

    cleanup_eventset(eventset)
    print(f"\n  PASSED")
    return True

# Sub-test 2: Same event across multiple devices

def test_same_event_multiple_devices(device_ids):
    """Test monitoring the same event across all available devices"""
    print(f"Sub-test 2: Same event across {len(device_ids)} devices {device_ids}")
    try:
        eventset = cyp.CypapiCreateEventset()
    except Exception as e:
        print(f"  Failed to create eventset: {e}")
        return False

    for d in device_ids:
        event_name = f"gaudi2:::TPC_KERNEL_EXECUTED:device={d}"
        print(f"  Adding: {event_name}")
        try:
            eventset.add_named_event(event_name)
        except Exception as e:
            print(f"  Failed to add event: {e}")
            cleanup_eventset(eventset)
            return False

    num_events = len(device_ids)
    print(f"\n  Starting counters on {num_events} devices...")
    try:
        eventset.start()
    except Exception as e:
        print(f"  Failed to start: {e}")
        cleanup_eventset(eventset)
        return False

    print(f"  Running workload...")
    run_pytorch_workload()

    try:
        values = eventset.stop()
    except Exception as e:
        print(f"  Failed to stop: {e}")
        cleanup_eventset(eventset)
        return False

    print(f"\n  Results (TPC_KERNEL_EXECUTED per device):")
    print(f"  {'Device':<10} {'Value':>15}")
    print(f"  {'-'*10} {'-'*15}")
    for i, d in enumerate(device_ids):
        print(f"  device={d:<5} {values[i]:>15,}")

    cleanup_eventset(eventset)
    print(f"\n  PASSED")
    return True

# Sub-test 3: Different events across different devices

def test_mixed_events_devices(device_ids):
    """Test monitoring different events on different devices"""
    print(f"Sub-test 3: Mixed events across devices {device_ids}")
    try:
        eventset = cyp.CypapiCreateEventset()
    except Exception as e:
        print(f"  Failed to create eventset: {e}")
        return False

    d0 = device_ids[0]
    events = [
        (f"gaudi2:::TPC_KERNEL_EXECUTED:device={d0}", f"TPC_KERNEL_EXECUTED:device={d0}"),
        (f"gaudi2:::TPC_STALL:device={d0}", f"TPC_STALL:device={d0}"),
        (f"gaudi2:::MME_NUM_OUTER_PRODUCTS:device={d0}", f"MME_NUM_OUTER_PRODUCTS:device={d0}"),
    ]

    if len(device_ids) > 1:
        d1 = device_ids[1]
        events.extend([
            (f"gaudi2:::TPC_KERNEL_EXECUTED:device={d1}", f"TPC_KERNEL_EXECUTED:device={d1}"),
            (f"gaudi2:::TPC_STALL:device={d1}", f"TPC_STALL:device={d1}"),
            (f"gaudi2:::MME_NUM_OUTER_PRODUCTS:device={d1}", f"MME_NUM_OUTER_PRODUCTS:device={d1}"),
        ])

    added_labels = []
    for full_name, label in events:
        print(f"  Adding: {full_name}")
        try:
            eventset.add_named_event(full_name)
            added_labels.append(label)
        except Exception as e:
            print(f"    (skipped: {e})")

    if len(added_labels) == 0:
        print("\n  ERROR: No events could be added")
        eventset.destroy_eventset()
        return False

    print(f"\n  Starting counters ({len(added_labels)} events)...")
    try:
        eventset.start()
    except Exception as e:
        print(f"  Failed to start: {e}")
        cleanup_eventset(eventset)
        return False

    print(f"  Running workload...")
    run_pytorch_workload()

    try:
        values = eventset.stop()
    except Exception as e:
        print(f"  Failed to stop: {e}")
        cleanup_eventset(eventset)
        return False

    print(f"\n  Results:")
    print(f"  {'Event':<40} {'Value':>15}")
    print(f"  {'-'*40} {'-'*15}")
    for label, value in zip(added_labels, values):
        print(f"  {label:<40} {value:>15,}")

    cleanup_eventset(eventset)
    print(f"\n  PASSED")
    return True

# Sub-test 4: Invalid device qualifier (negative test)

def test_invalid_device(num_devices):
    """Test that adding an event with an invalid device index fails gracefully"""
    print(f"Sub-test 4: Invalid device qualifier (negative test)")
    try:
        eventset = cyp.CypapiCreateEventset()
    except Exception as e:
        print(f"  Failed to create eventset: {e}")
        return False

    invalid_device = num_devices
    event_name = f"gaudi2:::TPC_KERNEL_EXECUTED:device={invalid_device}"
    print(f"  Adding event with invalid device: {event_name}")

    try:
        eventset.add_named_event(event_name)
        print(f"  ERROR: Should have rejected device={invalid_device} but accepted it")
        cleanup_eventset(eventset)
        return False
    except Exception as e:
        print(f"  Correctly rejected: {e}")
        eventset.destroy_eventset()
        print(f"\n  PASSED")
        return True

# Sub-test 5: Multiple reads during workload

def test_read_during_workload(device_id):
    """Test reading counter values during workload execution"""
    print(f"Sub-test 5: Multiple reads during workload on device={device_id}")
    try:
        eventset = cyp.CypapiCreateEventset()
    except Exception as e:
        print(f"  Failed to create eventset: {e}")
        return False

    events = [
        f"gaudi2:::TPC_KERNEL_EXECUTED:device={device_id}",
        f"gaudi2:::TPC_VECTOR_PIPE_EXEC:device={device_id}",
    ]
    added_events = []
    for event_name in events:
        print(f"  Adding: {event_name}")
        try:
            eventset.add_named_event(event_name)
            added_events.append(event_name)
        except Exception as e:
            print(f"    (skipped: {e})")

    if len(added_events) == 0:
        eventset.destroy_eventset()
        return False

    print(f"\n  Starting counters...")
    try:
        eventset.start()
    except Exception as e:
        print(f"  Failed to start: {e}")
        cleanup_eventset(eventset)
        return False

    # Read before workload
    values_before = eventset.read()
    print(f"  Before workload: {values_before}")

    # Run workload and read
    print(f"  Running workload...")
    run_pytorch_workload()

    values_after = eventset.read()
    print(f"  After workload:  {values_after}")

    # Run another workload and stop
    print(f"  Running second workload...")
    run_pytorch_workload()

    values_final = eventset.stop()
    print(f"  Final (stop):    {values_final}")

    if values_before and values_after and values_final:
        print(f"\n  Counter progression:")
        print(f"  {'Event':<30} {'Before':>12} {'After':>12} {'Final':>12}")
        print(f"  {'-'*30} {'-'*12} {'-'*12} {'-'*12}")
        labels = [e.split(":::")[-1] for e in added_events]
        for i, label in enumerate(labels):
            print(f"  {label:<30} {values_before[i]:>12,} {values_after[i]:>12,} {values_final[i]:>12,}")

    cleanup_eventset(eventset)
    print(f"\n  PASSED")
    return True

# Main

def main():
    # Initialize PyTorch FIRST (acquires device fd needed by PAPI)
    print("\n[SETUP] Initializing PyTorch HPU...")
    if not init_pytorch():
        print("  SKIP: PyTorch HPU not available")
        return 0

    # Initialize cyPAPI
    print("\n[SETUP] Initializing cyPAPI...")
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

    # Detect devices by probing PAPI (not all /dev/accel/* are Gaudi2)
    num_devices, device_ids = detect_devices_by_probing()
    print(f"\n  Detected {num_devices} Gaudi2 device(s) usable by PAPI")
    print(f"  PAPI device indices: {device_ids}")
    if num_devices == 0:
        print("  SKIP: No Gaudi2 devices found via PAPI")
        cyp.cyPAPI_shutdown()
        return 0

    # Run sub-tests
    results = {}

    results["Sub-test 1: Single device qualifier"] = test_single_device_qualifier(0)

    if num_devices > 1:
        test_devs = device_ids[:4]  # Cap at 4 devices
        results["Sub-test 2: Same event multi-device"] = test_same_event_multiple_devices(
            test_devs
        )
    else:
        print(f"\n  Skipping Sub-test 2 (need >1 device, have {num_devices})")
        results["Sub-test 2: Same event multi-device"] = None

    results["Sub-test 3: Mixed events/devices"] = test_mixed_events_devices(device_ids)

    results["Sub-test 4: Invalid device"] = test_invalid_device(num_devices)

    results["Sub-test 5: Read during workload"] = test_read_during_workload(0)

    # Summary
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
        print(f"  {name:<45} {status}")

    print(f"\n  Total: {passed} passed, {failed} failed, {skipped} skipped")

    cyp.cyPAPI_shutdown()

    if failed > 0:
        print("\nFAILED")
        return 1

    print("\nPASSED")
    return 0

if __name__ == "__main__":
    sys.exit(main())
