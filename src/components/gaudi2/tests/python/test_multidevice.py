#!/usr/bin/env python3
"""
test_multidevice.py

Test script for PAPI Gaudi2 component multi-device support.
Tests device qualifiers, monitoring the same event across multiple devices,
mixed events across devices, invalid device handling, and multiple reads
during workload execution.

Prerequisites:
  - PAPI built and installed with gaudi2 component
  - libpapi.so in LD_LIBRARY_PATH
  - PyTorch with Habana support
  - One or more Gaudi2 devices available

Usage:
    PT_HPU_LAZY_MODE=1 python3 test_multidevice.py
"""

import ctypes
import sys
import os

# PAPI Constants

PAPI_VER_CURRENT = 0x07030000
PAPI_NULL = -1
PAPI_OK = 0

# Load PAPI Library

def load_papi():
    """Load PAPI shared library"""
    lib_paths = [
        "libpapi.so",
    ]

    for path in lib_paths:
        try:
            papi = ctypes.CDLL(path)
            print(f"  Loaded PAPI from: {path}")
            return papi
        except OSError:
            continue

    print("  ERROR: Could not load libpapi.so")
    print("  Make sure PAPI is built with gaudi2 component and in LD_LIBRARY_PATH")
    return None

# PAPI Wrapper

class PAPIWrapper:
    """Wrapper for PAPI library functions using ctypes"""

    def __init__(self, papi):
        self.papi = papi
        self._setup_functions()

    def _setup_functions(self):
        """Setup ctypes function signatures"""
        self.papi.PAPI_library_init.argtypes = [ctypes.c_int]
        self.papi.PAPI_library_init.restype = ctypes.c_int

        self.papi.PAPI_create_eventset.argtypes = [ctypes.POINTER(ctypes.c_int)]
        self.papi.PAPI_create_eventset.restype = ctypes.c_int

        self.papi.PAPI_add_named_event.argtypes = [ctypes.c_int, ctypes.c_char_p]
        self.papi.PAPI_add_named_event.restype = ctypes.c_int

        self.papi.PAPI_start.argtypes = [ctypes.c_int]
        self.papi.PAPI_start.restype = ctypes.c_int

        self.papi.PAPI_stop.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_longlong)]
        self.papi.PAPI_stop.restype = ctypes.c_int

        self.papi.PAPI_read.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_longlong)]
        self.papi.PAPI_read.restype = ctypes.c_int

        self.papi.PAPI_reset.argtypes = [ctypes.c_int]
        self.papi.PAPI_reset.restype = ctypes.c_int

        self.papi.PAPI_cleanup_eventset.argtypes = [ctypes.c_int]
        self.papi.PAPI_cleanup_eventset.restype = ctypes.c_int

        self.papi.PAPI_destroy_eventset.argtypes = [ctypes.POINTER(ctypes.c_int)]
        self.papi.PAPI_destroy_eventset.restype = ctypes.c_int

        self.papi.PAPI_shutdown.argtypes = []
        self.papi.PAPI_shutdown.restype = None

        self.papi.PAPI_strerror.argtypes = [ctypes.c_int]
        self.papi.PAPI_strerror.restype = ctypes.c_char_p

    def library_init(self, version=PAPI_VER_CURRENT):
        """Initialize PAPI library"""
        ret = self.papi.PAPI_library_init(version)
        if ret != version:
            if ret > 0:
                print(f"  PAPI version mismatch: got 0x{ret:08x}, expected 0x{version:08x}")
            else:
                print(f"  PAPI init failed: {self.strerror(ret)}")
            return False
        return True

    def create_eventset(self):
        """Create a new event set"""
        eventset = ctypes.c_int(PAPI_NULL)
        ret = self.papi.PAPI_create_eventset(ctypes.byref(eventset))
        if ret != PAPI_OK:
            print(f"  Failed to create eventset: {self.strerror(ret)}")
            return None
        return eventset.value

    def add_named_event(self, eventset, name):
        """Add a named event to the event set"""
        ret = self.papi.PAPI_add_named_event(eventset, name.encode())
        if ret != PAPI_OK:
            print(f"    Failed to add event '{name}': {self.strerror(ret)}")
            return False
        return True

    def start(self, eventset):
        """Start counting"""
        ret = self.papi.PAPI_start(eventset)
        if ret != PAPI_OK:
            print(f"  Failed to start: {self.strerror(ret)}")
            return False
        return True

    def stop(self, eventset, num_events):
        """Stop counting and return values"""
        values = (ctypes.c_longlong * num_events)()
        ret = self.papi.PAPI_stop(eventset, values)
        if ret != PAPI_OK:
            print(f"  Failed to stop: {self.strerror(ret)}")
            return None
        return list(values)

    def read(self, eventset, num_events):
        """Read current counter values"""
        values = (ctypes.c_longlong * num_events)()
        ret = self.papi.PAPI_read(eventset, values)
        if ret != PAPI_OK:
            print(f"  Failed to read: {self.strerror(ret)}")
            return None
        return list(values)

    def reset(self, eventset):
        """Reset counters"""
        ret = self.papi.PAPI_reset(eventset)
        return ret == PAPI_OK

    def cleanup_eventset(self, eventset):
        """Cleanup event set"""
        self.papi.PAPI_cleanup_eventset(eventset)

    def destroy_eventset(self, eventset):
        """Destroy event set"""
        es = ctypes.c_int(eventset)
        self.papi.PAPI_destroy_eventset(ctypes.byref(es))

    def shutdown(self):
        """Shutdown PAPI"""
        self.papi.PAPI_shutdown()

    def strerror(self, code):
        """Get error string"""
        msg = self.papi.PAPI_strerror(code)
        return msg.decode() if msg else f"Unknown error {code}"

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

def detect_devices_by_probing(papi):
    """Detect the number of Gaudi2 devices that PAPI can actually use by
    attempting to add an event with increasing device indices.
    Not all /dev/accel/accel* files are necessarily Gaudi2 devices, so
    we probe PAPI directly to find the valid device count.
    Returns (num_devices, device_ids) where device_ids is a 0-based list."""
    device_ids = []
    for d in range(16):
        eventset = papi.create_eventset()
        if eventset is None:
            break
        event_name = f"gaudi2:::TPC_KERNEL_EXECUTED:device={d}"
        ret = papi.papi.PAPI_add_named_event(eventset, event_name.encode())
        if ret == PAPI_OK:
            device_ids.append(d)
            papi.cleanup_eventset(eventset)
        papi.destroy_eventset(eventset)
        if ret != PAPI_OK:
            break
    return len(device_ids), device_ids

def cleanup(papi, eventset):
    """Cleanup eventset and shutdown PAPI"""
    if eventset is not None:
        papi.cleanup_eventset(eventset)
        papi.destroy_eventset(eventset)

# Sub-test 1: Single event on a specific device

def test_single_device_qualifier(papi, device_id):
    """Test adding a single event with an explicit :device=N qualifier"""
    print(f"\n{'='*70}")
    print(f"Sub-test 1: Single event with :device={device_id} qualifier")
    print(f"{'='*70}")

    eventset = papi.create_eventset()
    if eventset is None:
        return False

    event_name = f"gaudi2:::TPC_KERNEL_EXECUTED:device={device_id}"
    print(f"  Adding event: {event_name}")
    if not papi.add_named_event(eventset, event_name):
        papi.destroy_eventset(eventset)
        return False
    print(f"  Event added successfully")

    print(f"  Starting counters...")
    if not papi.start(eventset):
        cleanup(papi, eventset)
        return False

    print(f"  Running workload...")
    run_pytorch_workload()

    values = papi.stop(eventset, 1)
    if values is None:
        cleanup(papi, eventset)
        return False

    print(f"\n  Result:")
    print(f"    TPC_KERNEL_EXECUTED:device={device_id} = {values[0]:>15,}")

    cleanup(papi, eventset)
    print(f"\n  PASSED")
    return True

# Sub-test 2: Same event across multiple devices

def test_same_event_multiple_devices(papi, device_ids):
    """Test monitoring the same event across all available devices"""
    print(f"\n{'='*70}")
    print(f"Sub-test 2: Same event across {len(device_ids)} devices {device_ids}")
    print(f"{'='*70}")

    eventset = papi.create_eventset()
    if eventset is None:
        return False

    event_names = []
    for d in device_ids:
        event_name = f"gaudi2:::TPC_KERNEL_EXECUTED:device={d}"
        print(f"  Adding: {event_name}")
        if not papi.add_named_event(eventset, event_name):
            cleanup(papi, eventset)
            return False
        event_names.append(event_name)

    num_events = len(event_names)
    print(f"\n  Starting counters on {num_events} devices...")
    if not papi.start(eventset):
        cleanup(papi, eventset)
        return False

    print(f"  Running workload...")
    run_pytorch_workload()

    values = papi.stop(eventset, num_events)
    if values is None:
        cleanup(papi, eventset)
        return False

    print(f"\n  Results (TPC_KERNEL_EXECUTED per device):")
    print(f"  {'Device':<10} {'Value':>15}")
    print(f"  {'-'*10} {'-'*15}")
    for i, d in enumerate(device_ids):
        print(f"  device={d:<5} {values[i]:>15,}")

    cleanup(papi, eventset)
    print(f"\n  PASSED")
    return True

# Sub-test 3: Different events across different devices

def test_mixed_events_devices(papi, device_ids):
    """Test monitoring different events on different devices"""
    print(f"\n{'='*70}")
    print(f"Sub-test 3: Mixed events across devices {device_ids}")
    print(f"{'='*70}")

    eventset = papi.create_eventset()
    if eventset is None:
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

    num_events = 0
    added_labels = []
    for full_name, label in events:
        print(f"  Adding: {full_name}")
        if papi.add_named_event(eventset, full_name):
            num_events += 1
            added_labels.append(label)
        else:
            print(f"    (skipped)")

    if num_events == 0:
        print("\n  ERROR: No events could be added")
        papi.destroy_eventset(eventset)
        return False

    print(f"\n  Starting counters ({num_events} events)...")
    if not papi.start(eventset):
        cleanup(papi, eventset)
        return False

    print(f"  Running workload...")
    run_pytorch_workload()

    values = papi.stop(eventset, num_events)
    if values is None:
        cleanup(papi, eventset)
        return False

    print(f"\n  Results:")
    print(f"  {'Event':<40} {'Value':>15}")
    print(f"  {'-'*40} {'-'*15}")
    for label, value in zip(added_labels, values):
        print(f"  {label:<40} {value:>15,}")

    cleanup(papi, eventset)
    print(f"\n  PASSED")
    return True

# Sub-test 4: Invalid device qualifier (negative test)

def test_invalid_device(papi, num_devices):
    """Test that adding an event with an invalid device index fails gracefully"""
    print(f"\n{'='*70}")
    print(f"Sub-test 4: Invalid device qualifier (negative test)")
    print(f"{'='*70}")

    eventset = papi.create_eventset()
    if eventset is None:
        return False

    invalid_device = num_devices
    event_name = f"gaudi2:::TPC_KERNEL_EXECUTED:device={invalid_device}"
    print(f"  Adding event with invalid device: {event_name}")

    ret = papi.papi.PAPI_add_named_event(eventset, event_name.encode())
    if ret != PAPI_OK:
        print(f"  Correctly rejected: {papi.strerror(ret)}")
        papi.destroy_eventset(eventset)
        print(f"\n  PASSED")
        return True
    else:
        print(f"  ERROR: Should have rejected device={invalid_device} but accepted it")
        cleanup(papi, eventset)
        return False

# Sub-test 5: Multiple reads during workload

def test_read_during_workload(papi, device_id):
    """Test reading counter values during workload execution"""
    print(f"\n{'='*70}")
    print(f"Sub-test 5: Multiple reads during workload on device={device_id}")
    print(f"{'='*70}")

    eventset = papi.create_eventset()
    if eventset is None:
        return False

    events = [
        f"gaudi2:::TPC_KERNEL_EXECUTED:device={device_id}",
        f"gaudi2:::TPC_VECTOR_PIPE_EXEC:device={device_id}",
    ]
    num_events = 0
    for event_name in events:
        print(f"  Adding: {event_name}")
        if papi.add_named_event(eventset, event_name):
            num_events += 1

    if num_events == 0:
        papi.destroy_eventset(eventset)
        return False

    print(f"\n  Starting counters...")
    if not papi.start(eventset):
        cleanup(papi, eventset)
        return False

    # Read before workload
    values_before = papi.read(eventset, num_events)
    print(f"  Before workload: {values_before}")

    # Run workload and read
    print(f"  Running workload...")
    run_pytorch_workload()

    values_after = papi.read(eventset, num_events)
    print(f"  After workload:  {values_after}")

    # Run another workload and stop
    print(f"  Running second workload...")
    run_pytorch_workload()

    values_final = papi.stop(eventset, num_events)
    print(f"  Final (stop):    {values_final}")

    if values_before and values_after and values_final:
        print(f"\n  Counter progression:")
        print(f"  {'Event':<30} {'Before':>12} {'After':>12} {'Final':>12}")
        print(f"  {'-'*30} {'-'*12} {'-'*12} {'-'*12}")
        labels = [e.split(":::")[-1] for e in events[:num_events]]
        for i, label in enumerate(labels):
            print(f"  {label:<30} {values_before[i]:>12,} {values_after[i]:>12,} {values_final[i]:>12,}")

    cleanup(papi, eventset)
    print(f"\n  PASSED")
    return True

# Main

def main():
    print("=" * 70)
    print("PAPI Gaudi2 Multi-Device Test Suite")
    print("=" * 70)

    # Initialize PyTorch FIRST (acquires device fd needed by PAPI)
    print("\n[SETUP] Initializing PyTorch HPU...")
    if not init_pytorch():
        print("  SKIP: PyTorch HPU not available")
        return 0

    # Load and initialize PAPI
    print("\n[SETUP] Loading PAPI library...")
    papi_lib = load_papi()
    if not papi_lib:
        print("\nFAILED")
        return 1

    papi = PAPIWrapper(papi_lib)

    print("\n[SETUP] Initializing PAPI...")
    if not papi.library_init():
        print("\nFAILED")
        return 1
    print("  PAPI initialized successfully")

    # Detect devices by probing PAPI (not all /dev/accel/* are Gaudi2)
    num_devices, device_ids = detect_devices_by_probing(papi)
    print(f"\n  Detected {num_devices} Gaudi2 device(s) usable by PAPI")
    print(f"  PAPI device indices: {device_ids}")
    if num_devices == 0:
        print("  SKIP: No Gaudi2 devices found via PAPI")
        papi.shutdown()
        return 0

    # Run sub-tests
    results = {}

    results["Sub-test 1: Single device qualifier"] = test_single_device_qualifier(papi, 0)

    if num_devices > 1:
        test_devs = device_ids[:4]  # Cap at 4 devices
        results["Sub-test 2: Same event multi-device"] = test_same_event_multiple_devices(
            papi, test_devs
        )
    else:
        print(f"\n  Skipping Sub-test 2 (need >1 device, have {num_devices})")
        results["Sub-test 2: Same event multi-device"] = None

    results["Sub-test 3: Mixed events/devices"] = test_mixed_events_devices(papi, device_ids)

    results["Sub-test 4: Invalid device"] = test_invalid_device(papi, num_devices)

    results["Sub-test 5: Read during workload"] = test_read_during_workload(papi, 0)

    # Summary
    print(f"\n{'='*70}")
    print("TEST SUMMARY")
    print(f"{'='*70}")
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
    print(f"{'='*70}")

    papi.shutdown()

    if failed > 0:
        print("\nFAILED")
        return 1

    print("\nPASSED")
    return 0

if __name__ == "__main__":
    sys.exit(main())
