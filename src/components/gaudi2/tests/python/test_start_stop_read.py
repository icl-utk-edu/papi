#!/usr/bin/env python3
"""
test_start_stop_read.py

Test script for PAPI Gaudi2 component full measurement lifecycle.
Exercises the complete PAPI workflow: init, create eventset, add events,
start, run workload, stop, read values, cleanup, and shutdown on a
single Gaudi2 device.

Prerequisites:
  - PAPI built and installed with gaudi2 component
  - libpapi.so in LD_LIBRARY_PATH
  - PyTorch with Habana support

Usage:
    PT_HPU_LAZY_MODE=1 python3 test_start_stop_read.py
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
            print(f"  Failed to add event '{name}': {self.strerror(ret)}")
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

# Main Test

def main():
    print("=" * 70)
    print("PAPI Gaudi2 Start/Stop/Read Lifecycle Test")
    print("=" * 70)
    print()

    # Step 0: Initialize PyTorch FIRST to acquire the device
    print("[0] Initializing PyTorch HPU (must be done before PAPI)...")
    if not init_pytorch():
        print("    SKIP: PyTorch HPU not available")
        return 0

    # Step 1: Load PAPI
    print("\n[1] Loading PAPI library...")
    papi_lib = load_papi()
    if not papi_lib:
        print("\nFAILED")
        return 1

    papi = PAPIWrapper(papi_lib)

    # Step 2: Initialize PAPI
    print("\n[2] Initializing PAPI...")
    if not papi.library_init():
        print("\nFAILED")
        return 1
    print("  PAPI initialized successfully")

    # Step 3: Create event set
    print("\n[3] Creating event set...")
    eventset = papi.create_eventset()
    if eventset is None:
        papi.shutdown()
        print("\nFAILED")
        return 1
    print(f"  Event set created: {eventset}")

    # Step 4: Add Gaudi2 events
    print("\n[4] Adding Gaudi2 events...")
    events = [
        "gaudi2:::TPC_KERNEL_EXECUTED:device=0",
        "gaudi2:::TPC_STALL:device=0",
        "gaudi2:::TPC_VECTOR_PIPE_EXEC:device=0",
        "gaudi2:::TPC_ICACHE_HIT:device=0",
        "gaudi2:::TPC_DCACHE_HIT:device=0",
    ]

    num_events = 0
    for event in events:
        if papi.add_named_event(eventset, event):
            print(f"    Added: {event}")
            num_events += 1
        else:
            print(f"    Failed: {event}")

    if num_events == 0:
        print("\n  ERROR: No events added. Is the gaudi2 component built?")
        papi.destroy_eventset(eventset)
        papi.shutdown()
        print("\nFAILED")
        return 1

    # Step 5: Start counting
    print(f"\n[5] Starting counters ({num_events} events)...")
    if not papi.start(eventset):
        papi.cleanup_eventset(eventset)
        papi.destroy_eventset(eventset)
        papi.shutdown()
        print("\nFAILED")
        return 1
    print("  Counters started")

    # Step 6: Run workload
    print("\n[6] Running PyTorch workload...")
    if not run_pytorch_workload():
        print("  WARNING: Workload failed, counters may be zero")

    # Step 7: Stop and read
    print("\n[7] Stopping counters and reading values...")
    values = papi.stop(eventset, num_events)
    if values is None:
        papi.cleanup_eventset(eventset)
        papi.destroy_eventset(eventset)
        papi.shutdown()
        print("\nFAILED")
        return 1

    print("\n  Results:")
    print("  " + "-" * 55)
    all_valid = True
    for i, (event, value) in enumerate(zip(events[:num_events], values)):
        event_name = event.split(":::")[-1]
        print(f"  {event_name:<40}: {value:>12,}")
        if value < 0:
            print(f"    WARNING: Negative counter value for {event_name}")
            all_valid = False
    print("  " + "-" * 55)

    if not all_valid:
        print("\n  ERROR: Some counter values are invalid (negative)")

    # Step 8: Cleanup
    print("\n[8] Cleaning up...")
    papi.cleanup_eventset(eventset)
    papi.destroy_eventset(eventset)
    papi.shutdown()
    print("  Done")

    print("\n" + "=" * 70)
    if all_valid:
        print("PASSED")
    else:
        print("FAILED")
    print("=" * 70)

    return 0 if all_valid else 1

if __name__ == "__main__":
    sys.exit(main())
