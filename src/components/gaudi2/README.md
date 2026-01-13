# PAPI Gaudi2 Component

This PAPI component provides access to hardware performance counters on Intel Gaudi2 AI Accelerators through the SPMU interface.

## Overview

The Gaudi2 component enables monitoring of:
- **TPC (Tensor Processing Core)** - 24 TPCs across 4 DCOREs
- **EDMA (External DMA)** - 8 EDMAs for data movement
- **PDMA (PCIe DMA)** - 2 PDMAs for host-device transfers
- **MME (Matrix Multiplication Engine)** - 4 MMEs for matrix operations

Each SPMU unit supports up to 6 programmable counters that can be configured to count various hardware events.

## Requirements

- **Hardware**: Intel Gaudi2 AI Accelerator
- **Software**:
  - Habana Labs driver and runtime
  - libhl-thunk.so (Habana thunk library)
  - Access to `/dev/accel/accel*` devices
- **Permissions**: User must have read/write access to accelerator devices

## Building

Set the `PAPI_GAUDI2_ROOT` environment variable to habanalabs installed directory for `hl-thunk` headers and libraries.
`export PAPI_GAUDI2_ROOT=/usr`

Configure the component using:
`./configure --with-components="gaudi2"`

then build with:
`make && make install`
