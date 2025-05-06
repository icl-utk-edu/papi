# AMD_SMI Component

The **AMD_SMI** (AMD System Management Interface) component exposes hardware
management counters (and selected controls) for AMD GPUs — e.g., power usage,
temperatures, clocks, PCIe link metrics, VRAM information, and RAS/ECC status —
by querying the AMD SMI library at runtime (ROCm ≥ 6.3.4).

- [Environment Variables](#environment-variables)
- [Enabling the AMD_SMI Component](#enabling-the-amd_smi-component)

---

## Environment Variables

For AMD_SMI, PAPI requires the environment variable `PAPI_AMDSMI_ROOT` to be set
so that the AMD SMI shared library and headers can be found. This variable is
required at both **compile** and **run** time.

There is a single case to consider (AMD SMI is available on ROCm ≥ 6.0):

1. **For ROCm versions 6.0 and newer:**  
   Set `PAPI_AMDSMI_ROOT` to the top-level ROCm directory. For example:

   ```bash
   export PAPI_AMDSMI_ROOT=/opt/rocm-6.4.0
   # or
   export PAPI_AMDSMI_ROOT=/opt/rocm
   ```

The directory specified by `PAPI_AMDSMI_ROOT` **must contain** the following
subdirectories:

- `PAPI_AMDSMI_ROOT/lib` (which should include the dynamic library `libamd_smi.so`)
- `PAPI_AMDSMI_ROOT/include/amd_smi` (AMD SMI headers)

If the library is not found or is not functional at runtime, the component will
appear as "disabled" in `papi_component_avail`, with a message describing the
problem (e.g., library not found).

---

## Enabling the AMD_SMI Component

To enable reading (and where supported, writing) of AMD_SMI counters, build
PAPI with this component enabled. For example:

```bash
./configure --with-components="amd_smi"
make
```

You can verify availability with the utilities in `papi/src/utils/`:

```bash
papi_component_avail            # shows enabled/disabled components
papi_native_avail -i amd_smi    # lists native events for this component
```

---

## File-by-file Summary

- **`linux-amd-smi.c`**  
  Declares the `papi_vector_t` for this component; initializes on first use; hands off work to `amds_*` for device/event management; implements PAPI hooks (`init_component`, `update_control_state`, `start`, `read`, `stop`, `reset`, `shutdown`, and native-event queries).

- **`amds.c`**  
  Dynamically loads `libamd_smi.so`, resolves AMD SMI symbols, discovers sockets/devices, and **builds the native event table**. Defines helpers to add simple and counter-based events. Manages global teardown (destroy event table, close library).

- **`amds_accessors.c`**  
  Implements the **accessors** that read/write individual metrics (e.g., temperatures, fans, PCIe, energy, power caps, RAS/ECC, clocks, VRAM, link topology, XGMI/PCIe metrics, firmware/board info, etc.). Each accessor maps an event’s `(variant, subvariant)` to the right SMI call and returns the value.

- **`amds_ctx.c`**  
  Provides the **per-eventset context**:  
  - `amds_ctx_open/close` — acquire/release devices, run per-event open/close hooks.  
  - `amds_ctx_start/stop` — start/stop counters where needed.  
  - `amds_ctx_read/write/reset` — read current values, optionally write supported controls (e.g., power cap), zero software view.  

- **`amds_evtapi.c`**  
  Implements native-event enumeration for PAPI (`enum`, `code_to_name`, `name_to_code`, `code_to_descr`) using the in-memory event table and a small hash map for fast lookups.

- **`amds_priv.h`**  
  Internal definitions: `native_event_t` (name/descr/device/mode/value + open/close/start/stop/access callbacks), global getters, and the AMD SMI function-pointer declarations (via `amds_funcs.h`).

- **`amds_funcs.h`**  
  Centralized macro list of **AMD SMI APIs** used by the component; generates function-pointer declarations/definitions so `amds.c` can `dlsym()` them at runtime. Conditional entries handle newer SMI features.

- **`htable.h`**  
  Minimal chained hash table for **name→event** mapping; used by `amds_evtapi.c` to resolve native event names quickly.

- **`amds.h`**  
  Public, component-internal API across files: init/shutdown, native-event queries, context ops, and error-string retrieval.

- **`Rules.amd_smi`**  
  Build integration for PAPI’s make system; compiles this component and sets include/library paths for AMD SMI.
