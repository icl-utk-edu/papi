# NVML Component

The NVML (NVIDIA Management Library) component exposes hardware management
counters and controls for NVIDIA GPUs, such as power consumption, fan speed and
temperature readings; it also allows capping of power consumption.

* [Enabling the NVML Component](#enabling-the-nvml-component)
* [Known Limitations](#known-limitations)
* [FAQ](#faq)
***
## Enabling the NVML Component

To enable reading or writing of NVML counters the user needs to link
against a PAPI library that was configured with the `nvml` component. As an example:
```
./configure --with-components="nvml"
```

For the component to be active, PAPI requires one environment variable to be set: `PAPI_CUDA_ROOT` (exact same environment variable as the `cuda` component). This environment variable **must** be set to the root of the
Cuda Toolkit that is desired to be used. As as example:
```
export PAPI_CUDA_ROOT=/packages/cuda/#.#.#
```

Within `PAPI_CUDA_ROOT`, we expect the following standard directory for building:
```
PAPI_CUDA_ROOT/include
```

For the `nvml` component to be operational at runtime it must find the following dynamic shared library:
```
libnvidia-ml.so
```

To verify the `nvml` component was configured with your PAPI build and is active, run `papi_component_avail` (available in `utils/papi_component_avail`).
This PAPI utility will display the components configured in your PAPI build and whether they are active or disabled. If the component is the latter a disabled
reason will be provided.

## Known Limitations

* Some systems require `sudo` (superuser) status in order to set or read 
power limits; such permissions are typically granted by your sysadmin.
***

## FAQ

1. [Unusual installations](#unusual-installations)

## Unusual installations
The dynamic shared library `libnvidia-ml` is commonly found under `/lib` or `/usr/lib` rather than the Cuda Toolkit installation. Therefore, by default `dlopen` is used to search `/lib` and `/usr/lib`.
If the dynamic shared library is not found, then two other options remain:

1. Setting the dynamic shared libraries corresponding environment variable:
   ```
   export PAPI_NVML_MAIN=/your/path/to/libnvidia-ml.so
   ```

   Note, that if using this option:
     * You must set the environment variable directly to the dynamic shared library as shown above.
     * If the set path fails to open a dynamic shared library then the `nvml` component will be disabled.

2. `dlopen` follows the search logic of the dynamic shared linker and so you can set `LD_LIBRARY_PATH` to the directory containing your `libnvidia-ml` dynamic shared library, i.e.
   ```
   export LD_LIBRARY_PATH=/your/path/to/WhereLibCanBeFound:$LD_LIBRARY_PATH
   ```
