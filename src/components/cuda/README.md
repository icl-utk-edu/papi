# Cuda Component

The `cuda` component exposes counters and controls for NVIDIA GPUs.

* [Enabling the Cuda Component](#enabling-the-cuda-component)
* [Hardware and Software Support](#hardware-and-software-support)
* [Partially Disabled Cuda Component](#partially-disabled-cuda-component)
* [Known Limitations](#known-limitations)
* [FAQ](#faq)
***

## Enabling the `cuda` Component

To enable reading or writing of CUDA counters the user needs to link against a
PAPI library that was configured with the `cuda` component. As an example:
```
./configure --with-components="cuda"
```

For the component to be active, PAPI requires one environment variable to be set: `PAPI_CUDA_ROOT`. This environment variable **must** be set to the root of the Cuda Toolkit that is desired to be used for both compiling and runtime. As an example:

```
PAPI_CUDA_ROOT=/packages/cuda/#.#.#
```

Within `PAPI_CUDA_ROOT`, we expect the following standard directories for building:

```
PAPI_CUDA_ROOT/include
PAPI_CUDA_ROOT/extras/CUPTI/include
```

and for runtime:

```
PAPI_CUDA_ROOT/lib64
PAPI_CUDA_ROOT/extras/CUPTI/lib64
```

To verify the `cuda` component was configured with your PAPI build and is active,
run `papi_component_avail` (available in `utils/papi_component_avail`). This 
utility will display the components configured in your PAPI build and whether they are active or disabled. If a component is disabled a message on why the component
has been disabled will be directly below it.

For the `cuda` component to be operational, the following dynamic libraries must be found at runtime for both the Event/Metric APIs and the PerfWorks Metrics API:

```
libcuda.so
libcudart.so
libcupti.so
```

For the PerfWorks Metrics API, the following dynamic library must also be found:
```
libnvperf_host.so
```

## Hardware and Software Support

To see the `cuda` component's current supported hardware and software please visit the GitHub wiki page [Hardware and Software Support - Cuda Component](https://github.com/icl-utk-edu/papi/wiki/Hardware-and-Software-Support-%E2%80%90-Cuda-Component).

## Partially Disabled Cuda Component
As shown on the GitHub wiki page [Hardware and Software Support - Cuda Component](https://github.com/icl-utk-edu/papi/wiki/Hardware-and-Software-Support-%E2%80%90-Cuda-Component) the `cuda` component supports a total of three primary APIs
to expose counters and controls for NVIDIA GPUs.

The Event/Metric APIs (Legacy) only overlaps with the PerfWorks Metrics API at CC 7.0 (V100). Meaning in the case of machines with NVIDIA GPUs with mixed compute capabilities e.g. P100 - CC 6.0 and A100 - CC 8.0 a choice must be made for which CCs the counters and controls will be exposed for.

To allow for this choice to be made the `cuda` component supports being ***Partially Disabled***. Which means:

* If exposing counters and controls for CCs <= 7.0 (e.g. P100 and V100), then support for exposing counters and controls for CCs > 7.0 will be disabled
* If exposing counters and controls for CCs >= 7.0 (e.g. V100 and A100), then support for exposing counters and controls for CCs < 7.0 will be disabled

By default on mixed compute capability machines, counters and controls for CCs >= 7.0 will be exposed. However, at runtime the choice of which CCs the counter and controls will be exposed for can be changed via the environment variable `PAPI_CUDA_API`. Simply
set `PAPI_CUDA_API` equal to `LEGACY`, e.g:

```
export PAPI_CUDA_API=LEGACY
```

Important note, in the case of machines that only have GPUs with CCs = 7.0 there will be no partially disabled Cuda component. Counter and controls will be exposed via the PerfWorks Metrics API; however, if you would like to expose counters and controls via the Legacy APIs please see the aforementioned environment variable.

## Known Limitations
* Exposing counters on machines that have NVIDIA GPUs with CCS >= 7.0 is done via the Pefworks API. This API vastly expands the number of possible counters from roughly a few hundred to over 140,000 per GPU. Due to this, the PAPI utility `utils/papi_native_avail` may take a few minutes to run (as much as 2 minutes per GPU). If the output from `utils/papi_native_avail` is redirected to a file, it may appear as if it has "hung"; however, give it time and it will complete.
***

## FAQ

1. [Unusual installations](#unusual-installations)
2. [CUDA contexts](#cuda-contexts)
3. [CUDA toolkit versions](#cuda-toolkit-versions)

## Unusual installations
If the dynamic libraries `libcupti`, `libnvperf_host`, and `libcudart` cannot be found by setting `PAPI_CUDA_ROOT` then two other options remain to find them:

1. Setting the dynamic libraries corresponding environment variable:
   ```
   export PAPI_CUDA_CUPTI=/your/path/to/libcupti.so
   export PAPI_CUDA_PERFWORKS=/your/path/to/libnvperf_host.so
   export PAPI_CUDA_RUNTIME=/your/path/to/libcudart.so
   ```

   Note, that if using this option:
     * You must set the enviornment variable directly to the dynamic library as shown above.
     * If the set path fails to open a dynamic library, the `cuda` component will be disabled.

2. Using `dlopen` and following the search logic used by the dynamic linker. For this option, it is advised to set `LD_LIBRARY_PATH` to the directories containing your `libcupti`, `libnvperf_host`, and `libcudart` dynamic libraries, i.e.
   ```
   export LD_LIBRARY_PATH=/your/path/to/WhereLib1CanBeFound:/your/path/to/WhereLib2CanBeFound:$LD_LIBRARY_PATH
   ```

   Note, that if using this option:
     * Make sure to separate the dynamic libraries by a colon (`:`).
     * This option serves as a final fallback if either `PAPI_CUDA_ROOT` is not set or is unable to load one of the dynamic libraries (`libcupti`, `libnvperf_host`, and `libcudart`).

* If CUDA libraries are installed on your system, such that the OS can find `nvcc`, the header files, and the shared libraries, then `PAPI_CUDA_ROOT` and `LD_LIBRARY_PATH` may not be necessary.

## CUDA contexts
The CUDA component can profile using contexts created by `cuCtxCreate` or primary device contexts activated by `cudaSetDevice`. Refer to test codes `HelloWorld`, `simpleMultiGPU`, `pthreads`, etc, that use created contexts. Refer to corresponding `*_noCuCtx` tests for profiling using primary device contexts.

## CUDA toolkit versions
Once your binaries are compiled, it is possible to swap the CUDA toolkit versions without needing to recompile the source. Simply update `PAPI_CUDA_ROOT` to point to the path where the cuda toolkit version can be found. You might need to update `LD_LIBRARY_PATH` as well.
